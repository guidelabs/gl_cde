from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, Self, final
import queue
import asyncio
import threading
import contextvars

import torch


class _TorchThreadInterface[**LoadT, **InT, OutT](Protocol):
    """Defines and executes a model in a separate thread.

    Warning:
        You can define basic initalization here.
        However, not create any cuda tensors here, or do any action involving cuda. Doing so will result in
        undefined behavior.
    """

    def load_model(self, *args: LoadT.args, **kwargs: LoadT.kwargs) -> None:
        """Load the model here.

        Args:
            *args: positional arguments forwarded from AsyncTorchRunner.context_args
            **kwargs: keyword arguments forwarded from AsyncTorchRunner.context_args
        """
        ...

    def execute(self, *args: InT.args, **kwargs: InT.kwargs) -> OutT:
        """Runs the batch.

        Note that the input must be on a cpu device and the output must be on a cpu device. Consider
        using numpy arrays to be safe.

        Args:
            *args: positional arguments forwarded from AsyncTorchRunner.execute
            **kwargs: keyword arguments forwarded from AsyncTorchRunner.execute

        Returns:
            OutT: will be the output of AsyncTorchRunner.execute
        """
        ...

    def unload_model(self) -> None:
        """Delete the model here."""
        ...


@dataclass(slots=True, frozen=True, eq=False)
class _LoadModelArgs:
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True, eq=False)
class _ExecuteJob[OutT]:
    args: tuple[object, ...]
    kwargs: dict[str, object]
    task: _ContextThreadTask[OutT]


@dataclass(slots=True, frozen=True, eq=False)
class _StopJob:
    task: _ContextThreadTask[None]


type _WorkerJob[OutT] = _ExecuteJob[OutT] | _StopJob


def _set_exception_if_pending[T](future: asyncio.Future[T], exc: BaseException) -> None:
    if not future.done():
        future.set_exception(exc)


def _set_result_if_pending[T](future: asyncio.Future[T], result: T) -> None:
    if not future.done():
        future.set_result(result)


@final
class _ContextThreadTask[R]:
    __slots__ = ("_context", "_future", "_loop")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._future: asyncio.Future[R] = loop.create_future()
        self._context = contextvars.copy_context()

    def run[**P](self, _fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> bool:
        """Run the task in the context, and forward results and exceptions.

        Returns:
            bool: returns true if sucessful (not exception)
        """
        try:
            res = self._context.run(_fn, *args, **kwargs)
        except Exception as exc:
            self._loop.call_soon_threadsafe(_set_exception_if_pending, self._future, exc)
            return False
        else:
            self._loop.call_soon_threadsafe(_set_result_if_pending, self._future, res)
            return True

    async def result(self, thread_future: asyncio.Future[None]) -> R:
        """Get the result of the future or raise when thread_future raises.

        Args:
            thread_future (asyncio.Future[None]): The future object for entire thread.

        Returns:
            R: results of the task
        """
        # The cancel event does not propagate to the self._future,
        # so this ensures the propagation.
        try:
            done, _ = await asyncio.wait((self._future, thread_future), return_when="FIRST_COMPLETED")
        except asyncio.CancelledError:
            self.cancel()
            raise

        # check for execeptions
        exceptions: list[BaseException] = []
        for future in done:
            if not future.cancelled() and (exc := future.exception()):
                exceptions.append(exc)

        if len(exceptions) == 1:
            raise exceptions[0]
        if len(exceptions) > 1:
            raise BaseExceptionGroup("multiple exceptions from thread occured", exceptions)

        # no exceptions, then get result
        if self._future in done:
            return self._future.result()

        raise RuntimeError("thread-future had no error but returned before task future")

    def cancel(self) -> None:
        self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()


def _create_torch_stream(torch_device: torch.device) -> torch.cuda.StreamContext | nullcontext[None]:
    if torch_device.type == "cuda":
        return torch.cuda.stream(torch.cuda.Stream(device=torch_device))
    return nullcontext()


def _torch_thread[**LoadT, **InT, OutT](
    thread_future: asyncio.Future[None],
    model_runner: _TorchThreadInterface[LoadT, InT, OutT],
    torch_device: torch.device,
    queue: queue.Queue[_WorkerJob[OutT]],
    startup_task: _ContextThreadTask[None],
    load_args: _LoadModelArgs,
) -> None:
    try:
        stop_job: _StopJob | None = None
        # set the device
        with torch.device(torch_device):
            # create a seperate cuda stream, such computation may run free of all other cuda computation
            with _create_torch_stream(torch_device):
                # load_model, if it fails stop the thread
                if not startup_task.run(model_runner.load_model, *load_args.args, **load_args.kwargs):  # pyright: ignore[reportCallIssue]
                    return

                # main task loop
                while True:
                    job = queue.get()

                    # if a _StopJob task was retrived, then stop the loop
                    if isinstance(job, _StopJob):
                        stop_job = job
                        break

                    if job.task.cancelled():
                        queue.task_done()
                        continue

                    # an _ExecuteJob task, so execute the job
                    job.task.run(model_runner.execute, *job.args, **job.kwargs)  # pyright: ignore[reportCallIssue]
                    queue.task_done()

                # completed main exceution, finalize with the unload_model
                stop_job.task.run(model_runner.unload_model)
                queue.task_done()
    except BaseException as exc:
        # bare threads do not provide a mecanism of forwarding errors, so use a thread_future for that.
        thread_future.get_loop().call_soon_threadsafe(_set_exception_if_pending, thread_future, exc)
    else:
        thread_future.get_loop().call_soon_threadsafe(_set_result_if_pending, thread_future, None)


@final
class AsyncTorchRunner[**LoadT, **InT, OutT](AbstractAsyncContextManager[None]):
    def __init__(
        self, model_runner: _TorchThreadInterface[LoadT, InT, OutT], torch_device: torch.device
    ) -> None:
        """Run a blocking PyTorch model runner on a dedicated worker thread.

        The intended usage is:

            runner = AsyncTorchRunner(MyModelRunner(), cuda_device="cuda:0")
            async with runner:
                result = await runner.execute(batch)

        Args:
            model_runner (_ModelRunnerInterface[InT, OutT]): defines how to load, execute, and unload model.
            torch_device (torch.device): which cuda device to use, must be fully specified.
        """
        if torch_device.type == "cuda" and torch_device.index is None:
            raise ValueError("cuda device must have an index specified")

        self._model_runner = model_runner
        self._torch_device = torch_device
        self._jobs: queue.Queue[_WorkerJob[OutT]] = queue.Queue()

        # thread_future will monitor for uncaught errors
        self._load_args: _LoadModelArgs = _LoadModelArgs()
        self._thread_future: asyncio.Future[None] | None = None
        self._thread: threading.Thread | None = None

    def context_args(self, *args: LoadT.args, **kwargs: LoadT.kwargs) -> Self:
        """Allow passing arguments to load_model.

        Example:
          runner = AsyncTorchRunner(MyRunnerInterface, torch.device('cuda:0'))
          async with runner.context_args(safetensor_path):
            result = await runner.execute(batch)

        Returns:
            Self: This runner with load arguments stored for the next context entry.
        """
        self._load_args = _LoadModelArgs(args, kwargs)
        return self

    async def __aenter__(self) -> None:
        if self._thread is not None:
            raise RuntimeError("AsyncTorchRunner is already running.")
        if not self._jobs.empty():
            raise RuntimeError("Expected job queue to be empty")

        loop = asyncio.get_running_loop()
        startup_task: _ContextThreadTask[None] = _ContextThreadTask(loop)
        self._thread_future = loop.create_future()

        # start new thread
        self._thread = threading.Thread(
            target=_torch_thread,
            name=type(self._model_runner).__qualname__,
            args=(
                self._thread_future,
                self._model_runner,
                self._torch_device,
                self._jobs,
                startup_task,
                self._load_args,
            ),
            daemon=False,
        )
        self._thread.start()

        # await startup completion
        await startup_task.result(self._thread_future)
        return

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._thread is None or self._thread_future is None:
            return
        thread = self._thread
        self._thread = None

        # cancel all pending tasks
        while not self._jobs.empty():
            job = self._jobs.get_nowait()
            job.task.cancel()
            self._jobs.task_done()

        # submit shutdown task
        loop = asyncio.get_running_loop()
        shutdown_task: _ContextThreadTask[None] = _ContextThreadTask(loop)
        self._jobs.put(_StopJob(shutdown_task))

        # await shutdown completion
        await shutdown_task.result(self._thread_future)
        await asyncio.to_thread(self._jobs.join)
        await asyncio.to_thread(thread.join)

    async def execute(self, *args: InT.args, **kwargs: InT.kwargs) -> OutT:
        """Schedule the batch to run in the thread.

        Notes:
        * The input must be on a cpu device and the output must be on a cpu device. Consider using numpy
          arrays to be safe.
        * One one input

        Args:
            *args: input argument to model_runner.execute
            **kwargs: input argument to model_runner.execute

        Returns:
            OutT: will be the output of model_runner.execute
        """
        if self._thread is None or self._thread_future is None or not self._thread.is_alive():
            raise RuntimeError("Thread is not running.")

        # submit job to queue, note that it might not start executing immediately.
        loop = asyncio.get_running_loop()
        task: _ContextThreadTask[OutT] = _ContextThreadTask(loop)
        self._jobs.put(_ExecuteJob(args, kwargs, task))
        return await task.result(self._thread_future)
