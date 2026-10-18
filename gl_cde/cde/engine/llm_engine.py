from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from types import TracebackType
from typing import Self, final
import asyncio

from ...async_torch import AsyncTorchRunner
from ...cdlm import InterpretableCausalDiffusionLMBackbone
from ..attribution_types import ChunkedAttribution
from ..configuration_cdlm import CDESettings
from ..sampling_params import SamplingParams
from .model_runner import ModelRunnerForDiffusionLM
from .sampler import SamplerForCDLM
from .scheduler import SchedulerForDiffusionLM
from .sequence import RequestIdType, RequestOutput, SequenceForDiffusionLM


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class CDEOutput:
    tokens: list[int]
    finished_generation: bool
    concepts: list[ChunkedAttribution]


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class CDERequest:
    request_id: RequestIdType
    prompt: list[int]
    sampling_params: SamplingParams
    queue: asyncio.Queue[CDEOutput | BaseException] = field(default_factory=asyncio.Queue, init=False)


@final
class _RunnerThread:
    def __init__(
        self,
        cde_config: CDESettings,
        model: InterpretableCausalDiffusionLMBackbone,
        sampler: SamplerForCDLM,
    ) -> None:
        self._cde_config = cde_config
        self._model = model
        self._sampler = sampler
        self._model_runner: ModelRunnerForDiffusionLM | None = None
        self._scheduler: SchedulerForDiffusionLM | None = None
        self._sequences: dict[RequestIdType, SequenceForDiffusionLM] = {}

    @property
    def model_runner(self) -> ModelRunnerForDiffusionLM:
        if self._model_runner is None:
            raise RuntimeError("did not call load_model")
        return self._model_runner

    @property
    def scheduler(self) -> SchedulerForDiffusionLM:
        if self._scheduler is None:
            raise RuntimeError("did not call load_model")
        return self._scheduler

    def load_model(self, safetensor_path: Path) -> None:
        # we construct the SchedulerForDiffusionLM and ModelRunnerForDiffusionLM here,
        # because they do cuda allocations. Idearly, they would not and we can just call
        # initialize for the cuda allocations.
        self._scheduler = SchedulerForDiffusionLM(
            self._cde_config,
            self._model.model_config,
            self._model.concept_config,
        )
        self._model_runner = ModelRunnerForDiffusionLM(
            self._scheduler.page, self._cde_config, self._sampler, self._model
        )
        self._model_runner.initialize(safetensor_path)

    def execute(
        self, request_add: Sequence[CDERequest], request_remove: Sequence[CDERequest]
    ) -> tuple[list[RequestOutput], bool]:
        for request in request_remove:
            self.scheduler.remove(self._sequences[request.request_id])
            del self._sequences[request.request_id]

        for request in request_add:
            # we construct the SequenceForDiffusionLM here, because it does cuda allocations
            seq = SequenceForDiffusionLM(
                request.request_id,
                request.prompt,
                request.sampling_params,
                cde_config=self._cde_config,
                model_config=self._model.model_config,
                concept_config=self._model.concept_config,
            )
            # we add the sequence to the scheduler here, because it does cuda allocations
            self.scheduler.add(seq)
            self._sequences[request.request_id] = seq

        # perform actual inference
        outputs: list[RequestOutput] = []
        if not self.scheduler.is_finished():
            seqs, is_prefill = self.scheduler.schedule()
            sample_output = self.model_runner.run(seqs, is_prefill=is_prefill)
            stream_ready_seqs = self.scheduler.postprocess(seqs, sample_output)
            outputs = [seq.request_output() for seq in stream_ready_seqs]
        return (outputs, self.scheduler.is_finished())

    def unload_model(self) -> None:
        self.model_runner.release_gpu_memory()
        self._scheduler = None
        self._model_runner = None


@final
class LLMEngine(AbstractAsyncContextManager["LLMEngine", None]):
    def __init__(
        self,
        cde_config: CDESettings,
        model: InterpretableCausalDiffusionLMBackbone,
        sampler: SamplerForCDLM,
    ) -> None:
        self.cde_config = cde_config
        self._model = model
        self._torch_runner = AsyncTorchRunner(
            _RunnerThread(cde_config, model, sampler),
            cde_config.device,
        )
        self._counter = count()

        self._loop_task: asyncio.Task[None] | None = None
        self._loop_resume = asyncio.Event()

        self._pending_result: dict[RequestIdType, CDERequest] = {}
        self._pending_add: dict[RequestIdType, CDERequest] = {}
        self._pending_remove: dict[RequestIdType, CDERequest] = {}

    async def __aenter__(self) -> Self:
        safetensor_path = await self._model.prepare()
        await self._torch_runner.context_args(safetensor_path).__aenter__()
        self._loop_task = asyncio.create_task(self._loop())
        return self

    async def _loop(self) -> None:
        try:
            finished = True

            while True:
                if (
                    finished
                    and len(self._pending_add.values()) == 0
                    and len(self._pending_remove.values()) == 0
                ):
                    await self._loop_resume.wait()

                # get requests
                request_add = list(self._pending_add.values())
                request_remove = list(self._pending_remove.values())

                # reset request que
                self._pending_add.clear()
                self._pending_remove.clear()
                self._loop_resume.clear()

                # process — failures are fanned out by _on_loop_done
                outputs, finished = await self._torch_runner.execute(request_add, request_remove)
                for output in outputs:
                    pending = self._pending_result.get(output.request_id)
                    # The caller can leave during execute(). That step was already dispatched, so it
                    # still returns an output, and finally has moved the request to _pending_remove
                    # for the next iteration. An output with no listener and no queued removal means
                    # the scheduler kept stepping a request that was never cancelled.
                    if pending is None:
                        if output.request_id not in self._pending_remove:
                            raise RuntimeError(
                                f"output for request {output.request_id} "
                                "with no listener and no pending removal"
                            )
                        continue
                    pending.queue.put_nowait(
                        CDEOutput(
                            tokens=output.token_ids,
                            finished_generation=output.finished,
                            concepts=output.concepts,
                        )
                    )

        except BaseException as exc:
            for pending_result in self._pending_result.values():
                pending_result.queue.put_nowait(exc)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        await self._torch_runner.__aexit__(exc_type, exc_value, traceback)

    async def generate(
        self,
        prompt: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[CDEOutput]:
        if self._loop_task is None or self._loop_task.done():
            raise RuntimeError("The LLMEngine is not running")

        # add request
        request = CDERequest(
            request_id=RequestIdType(next(self._counter)), prompt=prompt, sampling_params=sampling_params
        )
        self._pending_add[request.request_id] = request
        self._pending_result[request.request_id] = request
        self._loop_resume.set()

        try:
            # get results
            while True:
                result = await request.queue.get()
                if isinstance(result, BaseException):
                    raise result

                yield result
                if result.finished_generation:
                    break

        finally:
            # remove request
            if request.request_id in self._pending_result:
                del self._pending_result[request.request_id]
            if request.request_id in self._pending_add:
                del self._pending_add[request.request_id]
            else:
                # only if we started processing, do we need to schedule the remove
                self._pending_remove[request.request_id] = request
                self._loop_resume.set()
