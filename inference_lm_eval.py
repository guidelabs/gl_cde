from pathlib import Path
from typing import Any
import json
import time
import asyncio

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import postprocess_generated_text
from tqdm.asyncio import tqdm_asyncio
import torch

from gl_cde import CDESettings, Metrics, SamplingParams, average_metrics, init_engine, init_tokenizer
from gl_cde.metrics import timed_generate


def init_throughput_cde_config(max_num_seqs: int) -> CDESettings:
    return CDESettings(
        endofchunk_token_id=100279,
        mask_token_id=100280,
        banned_ids=[100280, 100277],
        eos_token_id=100257,
        eot_token_id=100283,
        steer_suppress_alpha=11,
        steer_amplify_alpha=8,
        warmup_model=True,
        compile_model=True,
        max_num_seqs=max_num_seqs,
        device=torch.device("cuda:0"),
    )


@register_model("modal")
class ModalLMEval(LM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(self, batch_size: str = "1", device: str = "cuda", progress: bool = True) -> None:  # noqa: FBT002 FBT001
        super().__init__()
        self.progress = progress
        self.model_name = "guidelabs/CDLM-8B-instruct"
        self.batch_size = int(batch_size)
        self.warmup_num = 10
        self.llm = init_engine(init_throughput_cde_config(self.batch_size))
        self.tokenizer = init_tokenizer()
        self.requests = {}
        with Path(__file__).with_name("gsm8k_processed.jsonl").open() as rows:
            for line in rows:
                # gsm8k_processed.jsonl spells the chunk marker <|end_of_chunk|>;
                # the tokenizer expects <|endofchunk|>
                row = json.loads(line.replace("<|end_of_chunk|>", "<|endofchunk|>"))
                self.requests[row["doc_id"]] = row

    @property
    def eot_token_id(self) -> int:
        raise NotImplementedError("eot_token_id is not implemented")

    @property
    def tokenizer_name(self) -> str:
        return self.model_name

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented")

    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:  # noqa: FBT002 FBT001
        raise NotImplementedError("loglikelihood_rolling is not implemented")

    def apply_chat_template(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,  # noqa: FBT002 FBT001
    ) -> list[dict[str, str]]:
        return chat_history

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:  # noqa: FBT002 FBT001
        return asyncio.run(self.async_generate_until(requests, disable_tqdm))

    def tokenize(self, messages: list[dict[str, str]]) -> list[int]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert isinstance(prompt, str)
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    async def async_generate(
        self, ids: list[int], sp: SamplingParams, sema: asyncio.Semaphore
    ) -> tuple[list[int], Metrics]:
        async with sema:
            return await timed_generate(self.llm, ids, sp)

    def detokenize(self, generation: list[int], gen_kwargs: dict[str, Any]) -> str:
        generated_text = self.tokenizer.decode(generation)
        assert isinstance(generated_text, str)
        return postprocess_generated_text(generated_text, gen_kwargs["until"], think_end_token=None)

    async def loop(self, req: Instance, sema: asyncio.Semaphore) -> tuple[str, Metrics]:
        assert req.doc_id is not None
        gen_kwargs = req.arguments[1]
        row = self.requests[req.doc_id]
        if row["task_name"] != req.task_name:
            raise ValueError(f"gsm8k_processed.jsonl has prompts for {row['task_name']}, not {req.task_name}")
        ids = self.tokenize(row["messages"])
        sp = SamplingParams(max_tokens=gen_kwargs.get("max_gen_toks", 256), seed=gen_kwargs.get("seed", 0))
        generation, metrics = await self.async_generate(ids, sp, sema)
        generated_text = self.detokenize(generation, gen_kwargs)
        return generated_text, metrics

    async def async_generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:  # noqa: FBT002 FBT001
        async with self.llm:
            # wanna limit concurrency for accurate ttfb ttlb latency measurement
            sema = asyncio.Semaphore(self.llm.cde_config.max_num_seqs)
            _ = await asyncio.gather(*[self.loop(req, sema) for req in requests[: self.warmup_num]])
            start_time = time.perf_counter()
            outputs = await tqdm_asyncio.gather(
                *[self.loop(req, sema) for req in requests], disable=disable_tqdm or not self.progress
            )
            generated_answers = [output[0] for output in outputs]
            metrics = [output[1] for output in outputs]
            final_time = time.perf_counter()
            num_tokens, ttfb, ibl, ttlb = average_metrics(metrics)
            print(f"requests:   {len(metrics)}")
            print(f"num_tokens: {num_tokens}")
            print(f"ttfb:       {ttfb:.2f} s")
            print(f"ibl:        {ibl:.2f} s")
            print(f"ttlb:       {ttlb:.2f} s")
            print(f"total time: {final_time - start_time:.2f} s")
            print(f"throughput: {num_tokens / (final_time - start_time):.1f} tok/s")
            return generated_answers


if __name__ == "__main__":
    cli_evaluate()
