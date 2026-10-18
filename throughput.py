import time
import random
import asyncio
import argparse

import torch

from gl_cde import CDESettings, LLMEngine, Metrics, SamplingParams, average_metrics, init_engine
from gl_cde.metrics import timed_generate

# token ids at or above this are special tokens; keep random prompts below it
VOCAB_SIZE = 100256


def init_throughput_cde_config(max_num_seqs: int) -> CDESettings:
    return CDESettings(
        endofchunk_token_id=100279,
        mask_token_id=100280,
        banned_ids=[100280, 100277],
        # disable any possibility of ending a sequence or a block early to help measure throughput/latency
        eos_token_id=-1,
        eot_token_id=-1,
        steer_suppress_alpha=11,
        steer_amplify_alpha=8,
        warmup_model=True,
        compile_model=True,
        max_num_seqs=max_num_seqs,
        device=torch.device("cuda:0"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure generation throughput and latency.")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--min-len", type=int, default=128, help="minimum prompt length in tokens")
    parser.add_argument("--max-len", type=int, default=1024, help="maximum prompt length in tokens")
    parser.add_argument("--max-tokens", type=int, default=256, help="tokens generated per request")
    parser.add_argument("--concurrency", type=int, default=4, help="max requests in flight")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def random_seqs(rng: random.Random, num: int, min_len: int, max_len: int) -> list[list[int]]:
    return [
        [rng.randint(1, VOCAB_SIZE - 1) for _ in range(rng.randint(min_len, max_len))] for _ in range(num)
    ]


async def async_process(
    seq: list[int], llm: LLMEngine, sp: SamplingParams, sema: asyncio.Semaphore
) -> Metrics:
    async with sema:
        _, metrics = await timed_generate(llm, seq, sp)
    return metrics


async def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    sp = SamplingParams(max_tokens=args.max_tokens)
    llm = init_engine(init_throughput_cde_config(args.concurrency))
    async with llm:
        warmup_seqs = random_seqs(rng, args.num_warmup, args.min_len, args.max_len)
        bench_seqs = random_seqs(rng, args.num_requests, args.min_len, args.max_len)

        # wanna limit concurrency for accurate ttfb ttlb latency measurement
        sem = asyncio.Semaphore(llm.cde_config.max_num_seqs)

        await asyncio.gather(*[async_process(seq, llm, sp, sem) for seq in warmup_seqs])

        start_time = time.perf_counter()
        metrics = await asyncio.gather(*[async_process(seq, llm, sp, sem) for seq in bench_seqs])
        end_time = time.perf_counter()
        num_tokens, ttfb, ibl, ttlb = average_metrics(metrics)
        print(f"requests:   {len(metrics)}")
        print(f"num_tokens: {num_tokens}")
        print(f"ttfb:       {ttfb:.2f} s")
        print(f"ibl:        {ibl:.2f} s")
        print(f"ttlb:       {ttlb:.2f} s")
        print(f"total time: {end_time - start_time:.2f} s")
        print(f"throughput: {num_tokens / (end_time - start_time):.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
