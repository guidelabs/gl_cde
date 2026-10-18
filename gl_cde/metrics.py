from dataclasses import dataclass
import math
import time
import contextlib

from .cde import LLMEngine, SamplingParams


@dataclass(slots=True, frozen=True, kw_only=True, eq=False)
class Metrics:
    num_tokens: int
    ttfb: float
    ibl: list[float]
    ttlb: float


async def timed_generate(llm: LLMEngine, ids: list[int], sp: SamplingParams) -> tuple[list[int], Metrics]:
    tokens = []
    ibl = []
    ttfb = math.nan
    start_time = prev_time = time.perf_counter()
    async with contextlib.aclosing(llm.generate(ids, sp)) as gen:
        async for output in gen:
            now = time.perf_counter()
            if math.isnan(ttfb):
                ttfb = now - start_time
            else:
                ibl.append(now - prev_time)
            prev_time = now
            tokens.extend(output.tokens)
    ttlb = prev_time - start_time if not math.isnan(ttfb) else math.nan
    return tokens, Metrics(num_tokens=len(tokens), ttfb=ttfb, ibl=ibl, ttlb=ttlb)


def average_metrics(metrics: list[Metrics]) -> tuple[int, float, float, float]:
    if len(metrics) == 0:
        raise ValueError("average_metrics needs at least one Metrics")
    num_tokens = sum(metric.num_tokens for metric in metrics)
    ttfb = sum(metric.ttfb for metric in metrics) / len(metrics)
    # mean of per-request means, over requests that produced more than one block
    ibls = [sum(metric.ibl) / len(metric.ibl) for metric in metrics if len(metric.ibl) != 0]
    ibl = sum(ibls) / len(ibls) if len(ibls) != 0 else math.nan
    ttlb = sum(metric.ttlb for metric in metrics) / len(metrics)
    return num_tokens, ttfb, ibl, ttlb
