from dataclasses import dataclass, field
import random


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    repetition_penalty: float = 1.0
    top_p: float = 0.0
    seed: int = field(default_factory=lambda: random.randrange(0, 2**32))
    use_entropy_sampling: bool = False
    concept_ids: list[int] = field(default_factory=list)
    mai_lm_target: float = 0.0
    steering_cutoff_tokens: int = 32
    stop_token_ids: list[int] = field(default_factory=list)
