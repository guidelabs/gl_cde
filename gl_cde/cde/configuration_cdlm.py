from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, eq=False, slots=True)
class CDESettings:
    endofchunk_token_id: int
    mask_token_id: int
    banned_ids: list[int]
    eos_token_id: int | None
    eot_token_id: int | None
    # 0-indexed steering injection start
    inject_layer: int = 7
    steer_amplify_alpha: float = 1.0
    steer_suppress_alpha: float = 1.0
    warmup_model: bool = False
    compile_model: bool = True
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 5
    kvcache_block_size: int = 64
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    def __post_init__(self) -> None:
        if self.kvcache_block_size % 32 != 0:
            raise ValueError(f"kvcache_block_size ({self.kvcache_block_size}) must be divisible by 32")
