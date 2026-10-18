"""Shared model weight loading helpers."""

import torch


def is_safe_cast(src: torch.dtype, dst: torch.dtype) -> bool:
    if src == dst:
        return True

    safe_casts = {
        torch.bfloat16: {torch.float32},
    }
    return dst in safe_casts.get(src, set())


def cast_state_dict_to_meta(
    state_dict: dict[str, torch.Tensor],
    meta_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a state dict with tensors cast to match meta-model dtypes when safe.

    Keys present in ``state_dict`` but absent from ``meta_state_dict`` are kept
    unchanged (e.g. tied weights that HuggingFace omits from the model state).
    """
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name not in meta_state_dict:
            out[name] = tensor
            continue
        target_dtype = meta_state_dict[name].dtype
        if not is_safe_cast(tensor.dtype, target_dtype):
            raise TypeError(f"Unsafe dtype cast for {name}: checkpoint={tensor.dtype}, model={target_dtype}")
        out[name] = tensor if tensor.dtype == target_dtype else tensor.to(dtype=target_dtype)
    return out
