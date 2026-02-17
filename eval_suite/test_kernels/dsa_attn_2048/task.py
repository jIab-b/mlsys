import torch
from typing import TypeVar

input_t = TypeVar(
    "input_t",
    bound=tuple[
        torch.Tensor,  # q_nope
        torch.Tensor,  # q_pe
        torch.Tensor,  # ckv_cache
        torch.Tensor,  # kpe_cache
        torch.Tensor,  # sparse_indices
        float,         # sm_scale
    ],
)
output_t = TypeVar("output_t", bound=tuple[torch.Tensor, torch.Tensor])  # output, lse
