import torch
from typing import TypeVar

input_t = TypeVar(
    "input_t",
    bound=tuple[
        torch.Tensor,  # q_index_fp8
        torch.Tensor,  # k_index_cache_fp8
        torch.Tensor,  # weights
        torch.Tensor,  # seq_lens
        torch.Tensor,  # block_table
    ],
)
output_t = TypeVar("output_t", bound=tuple[torch.Tensor])  # topk_indices
