import torch
from typing import TypedDict, TypeVar

input_t = TypeVar(
    "input_t",
    bound=tuple[
        torch.Tensor,  # q_index_fp8
        torch.Tensor,  # k_index_cache_fp8
        torch.Tensor,  # weights
        torch.Tensor,  # seq_lens
        torch.Tensor,  # block_table
        torch.Tensor,  # q_nope
        torch.Tensor,  # q_pe
        torch.Tensor,  # ckv_cache
        torch.Tensor,  # kpe_cache
        float,         # sm_scale
    ],
)
output_t = TypeVar("output_t", bound=tuple[torch.Tensor, torch.Tensor])


class TestSpec(TypedDict):
    batch: int
    num_pages: int
    seq_len: int
    seed: int
