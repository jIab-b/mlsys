import torch
from torch.utils.cpp_extension import load_inline

_module = None


def _get_module():
    global _module
    if _module is None:
        with open(__file__.replace("index_bindings.py", "dsa_index.cu")) as f:
            cuda_src = f.read()
        _module = load_inline(
            name="dsa_topk_indexer_ext",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _module


def compile_kernel() -> None:
    _get_module()


def dsa_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )

    return topk_indices
