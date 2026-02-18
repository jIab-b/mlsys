import torch
from pathlib import Path
from torch.utils.cpp_extension import load_inline


_module = None


def _read_cuda_source() -> str:
    here = Path(__file__).resolve().parent
    for candidate in ("dsa_index.cu", "kernel.cu"):
        path = here / candidate
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"No CUDA source found in {here} (expected dsa_index.cu or kernel.cu)")


_CPP_DECL_SRC = """
#include <torch/extension.h>
void dsa_topk_indexer_launch(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices);
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_indexer_ext",
            cpp_sources=_CPP_DECL_SRC,
            cuda_sources=_read_cuda_source(),
            functions=["dsa_topk_indexer_launch"],
            verbose=True,
            extra_cuda_cflags=[
                "-O0",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--threads=4",
            ],
            extra_ldflags=["-lcuda"],
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


def kernel(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
    topk_indices,
):
    return dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )
