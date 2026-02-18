import os
import torch
from pathlib import Path
from torch.utils.cpp_extension import load_inline

try:
    from tvm.ffi import register_func
except Exception:
    def register_func(_name):
        def deco(fn):
            return fn
        return deco


_module = None
_op = None

os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "10.0a")


def _read_cuda_source() -> str:
    here = Path(__file__).resolve().parent
    for candidate in ("dsa_index.cu", "kernel.cu"):
        path = here / candidate
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(f"No CUDA source found in {here} (expected dsa_index.cu or kernel.cu)")


def _load_extension():
    global _module, _op
    if _op is not None:
        return _op

    cuda_src = _read_cuda_source()
    _module = load_inline(
        name="dsa_topk_indexer_ext",
        cpp_sources="",
        cuda_sources=cuda_src,
        verbose=True,
        is_python_module=False,
        no_implicit_headers=True,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
        ],
        extra_ldflags=["-lcuda"],
    )
    _op = torch.ops.index_bindings.dsa_topk_indexer_launch
    return _op


def compile_kernel() -> None:
    _load_extension()


def dsa_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    op = _load_extension()
    op(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )
    return topk_indices


@register_func("flashinfer.kernel")
def kernel(
    q_index_fp8,
    k_index_cache_fp8,
    weights,
    seq_lens,
    block_table,
    topk_indices,
):
    dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )
    return topk_indices
