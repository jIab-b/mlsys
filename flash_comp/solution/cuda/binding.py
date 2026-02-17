"""
TVM FFI Bindings Template for CUDA Kernels.

This file provides Python bindings for your CUDA kernel using TVM FFI.
The entry point function name should match the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""

from tvm.ffi import register_func
from index_bindings import compile_kernel as _compile_kernel
from index_bindings import dsa_topk_indexer


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


def compile_kernel():
    _compile_kernel()
