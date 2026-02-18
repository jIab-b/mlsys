"""
TVM FFI bindings for CUDA kernel.

This follows the starter-kit format: expose a `flashinfer.kernel`
entrypoint and optional compile helper.
"""

from tvm.ffi import register_func
from dsa_index import compile_kernel as _compile_kernel
from dsa_index import dsa_topk_indexer


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
