"""Submission template for sparse_attn kernel.

Implement custom_kernel to compute sparse attention over selected KV entries.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Compute sparse attention over top-K selected KV cache entries.

    Args:
        data: Tuple of (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

    Returns:
        Tuple of (output, lse) where:
            output: [batch, num_heads, head_dim_ckv] bfloat16
            lse: [batch, num_heads] float32 (log-sum-exp)
    """
    # TODO: Replace with optimized implementation
    from reference import ref_kernel
    return ref_kernel(data)


def compile_kernel():
    """Optional: Pre-compile/warm up the kernel."""
    pass
