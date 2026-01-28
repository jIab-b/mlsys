"""Submission template for sparse_index kernel.

Implement custom_kernel to compute top-K indices from FP8 quantized inputs.
"""
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Compute top-K sparse attention indices.

    Args:
        data: Tuple of (q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)

    Returns:
        Tuple of (topk_indices,) where topk_indices is [batch, topk] int32
    """
    # TODO: Replace with optimized implementation
    from reference import ref_kernel
    return ref_kernel(data)


def compile_kernel():
    """Optional: Pre-compile/warm up the kernel."""
    pass
