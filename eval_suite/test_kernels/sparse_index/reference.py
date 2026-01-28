"""Reference implementation for DSA TopK Indexer.

This kernel computes sparse attention indices using FP8 quantized inputs.
Formula: topk(sum(relu(q @ K.T) * weights, dim=heads))
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # eval_suite/
sys.path.insert(0, str(ROOT))

import torch

from sub_test import sparse_index as sparse_index_run
from task import input_t, output_t
from common.utils import make_match_reference


PAGE_SIZE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 256


def _make_k_index_cache_fp8(num_pages: int, device: torch.device) -> torch.Tensor:
    """Create FP8 quantized K index cache in deep_gemm format."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn not available")

    fp8 = torch.randn(
        (num_pages, PAGE_SIZE, INDEX_HEAD_DIM),
        device=device,
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8)

    scales = torch.rand((num_pages, PAGE_SIZE, 1), device=device, dtype=torch.float32)
    scales = scales * 0.5 + 0.5
    scale_bytes = scales.view(torch.uint8)

    packed = torch.cat([fp8_bytes, scale_bytes], dim=2)
    return packed.unsqueeze(2)


def _convert_mla_to_sparse_index(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    device: torch.device,
    max_pages: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert MLA paging metadata to sparse_index format.

    Args:
        kv_indptr: [batch+1] cumulative token counts
        kv_indices: [total_tokens] page indices for each token
        device: Target device
        max_pages: Maximum pages per sequence (for memory)

    Returns:
        seq_lens: [batch] sequence lengths
        block_table: [batch, max_pages_per_seq] page indices
        actual_num_pages: Number of unique pages used
    """
    batch_size = len(kv_indptr) - 1

    # Compute sequence lengths from indptr
    seq_lens = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32).to(device)

    # Find max sequence length (capped)
    max_seq_len = min(int(seq_lens.max().item()), max_pages * PAGE_SIZE)

    # Build block table from kv_indices
    max_pages_per_seq = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages_per_seq = min(max_pages_per_seq, max_pages)

    block_table = torch.zeros((batch_size, max_pages_per_seq), dtype=torch.int32, device=device)

    for b in range(batch_size):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        seq_indices = kv_indices[start:end]

        # Get unique pages in order (page = token_idx // PAGE_SIZE for paged layout)
        # kv_indices are already page indices in MLA format
        n_tokens = min(end - start, max_pages_per_seq * PAGE_SIZE)
        n_pages = min((n_tokens + PAGE_SIZE - 1) // PAGE_SIZE, max_pages_per_seq)

        if len(seq_indices) > 0:
            # Take first n_pages worth of indices
            page_indices = seq_indices[:n_pages].to(torch.int32).to(device)
            block_table[b, :len(page_indices)] = page_indices

    # Cap seq_lens to match block_table capacity
    seq_lens = torch.clamp(seq_lens, max=max_pages_per_seq * PAGE_SIZE)

    # Find actual number of unique pages needed
    actual_num_pages = max(int(block_table.max().item()) + 1, 1)

    return seq_lens, block_table, actual_num_pages


def generate_input(
    batch: int = 1,
    num_pages: int = 16,
    seq_len: int = 512,
    seed: int = 42,
    __real_workload__: dict | None = None,
) -> input_t:
    """Generate inputs for sparse index kernel.

    If __real_workload__ is provided, load tensors from safetensors files.
    Otherwise, generate random tensors.
    """
    device = torch.device("cuda")

    # Handle real workload loading from safetensors
    if __real_workload__ is not None:
        sys.path.insert(0, str(ROOT))
        from common.workload_loader import load_real_workload_tensors, FLASHINFER_TRACE

        workload = __real_workload__.get("workload", __real_workload__)
        tensors = load_real_workload_tensors(workload, base_path=FLASHINFER_TRACE, device="cuda")
        axes = workload.get("axes", {})

        torch.manual_seed(seed)

        # Check if this is an MLA workload (has kv_indptr/kv_indices)
        kv_indptr = tensors.get("kv_indptr")
        kv_indices = tensors.get("kv_indices")

        if kv_indptr is not None and kv_indices is not None:
            # MLA workload - convert paging metadata
            seq_lens_t, block_table, actual_num_pages = _convert_mla_to_sparse_index(
                kv_indptr, kv_indices, device, max_pages=256
            )
            b = len(seq_lens_t)
            n_pages = actual_num_pages

            # Generate random compute tensors with correct shapes
            q_index_fp8 = torch.randn((b, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)
            k_index_cache_fp8 = _make_k_index_cache_fp8(n_pages, device)
            weights = torch.rand((b, NUM_INDEX_HEADS), device=device, dtype=torch.float32)

            return (q_index_fp8, k_index_cache_fp8, weights, seq_lens_t, block_table)

        # Direct sparse_index workload format (if available in future)
        q_index_fp8 = tensors.get("q_index_fp8")
        if q_index_fp8 is None:
            b = axes.get("batch_size", axes.get("batch", batch))
            q_index_fp8 = torch.randn((b, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)

        k_index_cache_fp8 = tensors.get("k_index_cache_fp8")
        if k_index_cache_fp8 is None:
            n_pages = min(axes.get("num_pages", num_pages), 256)  # Cap for memory
            k_index_cache_fp8 = _make_k_index_cache_fp8(n_pages, device)

        weights = tensors.get("weights")
        if weights is None:
            b = axes.get("batch_size", axes.get("batch", batch))
            weights = torch.rand((b, NUM_INDEX_HEADS), device=device, dtype=torch.float32)

        seq_lens_t = tensors.get("seq_lens")
        if seq_lens_t is None:
            b = axes.get("batch_size", axes.get("batch", batch))
            s_len = min(axes.get("seq_len", axes.get("num_kv_indices", seq_len)), 256 * PAGE_SIZE)
            seq_lens_t = torch.full((b,), s_len, device=device, dtype=torch.int32)

        block_table = tensors.get("block_table")
        if block_table is None:
            b = axes.get("batch_size", axes.get("batch", batch))
            n_pages = min(axes.get("num_pages", num_pages), 256)
            block_table = torch.arange(n_pages, device=device, dtype=torch.int32).unsqueeze(0).repeat(b, 1)

        return (q_index_fp8, k_index_cache_fp8, weights, seq_lens_t, block_table)

    # Standard random generation
    torch.manual_seed(seed)

    total_tokens = num_pages * PAGE_SIZE
    if seq_len > total_tokens:
        raise ValueError(f"seq_len ({seq_len}) exceeds total tokens ({total_tokens}).")

    q_index_fp8 = torch.randn(
        (batch, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16
    )
    k_index_cache_fp8 = _make_k_index_cache_fp8(num_pages, device)
    weights = torch.rand((batch, NUM_INDEX_HEADS), device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    block_table = (
        torch.arange(num_pages, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch, 1)
    )

    return (q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)


def ref_kernel(data: input_t) -> output_t:
    """Reference kernel using sub_test implementation."""
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    return sparse_index_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
