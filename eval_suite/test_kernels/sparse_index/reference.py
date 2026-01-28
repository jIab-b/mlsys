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


def generate_input(batch: int, num_pages: int, seq_len: int, seed: int) -> input_t:
    """Generate inputs for sparse index kernel."""
    torch.manual_seed(seed)
    device = torch.device("cuda")

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
