#!/usr/bin/env python3
"""Local test runner for sparse_index.py and sparse_attn.py kernels.

Usage:
    python local_test.py sparse_index --random 3
    python local_test.py sparse_attn --random 3
    python local_test.py sparse_index --real 3
    python local_test.py sparse_attn --real 3
    python local_test.py sparse_index  # uses default test spec
"""
import argparse
import math
import sys
import time
from pathlib import Path

import torch

# Add eval_suite to path for workload_loader
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "eval_suite"))

from common.workload_loader import (
    generate_sparse_attention_specs,
    generate_real_workload_specs,
    load_real_workload_tensors,
    parse_real_spec,
    FLASHINFER_TRACE,
)

# Constants matching the kernel expectations
PAGE_SIZE = 64
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 256


def _make_k_index_cache_fp8(num_pages: int, device: torch.device) -> torch.Tensor:
    """Create FP8 quantized K index cache in deep_gemm format."""
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


def parse_spec(spec: str) -> dict:
    """Parse a test spec string like 'batch: 1; num_pages: 16; seed: 42'."""
    result = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition(":")
        key = key.strip()
        val = val.strip()
        try:
            result[key] = int(val)
        except ValueError:
            result[key] = val
    return result


def generate_index_input(batch: int, num_pages: int, seq_len: int, seed: int, device: torch.device):
    """Generate inputs for sparse_index kernel."""
    torch.manual_seed(seed)

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


def generate_attn_input(batch: int, num_pages: int, seq_len: int, seed: int, device: torch.device):
    """Generate inputs for sparse_attn kernel (needs sparse_indices from indexer)."""
    from sparse_index import run as sparse_index_run

    torch.manual_seed(seed)

    # First generate indexer inputs and compute indices
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

    # Compute sparse indices using reference indexer
    topk_indices = sparse_index_run(
        q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table
    )[0]
    sparse_indices = topk_indices.to(torch.int32)

    # Generate attention inputs
    q_nope = torch.randn(
        (batch, NUM_QO_HEADS, HEAD_DIM_CKV), device=device, dtype=torch.float16
    )
    q_pe = torch.randn(
        (batch, NUM_QO_HEADS, HEAD_DIM_KPE), device=device, dtype=torch.float16
    )
    ckv_cache = torch.randn(
        (num_pages, PAGE_SIZE, HEAD_DIM_CKV), device=device, dtype=torch.float16
    )
    kpe_cache = torch.randn(
        (num_pages, PAGE_SIZE, HEAD_DIM_KPE), device=device, dtype=torch.float16
    )

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV)

    return (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def generate_index_input_real(workload: dict, seed: int, device: torch.device):
    """Generate inputs for sparse_index from real workload."""
    tensors = load_real_workload_tensors(workload, base_path=FLASHINFER_TRACE, device=str(device))
    axes = workload.get("axes", {})
    torch.manual_seed(seed)

    batch = axes.get("batch_size", axes.get("batch", 1))
    num_pages = axes.get("num_pages", 16)
    seq_len = axes.get("seq_len", 512)

    q_index_fp8 = tensors.get("q_index_fp8")
    if q_index_fp8 is None:
        q_index_fp8 = torch.randn((batch, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)

    k_index_cache_fp8 = tensors.get("k_index_cache_fp8")
    if k_index_cache_fp8 is None:
        k_index_cache_fp8 = _make_k_index_cache_fp8(num_pages, device)

    weights = tensors.get("weights")
    if weights is None:
        weights = torch.rand((batch, NUM_INDEX_HEADS), device=device, dtype=torch.float32)

    seq_lens_t = tensors.get("seq_lens")
    if seq_lens_t is None:
        seq_lens_t = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    block_table = tensors.get("block_table")
    if block_table is None:
        block_table = torch.arange(num_pages, device=device, dtype=torch.int32).unsqueeze(0).repeat(batch, 1)

    return (q_index_fp8, k_index_cache_fp8, weights, seq_lens_t, block_table)


def generate_attn_input_real(workload: dict, seed: int, device: torch.device):
    """Generate inputs for sparse_attn from real workload."""
    from sparse_index import run as sparse_index_run

    tensors = load_real_workload_tensors(workload, base_path=FLASHINFER_TRACE, device=str(device))
    axes = workload.get("axes", {})
    torch.manual_seed(seed)

    batch = axes.get("batch_size", axes.get("batch", 1))
    num_pages = axes.get("num_pages", 16)
    seq_len = axes.get("seq_len", 512)

    q_nope = tensors.get("q_nope")
    if q_nope is None:
        q_nope = torch.randn((batch, NUM_QO_HEADS, HEAD_DIM_CKV), device=device, dtype=torch.float16)

    q_pe = tensors.get("q_pe")
    if q_pe is None:
        q_pe = torch.randn((batch, NUM_QO_HEADS, HEAD_DIM_KPE), device=device, dtype=torch.float16)

    ckv_cache = tensors.get("ckv_cache")
    if ckv_cache is None:
        ckv_cache = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_CKV), device=device, dtype=torch.float16)

    kpe_cache = tensors.get("kpe_cache")
    if kpe_cache is None:
        kpe_cache = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_KPE), device=device, dtype=torch.float16)

    # Compute sparse_indices
    sparse_indices = tensors.get("sparse_indices")
    if sparse_indices is None:
        q_index_fp8 = tensors.get("q_index_fp8")
        if q_index_fp8 is None:
            q_index_fp8 = torch.randn((batch, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)

        k_index_cache_fp8 = tensors.get("k_index_cache_fp8")
        if k_index_cache_fp8 is None:
            k_index_cache_fp8 = _make_k_index_cache_fp8(num_pages, device)

        weights = tensors.get("weights")
        if weights is None:
            weights = torch.rand((batch, NUM_INDEX_HEADS), device=device, dtype=torch.float32)

        seq_lens_t = tensors.get("seq_lens")
        if seq_lens_t is None:
            seq_lens_t = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

        block_table = tensors.get("block_table")
        if block_table is None:
            block_table = torch.arange(num_pages, device=device, dtype=torch.int32).unsqueeze(0).repeat(batch, 1)

        topk_indices = sparse_index_run(
            q_index_fp8, k_index_cache_fp8, weights, seq_lens_t, block_table
        )[0]
        sparse_indices = topk_indices.to(torch.int32)

    sm_scale = tensors.get("sm_scale")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV)

    return (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run_test(kernel: str, specs: list[str], real_workloads: list[dict] | None = None):
    """Run tests for the specified kernel."""
    device = torch.device("cuda")

    if kernel == "sparse_index":
        from sparse_index import run as kernel_fn
    elif kernel == "sparse_attn":
        from sparse_attn import run as kernel_fn
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    print(f"\n{'='*60}")
    print(f"Testing {kernel}")
    print(f"{'='*60}")

    for i, spec in enumerate(specs):
        # Check if this is a real workload spec
        real_wl = parse_real_spec(spec) if real_workloads is None else None
        if real_wl is None and real_workloads is not None and i < len(real_workloads):
            real_wl = real_workloads[i]

        if real_wl is not None:
            workload = real_wl.get("workload", real_wl)
            axes = workload.get("axes", {})
            print(f"\n[{i+1}] REAL workload: batch={axes.get('batch_size', axes.get('batch', '?'))}, "
                  f"num_pages={axes.get('num_pages', '?')}, seq_len={axes.get('seq_len', '?')}")

            if kernel == "sparse_index":
                data = generate_index_input_real(workload, seed=42, device=device)
            else:
                data = generate_attn_input_real(workload, seed=42, device=device)
        else:
            params = parse_spec(spec)
            print(f"\n[{i+1}] {spec}")

            batch = params.get("batch", 1)
            num_pages = params.get("num_pages", 16)
            seq_len = params.get("seq_len", 512)
            seed = params.get("seed", 42)

            if kernel == "sparse_index":
                data = generate_index_input(batch, num_pages, seq_len, seed, device)
            else:
                data = generate_attn_input(batch, num_pages, seq_len, seed, device)

        # Warmup
        torch.cuda.synchronize()
        _ = kernel_fn(*data)
        torch.cuda.synchronize()

        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        output = kernel_fn(*data)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"    Output shapes: {[o.shape for o in output] if isinstance(output, tuple) else output.shape}")
        print(f"    Time: {elapsed_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Local test runner for sparse kernels")
    parser.add_argument("kernel", choices=["sparse_index", "sparse_attn"], help="Kernel to test")
    parser.add_argument(
        "--random",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help="Sample N random test specs from FlashInfer workloads (default: 3)",
    )
    parser.add_argument(
        "--real",
        type=int,
        nargs="?",
        const=3,
        metavar="N",
        help="Sample N REAL workloads with safetensor data (default: 3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--spec",
        type=str,
        help="Manual spec string (e.g., 'batch: 4; num_pages: 32; seq_len: 1024; seed: 42')",
    )
    args = parser.parse_args()

    # Generate specs based on mode
    if args.real is not None:
        print(f"Generating {args.real} REAL workload specs (with safetensors)...")
        # Map kernel name to op_type for workload lookup
        op_type = "sparse_index" if args.kernel == "sparse_index" else "sparse_attn"
        specs = generate_real_workload_specs(op_type, count=args.real, seed=args.seed)
        if not specs:
            print("No real workloads found, falling back to mla_paged...")
            specs = generate_real_workload_specs("mla_paged", count=args.real, seed=args.seed)
        real_workloads = [parse_real_spec(s) for s in specs]
        run_test(args.kernel, specs, real_workloads)
    elif args.random is not None:
        print(f"Generating {args.random} random test specs...")
        specs = generate_sparse_attention_specs(count=args.random, seed=args.seed)
        print(f"Generated specs:\n" + "\n".join(f"  {s}" for s in specs))
        run_test(args.kernel, specs)
    elif args.spec:
        run_test(args.kernel, [args.spec])
    else:
        # Default test
        default_spec = "batch: 1; num_pages: 16; seq_len: 512; seed: 42"
        print(f"Using default spec: {default_spec}")
        run_test(args.kernel, [default_spec])


if __name__ == "__main__":
    main()
