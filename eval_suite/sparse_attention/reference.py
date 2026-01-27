import json
import math
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from sub_test import sparse_index as sparse_index_run
from sub_test import sparse_attn as sparse_attn_run
from task import input_t, output_t
from common.utils import make_match_reference


PAGE_SIZE = 64
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 256


def _make_k_index_cache_fp8(num_pages: int, device: torch.device) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn not available; float8 required for this reference.")

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
    torch.manual_seed(seed)
    device = torch.device("cuda")

    total_tokens = num_pages * PAGE_SIZE
    if seq_len > total_tokens:
        raise ValueError(f"seq_len ({seq_len}) exceeds total tokens ({total_tokens}).")

    q_index_fp8 = torch.randn((batch, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)
    k_index_cache_fp8 = _make_k_index_cache_fp8(num_pages, device)
    weights = torch.rand((batch, NUM_INDEX_HEADS), device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)

    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).unsqueeze(0).repeat(batch, 1)

    q_nope = torch.randn((batch, NUM_QO_HEADS, HEAD_DIM_CKV), device=device, dtype=torch.float16)
    q_pe = torch.randn((batch, NUM_QO_HEADS, HEAD_DIM_KPE), device=device, dtype=torch.float16)
    ckv_cache = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_CKV), device=device, dtype=torch.float16)
    kpe_cache = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_KPE), device=device, dtype=torch.float16)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV)

    data = (
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sm_scale,
    )
    _maybe_dump_trace(data, batch=batch, num_pages=num_pages, seq_len=seq_len, seed=seed)
    return data


def _maybe_dump_trace(data: input_t, **spec) -> None:
    trace_root = os.environ.get("TRACE_OUT")
    if not trace_root:
        return

    trace_root = Path(trace_root)
    def_dir = trace_root / "definitions"
    wl_dir = trace_root / "workloads" / "sparse_attention"
    tr_dir = trace_root / "traces" / "sparse_attention"
    def_dir.mkdir(parents=True, exist_ok=True)
    wl_dir.mkdir(parents=True, exist_ok=True)
    tr_dir.mkdir(parents=True, exist_ok=True)

    definition_path = def_dir / "sparse_attention.json"
    if not definition_path.exists():
        definition = {
            "name": "sparse_attention",
            "inputs": [
                "q_index_fp8",
                "k_index_cache_fp8",
                "weights",
                "seq_lens",
                "block_table",
                "q_nope",
                "q_pe",
                "ckv_cache",
                "kpe_cache",
                "sm_scale",
            ],
            "outputs": ["output", "lse"],
        }
        definition_path.write_text(json.dumps(definition, indent=2))

    uid = uuid.uuid4().hex
    payload_path = wl_dir / f"{uid}.pt"
    meta_path = wl_dir / f"{uid}.json"
    torch.save(data, payload_path)
    meta = {"id": uid, "spec": spec, "payload": str(payload_path.name)}
    meta_path.write_text(json.dumps(meta, indent=2))

    trace_path = tr_dir / f"{uid}.json"
    if not trace_path.exists():
        trace_path.write_text(json.dumps({"id": uid, "status": "generated"}, indent=2))


def ref_kernel(data: input_t) -> output_t:
    (
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sm_scale,
    ) = data

    topk_indices = sparse_index_run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)[0]
    sparse_indices = topk_indices.to(torch.int32)

    return sparse_attn_run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


check_implementation = make_match_reference(ref_kernel, rtol=1e-03, atol=1e-03)
