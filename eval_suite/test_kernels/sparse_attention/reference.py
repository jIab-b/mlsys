import json
import math
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # eval_suite/
sys.path.insert(0, str(ROOT))

import torch

from sub_test import sparse_index as sparse_index_run
from sub_test import sparse_attn as sparse_attn_run
from task import input_t, output_t
from common.utils import make_match_reference
from common import trace as trace_ctx
from safetensors.torch import save_file as save_safetensors


PAGE_SIZE = 64
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 256


OP_TYPE = os.environ.get("TRACE_OP_TYPE", "sparse_attention")
DEFINITION_NAME = os.environ.get("TRACE_DEFINITION", "sparse_attention")


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
    trace_root = trace_ctx.get_trace_root()
    if trace_root is None:
        return
    def_dir = trace_root / "definitions"
    wl_dir = trace_root / "workloads" / OP_TYPE
    blob_dir = trace_root / "blob" / "workloads" / OP_TYPE / DEFINITION_NAME
    def_dir.mkdir(parents=True, exist_ok=True)
    wl_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    definition_path = def_dir / f"{DEFINITION_NAME}.json"
    if not definition_path.exists():
        definition = {
            "name": DEFINITION_NAME,
            "description": "Sparse attention with separate index + attention stages.",
            "op_type": OP_TYPE,
            "tags": [
                "status:generated",
                "layout:paged_kv",
            ],
            "axes": {
                "batch": {"type": "var", "description": "Number of query tokens."},
                "num_pages": {"type": "var", "description": "Number of KV pages."},
                "seq_len": {"type": "var", "description": "Sequence length."},
                "page_size": {"type": "const", "value": PAGE_SIZE, "description": "KV page size."},
                "num_qo_heads": {"type": "const", "value": NUM_QO_HEADS, "description": "Number of query heads."},
                "head_dim_ckv": {"type": "const", "value": HEAD_DIM_CKV, "description": "CKV head dimension."},
                "head_dim_kpe": {"type": "const", "value": HEAD_DIM_KPE, "description": "KPE head dimension."},
                "num_index_heads": {"type": "const", "value": NUM_INDEX_HEADS, "description": "Index heads."},
                "index_head_dim": {"type": "const", "value": INDEX_HEAD_DIM, "description": "Index head dimension."},
                "topk": {"type": "const", "value": TOPK, "description": "Top-k selected tokens."},
            },
            "inputs": {
                "q_index_fp8": {"shape": ["batch", "num_index_heads", "index_head_dim"], "dtype": "float16"},
                "k_index_cache_fp8": {"shape": ["num_pages", "page_size", 1, 132], "dtype": "uint8"},
                "weights": {"shape": ["batch", "num_index_heads"], "dtype": "float32"},
                "seq_lens": {"shape": ["batch"], "dtype": "int32"},
                "block_table": {"shape": ["batch", "num_pages"], "dtype": "int32"},
                "q_nope": {"shape": ["batch", "num_qo_heads", "head_dim_ckv"], "dtype": "float16"},
                "q_pe": {"shape": ["batch", "num_qo_heads", "head_dim_kpe"], "dtype": "float16"},
                "ckv_cache": {"shape": ["num_pages", "page_size", "head_dim_ckv"], "dtype": "float16"},
                "kpe_cache": {"shape": ["num_pages", "page_size", "head_dim_kpe"], "dtype": "float16"},
                "sm_scale": {"shape": None, "dtype": "float32"},
            },
            "outputs": {
                "output": {"shape": ["batch", "num_qo_heads", "head_dim_ckv"], "dtype": "bfloat16"},
                "lse": {"shape": ["batch", "num_qo_heads"], "dtype": "float32"},
            },
            "reference": Path(__file__).with_name("reference.py").read_text(encoding="utf-8"),
        }
        definition_path.write_text(json.dumps(definition, indent=2))

    uid = uuid.uuid4().hex
    input_names = [
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
    ]
    inputs = {}
    for name, value in zip(input_names, data):
        if torch.is_tensor(value):
            file_name = f"{DEFINITION_NAME}_{uid}_{name}.safetensors"
            rel_path = f"./blob/workloads/{OP_TYPE}/{DEFINITION_NAME}/{file_name}"
            save_safetensors({name: value}, blob_dir / file_name)
            inputs[name] = {"type": "safetensors", "path": rel_path, "tensor_key": name}
        else:
            inputs[name] = {"type": "scalar", "value": value}

    workload = {"uuid": uid, "axes": spec, "inputs": inputs}
    record = {
        "definition": DEFINITION_NAME,
        "solution": None,
        "workload": workload,
        "evaluation": None,
    }
    workloads_path = wl_dir / f"{DEFINITION_NAME}.jsonl"
    with workloads_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    trace_ctx.set_current_workload(workload)


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
