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


def _convert_mla_to_sparse_attention(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    device: torch.device,
    max_pages: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert MLA paging metadata to sparse_attention format.

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

        n_tokens = min(end - start, max_pages_per_seq * PAGE_SIZE)
        n_pages = min((n_tokens + PAGE_SIZE - 1) // PAGE_SIZE, max_pages_per_seq)

        if len(seq_indices) > 0:
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
    """Generate inputs for sparse attention.

    If __real_workload__ is provided, load tensors from safetensors files.
    Otherwise, generate random tensors.
    """
    device = torch.device("cuda")

    # Handle real workload loading from safetensors
    if __real_workload__ is not None:
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
            seq_lens_t, block_table, actual_num_pages = _convert_mla_to_sparse_attention(
                kv_indptr, kv_indices, device, max_pages=256
            )
            b = len(seq_lens_t)
            n_pages = actual_num_pages

            # Generate random compute tensors with correct shapes
            q_index_fp8 = torch.randn((b, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)
            k_index_cache_fp8 = _make_k_index_cache_fp8(n_pages, device)
            weights = torch.rand((b, NUM_INDEX_HEADS), device=device, dtype=torch.float32)
            q_nope = torch.randn((b, NUM_QO_HEADS, HEAD_DIM_CKV), device=device, dtype=torch.float16)
            q_pe = torch.randn((b, NUM_QO_HEADS, HEAD_DIM_KPE), device=device, dtype=torch.float16)
            ckv_cache = torch.randn((n_pages, PAGE_SIZE, HEAD_DIM_CKV), device=device, dtype=torch.float16)
            kpe_cache = torch.randn((n_pages, PAGE_SIZE, HEAD_DIM_KPE), device=device, dtype=torch.float16)

            sm_scale = tensors.get("sm_scale")
            if sm_scale is None:
                sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV)

            data = (
                q_index_fp8,
                k_index_cache_fp8,
                weights,
                seq_lens_t,
                block_table,
                q_nope,
                q_pe,
                ckv_cache,
                kpe_cache,
                sm_scale,
            )
            trace_ctx.set_current_workload(workload)
            return data

        # Direct sparse_attention workload format (fallback)
        b = axes.get("batch_size", axes.get("batch", batch))
        n_pages = min(axes.get("num_pages", num_pages), 256)  # Cap for memory
        s_len = min(axes.get("seq_len", axes.get("num_kv_indices", seq_len)), 256 * PAGE_SIZE)

        q_index_fp8 = tensors.get("q_index_fp8")
        if q_index_fp8 is None:
            q_index_fp8 = torch.randn((b, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float16)

        k_index_cache_fp8 = tensors.get("k_index_cache_fp8")
        if k_index_cache_fp8 is None:
            k_index_cache_fp8 = _make_k_index_cache_fp8(n_pages, device)

        weights = tensors.get("weights")
        if weights is None:
            weights = torch.rand((b, NUM_INDEX_HEADS), device=device, dtype=torch.float32)

        seq_lens_t = tensors.get("seq_lens")
        if seq_lens_t is None:
            seq_lens_t = torch.full((b,), s_len, device=device, dtype=torch.int32)

        block_table = tensors.get("block_table")
        if block_table is None:
            block_table = torch.arange(n_pages, device=device, dtype=torch.int32).unsqueeze(0).repeat(b, 1)

        q_nope = tensors.get("q_nope")
        if q_nope is None:
            q_nope = torch.randn((b, NUM_QO_HEADS, HEAD_DIM_CKV), device=device, dtype=torch.float16)

        q_pe = tensors.get("q_pe")
        if q_pe is None:
            q_pe = torch.randn((b, NUM_QO_HEADS, HEAD_DIM_KPE), device=device, dtype=torch.float16)

        ckv_cache = tensors.get("ckv_cache")
        if ckv_cache is None:
            ckv_cache = torch.randn((n_pages, PAGE_SIZE, HEAD_DIM_CKV), device=device, dtype=torch.float16)

        kpe_cache = tensors.get("kpe_cache")
        if kpe_cache is None:
            kpe_cache = torch.randn((n_pages, PAGE_SIZE, HEAD_DIM_KPE), device=device, dtype=torch.float16)

        sm_scale = tensors.get("sm_scale")
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV)

        data = (
            q_index_fp8,
            k_index_cache_fp8,
            weights,
            seq_lens_t,
            block_table,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            sm_scale,
        )
        trace_ctx.set_current_workload(workload)
        return data

    # Standard random generation
    torch.manual_seed(seed)

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
