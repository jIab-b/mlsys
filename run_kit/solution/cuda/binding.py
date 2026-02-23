import torch

import ctypes
from tvm.ffi import register_func
from torch.utils.cpp_extension import load_inline




@register_func("flashinfer.kernel")
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, out_topk_indices):
    q_index_fp8 = torch.from_dlpack(q_index_fp8)
    k_index_cache_fp8 = torch.from_dlpack(k_index_cache_fp8)
    weights = torch.from_dlpack(weights)
    seq_lens = torch.from_dlpack(seq_lens)
    block_table = torch.from_dlpack(block_table)
    out_topk_indices = torch.from_dlpack(out_topk_indices)

    batch = int(q_index_fp8.shape[0])
    topk = 2048

    if batch == 0:
        out_topk_indices.fill_(-1)
        return

    max_seq = min(int(seq_lens.max().item()), int(block_table.shape[1]) * 64)
    if max_seq <= 0:
        out_topk_indices.fill_(-1)
        return

    out_scores = torch.empty((batch, max_seq), dtype=torch.float32, device=q_index_fp8.device)
    out_ids = torch.empty((batch, max_seq), dtype=torch.int32, device=q_index_fp8.device)

    _dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        out_scores,
        out_ids,
    )

    k = min(topk, max_seq)
    top_pos = torch.topk(out_scores, k, dim=1).indices
    out_topk_indices.fill_(-1)
    out_topk_indices[:, :k] = torch.gather(out_ids, 1, top_pos).to(torch.int32)












_module = None

cpp_decl_src = """
#include <torch/extension.h>
void dsa_topk_indexer_launch(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor out_scores,
    torch::Tensor out_ids);
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_indexer_ext",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            #verbose=True,
            no_implicit_headers=True,
            extra_cuda_cflags=[
                "-O1",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--split-compile=4",
                "--relocatable-device-code=false",
            ],
            extra_ldflags=["-lcuda"],
        )
    return _module


def compile_kernel():
    _get_module()


def _dsa_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    out_scores: torch.Tensor,
    out_ids: torch.Tensor,
):
    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        out_scores,
        out_ids,
    )
    return out_scores, out_ids


def _dequant_fp8_kv_cache(k_index_cache_fp8: torch.Tensor) -> torch.Tensor:
    k_u8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_plus_scale = k_u8.shape
    head_dim = head_dim_plus_scale - 4

    kv_flat = k_u8.view(num_pages, page_size * head_dim_plus_scale)
    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_vals = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn).to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scales = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)
    return fp8_vals * scales


def custom_kernel(data: input_t) -> output_t:
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    batch = int(q_index_fp8.shape[0])
    topk = 2048
    out_topk_indices = torch.full((batch, topk), -1, dtype=torch.int32, device=q_index_fp8.device)
    if batch == 0:
        return (out_topk_indices,)

    max_seq = min(int(seq_lens.max().item()), int(block_table.shape[1]) * 64)
    if max_seq <= 0:
        return (out_topk_indices,)

    out_scores = torch.empty((batch, max_seq), dtype=torch.float32, device=q_index_fp8.device)
    out_ids = torch.empty((batch, max_seq), dtype=torch.int32, device=q_index_fp8.device)

    _dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        out_scores,
        out_ids,
    )

    k = min(topk, max_seq)
    top_pos = torch.topk(out_scores, k, dim=1).indices
    out_topk_indices[:, :k] = torch.gather(out_ids, 1, top_pos).to(torch.int32)
    return (out_topk_indices,)
