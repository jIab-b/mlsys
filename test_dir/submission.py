"""Standalone DSA top-k=2048 submission (no cross-file imports)."""

torch = __import__("torch")

PAGE_SIZE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 2048
_DEBUG_PRINTED_POST_ONCE = False
_DEBUG_POST_TOKENS = (0, 1, 2, 3, 4, 60, 61, 62, 63)


def _fixed_debug_tok(seq_len, device):
    out = torch.zeros((len(_DEBUG_POST_TOKENS),), dtype=torch.int64, device=device)
    if seq_len <= 0:
        return out
    last = seq_len - 1
    for i, t in enumerate(_DEBUG_POST_TOKENS):
        out[i] = t if t <= last else last
    return out


def decode_fp8_kv_cache_parts(k_index_cache_fp8):
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float, scale


def dequant_fp8_kv_cache(k_index_cache_fp8):
    fp8_float, scale = decode_fp8_kv_cache_parts(k_index_cache_fp8)
    return fp8_float * scale


@torch.no_grad()
def custom_kernel(data):
    global _DEBUG_PRINTED_POST_ONCE
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data

    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape

    assert num_index_heads == NUM_INDEX_HEADS
    assert index_head_dim == INDEX_HEAD_DIM
    assert page_size == PAGE_SIZE

    device = q_index_fp8.device
    q = q_index_fp8.to(torch.float32)
    k_raw_all, k_scale_all = decode_fp8_kv_cache_parts(k_index_cache_fp8)
    k_all = k_raw_all * k_scale_all

    topk_indices = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len <= 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        k_paged = k_all[page_indices]
        k = k_paged.reshape(-1, index_head_dim)[:seq_len]

        scores = q[b] @ k.T
        scores_relu = torch.relu(scores)
        final_scores = (scores_relu * weights[b][:, None]).sum(dim=0)

        if (not _DEBUG_PRINTED_POST_ONCE) and b == 0:
            dbg_tok = _fixed_debug_tok(seq_len, device)
            for i in range(len(_DEBUG_POST_TOKENS)):
                t = int(dbg_tok[i].item())
                page_slot = t // page_size
                offset = t % page_size
                global_page = int(page_indices[page_slot].item())
                global_tok = global_page * page_size + offset
                v = float(final_scores[t].item())
                print(
                    f"post_ep_sample{i}: local_tok={t} global_tok={global_tok} "
                    f"post_ep_final_score={v:.7f}"
                )
            _DEBUG_PRINTED_POST_ONCE = True

        actual_topk = min(TOPK, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)


def compile_kernel():
    return None
