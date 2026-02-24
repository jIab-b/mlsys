"""DSA top-k=2048 – no batch loop, scale after MMA (matches ld_smem computation order)."""

torch = __import__("torch")

PAGE_SIZE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 2048


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


@torch.no_grad()
def custom_kernel(data):
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data

    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape

    assert num_index_heads == NUM_INDEX_HEADS
    assert index_head_dim == INDEX_HEAD_DIM
    assert page_size == PAGE_SIZE

    device = q_index_fp8.device
    q = q_index_fp8.to(torch.float32)
    k_raw_all, k_scale_all = decode_fp8_kv_cache_parts(k_index_cache_fp8)

    max_pages = block_table.shape[1]
    page_indices = block_table.to(torch.long)
    flat_pi = page_indices.reshape(-1)
    k_paged = k_raw_all[flat_pi].reshape(
        batch_size, max_pages * page_size, index_head_dim
    )
    k_paged_scale = k_scale_all[flat_pi].reshape(
        batch_size, max_pages * page_size, 1
    )

    scores = q @ k_paged.transpose(1, 2)
    scores = scores * k_paged_scale.squeeze(-1).unsqueeze(1)
    scores_relu = torch.relu(scores)
    final_scores = (scores_relu * weights[:, :, None]).sum(dim=1)

    T = final_scores.shape[1]
    token_ids = torch.arange(T, device=device).unsqueeze(0)
    mask = token_ids >= seq_lens[:, None]
    final_scores.masked_fill_(mask, 0.0)

    actual_topk = TOPK
    _, topk_idx = torch.topk(final_scores, actual_topk, dim=1)

    page_idx_per_token = topk_idx // page_size
    offset_per_token = topk_idx % page_size
    global_page_idx = torch.gather(page_indices, 1, page_idx_per_token)
    topk_tokens = global_page_idx * page_size + offset_per_token

    invalid = torch.gather(mask, 1, topk_idx)
    topk_tokens[invalid] = -1

    topk_indices = topk_tokens.to(torch.int32)

    return (topk_indices,)


def compile_kernel():
    return None
