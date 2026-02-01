"""Reference implementation for gqa_paged_decode_h32_kv4_d128_ps64.
Auto-extracted from flashinfer-bench definitions.
Op type: gqa_paged
"""
import torch
import math


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    assert page_size == 64

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q.device

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    gqa_ratio = num_qo_heads // num_kv_heads

    k_cache_f32 = k_cache.to(torch.float32)  # [num_pages, page_size, num_kv_heads, head_dim]
    v_cache_f32 = v_cache.to(torch.float32)  # [num_pages, page_size, num_kv_heads, head_dim]

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        last_page_len = int(kv_last_page_len[b].item())

        if page_start >= page_end:
            output[b].zero_()
            continue

        page_ids = kv_indices[page_start:page_end].to(torch.long)
        num_pages_for_seq = page_ids.shape[0]

        if num_pages_for_seq == 0:
            output[b].zero_()
            continue

        # Gather all K and V tokens for this sequence
        # Full pages have page_size tokens, last page has last_page_len tokens
        num_full_pages = num_pages_for_seq - 1
        total_tokens = num_full_pages * page_size + last_page_len

        if total_tokens == 0:
            output[b].zero_()
            continue

        # Gather K and V from pages
        k_batch = torch.zeros((total_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device)
        v_batch = torch.zeros((total_tokens, num_kv_heads, head_dim), dtype=torch.float32, device=device)

        token_idx = 0
        for p_idx, page_id in enumerate(page_ids):
            if p_idx < num_full_pages:
                # Full page
                k_batch[token_idx:token_idx + page_size] = k_cache_f32[page_id]
                v_batch[token_idx:token_idx + page_size] = v_cache_f32[page_id]
                token_idx += page_size
            else:
                # Last page (partial)
                k_batch[token_idx:token_idx + last_page_len] = k_cache_f32[page_id, :last_page_len]
                v_batch[token_idx:token_idx + last_page_len] = v_cache_f32[page_id, :last_page_len]
                token_idx += last_page_len

        q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]

        for h in range(num_qo_heads):
            kv_head = h // gqa_ratio

            q_head = q_batch[h]  # [head_dim]
            k_head = k_batch[:, kv_head]  # [total_tokens, head_dim]
            v_head = v_batch[:, kv_head]  # [total_tokens, head_dim]

            logits = torch.matmul(q_head, k_head.T)  # [total_tokens]
            logits_scaled = logits * sm_scale

            lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

            attn = torch.softmax(logits_scaled, dim=-1)  # [total_tokens]
            out_head = torch.matmul(attn, v_head)  # [head_dim]
            output[b, h] = out_head.to(torch.bfloat16)

    return output, lse
