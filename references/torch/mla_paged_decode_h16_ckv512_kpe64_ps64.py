"""Reference implementation for mla_paged_decode_h16_ckv512_kpe64_ps64.
Auto-extracted from flashinfer-bench definitions.
Op type: mla_paged
"""
import math
import torch


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Check constants
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert page_size == 64

    # Check constraints
    assert len_indptr == batch_size + 1
    assert num_kv_indices == kv_indptr[-1].item()

    device = q_nope.device

    ckv_cache_f32 = ckv_cache.to(torch.float32)  # [num_pages, page_size, head_dim_ckv]
    kpe_cache_f32 = kpe_cache.to(torch.float32)  # [num_pages, page_size, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        last_page_len = int(kv_last_page_len[b].item())

        if page_beg >= page_end:
            output[b].zero_()
            continue

        page_ids = kv_indices[page_beg:page_end].to(torch.long)
        num_pages_for_seq = page_ids.shape[0]

        # Calculate total tokens
        num_full_pages = num_pages_for_seq - 1
        L_tokens = num_full_pages * page_size + last_page_len

        if L_tokens <= 0:
            output[b].zero_()
            continue

        # Gather Kc and Kp from pages
        Kc = torch.zeros((L_tokens, head_dim_ckv), dtype=torch.float32, device=device)
        Kp = torch.zeros((L_tokens, head_dim_kpe), dtype=torch.float32, device=device)

        token_idx = 0
        for p_idx, page_id in enumerate(page_ids):
            if p_idx < num_full_pages:
                Kc[token_idx:token_idx + page_size] = ckv_cache_f32[page_id]
                Kp[token_idx:token_idx + page_size] = kpe_cache_f32[page_id]
                token_idx += page_size
            else:
                Kc[token_idx:token_idx + last_page_len] = ckv_cache_f32[page_id, :last_page_len]
                Kp[token_idx:token_idx + last_page_len] = kpe_cache_f32[page_id, :last_page_len]
                token_idx += last_page_len

        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]

        logits = (qn @ Kc.T) + (qp @ Kp.T)  # [num_qo_heads, L_tokens]
        logits_scaled = logits * sm_scale

        # Compute 2-base LSE
        lse[b] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)

        attn = torch.softmax(logits_scaled, dim=-1)  # [num_qo_heads, L_tokens]
        out = attn @ Kc  # [num_qo_heads, head_dim_ckv]
        output[b] = out.to(torch.bfloat16)

    return output, lse
