"""Standalone DSA sparse-attn top-k=2048 submission (no cross-file imports)."""

torch = __import__("torch")
math = __import__("math")


@torch.no_grad()
def custom_kernel(data):
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale = data

    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]

    kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.full((num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

    for t in range(num_tokens):
        indices = sparse_indices[t]
        valid_indices = indices[indices != -1]
        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        tok_idx = valid_indices.to(torch.long)
        kc = kc_all[tok_idx]
        kp = kp_all[tok_idx]
        qn = q_nope[t].to(torch.float32)
        qp = q_pe[t].to(torch.float32)

        logits = (qn @ kc.T) + (qp @ kp.T)
        logits_scaled = logits * sm_scale

        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
        output[t] = (torch.softmax(logits_scaled, dim=-1) @ kc).to(torch.bfloat16)

    return output, lse


def compile_kernel():
    return None
