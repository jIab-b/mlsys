"""Standalone DSA sparse-attn top-k=2048 reference (no cross-file imports)."""

torch = __import__("torch")
math = __import__("math")

PAGE_SIZE = 64
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOPK = 2048


@torch.no_grad()
def _run_reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    _, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    assert num_qo_heads == NUM_QO_HEADS
    assert head_dim_ckv == HEAD_DIM_CKV
    assert head_dim_kpe == HEAD_DIM_KPE
    assert page_size == PAGE_SIZE
    assert topk == TOPK

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
        attn = torch.softmax(logits_scaled, dim=-1)
        output[t] = (attn @ kc).to(torch.bfloat16)

    return output, lse


def generate_input(batch=1, num_pages=16, seq_len=2048, seed=42, __real_workload__=None):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    if __real_workload__ is not None:
        workload = __real_workload__.get("workload", __real_workload__)
        axes = workload.get("axes", {}) if isinstance(workload, dict) else {}
        batch = int(axes.get("num_tokens", axes.get("batch_size", axes.get("batch", batch))))
        num_pages = int(max(1, axes.get("num_pages", num_pages)))
        seq_len = int(min(axes.get("seq_len", axes.get("num_kv_indices", seq_len)), num_pages * PAGE_SIZE))

    total_tokens = num_pages * PAGE_SIZE
    if seq_len > total_tokens:
        raise ValueError(f"seq_len ({seq_len}) exceeds total tokens ({total_tokens}).")

    q_nope = torch.randn((batch, NUM_QO_HEADS, HEAD_DIM_CKV), device=device, dtype=torch.bfloat16)
    q_pe = torch.randn((batch, NUM_QO_HEADS, HEAD_DIM_KPE), device=device, dtype=torch.bfloat16)
    ckv_cache = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_CKV), device=device, dtype=torch.bfloat16)
    kpe_cache = torch.randn((num_pages, PAGE_SIZE, HEAD_DIM_KPE), device=device, dtype=torch.bfloat16)

    sparse_indices = torch.randint(0, total_tokens, (batch, TOPK), device=device, dtype=torch.int32)
    sm_scale = 1.0 / math.sqrt(128 + HEAD_DIM_KPE)

    return (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def ref_kernel(data):
    return _run_reference(*data)


def check_implementation(data, output):
    exp_out, exp_lse = ref_kernel(data)
    got_out, got_lse = output

    if got_out.shape != exp_out.shape or got_lse.shape != exp_lse.shape:
        return False, f"shape mismatch: got={tuple(got_out.shape)},{tuple(got_lse.shape)} ref={tuple(exp_out.shape)},{tuple(exp_lse.shape)}"

    ok_out = torch.allclose(got_out.float(), exp_out.float(), rtol=1e-3, atol=1e-3)
    ok_lse = torch.allclose(got_lse.float(), exp_lse.float(), rtol=1e-3, atol=1e-3)
    if ok_out and ok_lse:
        return True, ""
    return False, "mismatch found! custom implementation doesn't match reference"
