"""Standalone DSA top-k=2048 reference (no cross-file imports)."""

torch = __import__("torch")

PAGE_SIZE = 64
NUM_INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
TOPK = 2048


def _make_k_index_cache_fp8(num_pages, device):
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn not available")

    fp8 = torch.randn((num_pages, PAGE_SIZE, INDEX_HEAD_DIM), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8)

    scales = torch.rand((num_pages, PAGE_SIZE, 1), device=device, dtype=torch.float32) * 0.5 + 0.5
    scale_bytes = scales.view(torch.uint8)

    kv_flat = torch.empty((num_pages, PAGE_SIZE * (INDEX_HEAD_DIM + 4)), device=device, dtype=torch.uint8)
    kv_flat[:, : PAGE_SIZE * INDEX_HEAD_DIM] = fp8_bytes.reshape(num_pages, -1)
    kv_flat[:, PAGE_SIZE * INDEX_HEAD_DIM :] = scale_bytes.reshape(num_pages, -1)

    return kv_flat.view(num_pages, PAGE_SIZE, 1, INDEX_HEAD_DIM + 4).view(torch.int8)


def dequant_fp8_kv_cache(k_index_cache_fp8):
    k_index_cache_fp8 = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k_index_cache_fp8.shape
    head_dim = head_dim_sf - 4

    kv_flat = k_index_cache_fp8.view(num_pages, page_size * head_dim_sf)

    fp8_bytes = kv_flat[:, : page_size * head_dim].contiguous()
    fp8_tensor = fp8_bytes.view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_float = fp8_tensor.to(torch.float32)

    scale_bytes = kv_flat[:, page_size * head_dim :].contiguous()
    scale = scale_bytes.view(num_pages, page_size, 4).view(torch.float32)

    return fp8_float * scale


@torch.no_grad()
def _run_reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    _, page_size, _, _ = k_index_cache_fp8.shape

    assert num_index_heads == NUM_INDEX_HEADS
    assert index_head_dim == INDEX_HEAD_DIM
    assert page_size == PAGE_SIZE

    device = q_index_fp8.device
    q = q_index_fp8.to(torch.float32)
    k_all = dequant_fp8_kv_cache(k_index_cache_fp8)

    topk_indices = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=device)

    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue

        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)

        k_paged = k_all[page_indices]
        k = k_paged.reshape(-1, index_head_dim)[:seq_len]

        scores = q[b] @ k.T
        scores_relu = torch.relu(scores)
        final_scores = (scores_relu * weights[b][:, None]).sum(dim=0)

        actual_topk = min(TOPK, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)

        page_idx_per_token = topk_idx // page_size
        offset_per_token = topk_idx % page_size
        global_page_idx = page_indices[page_idx_per_token]
        topk_tokens = global_page_idx * page_size + offset_per_token

        topk_indices[b, :actual_topk] = topk_tokens.to(torch.int32)

    return (topk_indices,)


def generate_input(batch=1, num_pages=16, seq_len=2048, seed=42, __real_workload__=None):
    device = torch.device("cuda")
    torch.manual_seed(seed)

    if __real_workload__ is not None:
        workload = __real_workload__.get("workload", __real_workload__)
        axes = workload.get("axes", {}) if isinstance(workload, dict) else {}
        batch = int(axes.get("batch_size", axes.get("batch", batch)))
        num_pages = int(max(1, axes.get("num_pages", num_pages)))
        seq_len = int(min(axes.get("seq_len", axes.get("num_kv_indices", seq_len)), num_pages * PAGE_SIZE))

    total_tokens = num_pages * PAGE_SIZE
    if seq_len > total_tokens:
        raise ValueError(f"seq_len ({seq_len}) exceeds total tokens ({total_tokens}).")

    q_index_fp8 = torch.randn((batch, NUM_INDEX_HEADS, INDEX_HEAD_DIM), device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    k_index_cache_fp8 = _make_k_index_cache_fp8(num_pages, device)
    weights = torch.rand((batch, NUM_INDEX_HEADS), device=device, dtype=torch.float32)
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).unsqueeze(0).repeat(batch, 1)

    return (q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)


def ref_kernel(data):
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    return _run_reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table)


def check_implementation(data, output):
    expected = ref_kernel(data)
    got = output[0]
    ref = expected[0]
    if got.shape != ref.shape:
        return False, f"shape mismatch: got={tuple(got.shape)} ref={tuple(ref.shape)}"
    same = torch.equal(got, ref)
    if same:
        return True, ""
    diff = (got != ref).nonzero()
    idx = tuple(diff[0].tolist()) if diff.numel() else ()
    return False, f"mismatch found! first mismatch at {idx}: got={got[idx].item()} ref={ref[idx].item()}"
