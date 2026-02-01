"""Reference implementation for top_k_top_p_sampling_from_probs_v151936.
Auto-extracted from flashinfer-bench definitions.
Op type: sampling
"""
import torch

@torch.no_grad()
def run(probs, top_k, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    assert vocab_size == 151936

    probs = probs.to(torch.float32)
    samples = torch.empty(batch_size, dtype=torch.int64, device=device)

    for i in range(batch_size):
        row = probs[i]
        k = int(top_k[i].item())
        p = float(top_p[i].item())

        # Apply top-k filtering
        if 0 < k < vocab_size:
            idx_sorted = torch.argsort(row, descending=True)
            keep_idx_k = idx_sorted[:k]
            filtered_k = torch.zeros_like(row)
            filtered_k[keep_idx_k] = row[keep_idx_k]
            row = filtered_k / filtered_k.sum()

        # Then apply top-p filtering
        if p <= 0.0:
            samples[i] = torch.argmax(row).to(torch.int64)
            continue

        if p < 1.0:
            vals, idx = torch.sort(row, descending=True)
            cdf = torch.cumsum(vals, dim=0)

            to_remove = cdf > p
            if vocab_size > 1:
                to_remove[1:] = to_remove[:-1].clone()
                to_remove[0] = False

            keep_idx_p = idx[~to_remove]
            filtered_p = torch.zeros_like(row)
            filtered_p[keep_idx_p] = row[keep_idx_p]
            row = filtered_p / filtered_p.sum()

        # sample
        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return samples

