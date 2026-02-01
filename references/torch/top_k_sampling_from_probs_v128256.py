"""Reference implementation for top_k_sampling_from_probs_v128256.
Auto-extracted from flashinfer-bench definitions.
Op type: sampling
"""
import torch

@torch.no_grad()
def run(probs, top_k):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    assert vocab_size == 128256

    probs = probs.to(torch.float32)
    samples = torch.empty(batch_size, dtype=torch.int64, device=device)

    for i in range(batch_size):
        row = probs[i]
        k = int(top_k[i].item())

        # No filtering on invalid k
        if 0 < k < vocab_size:
            idx_sorted = torch.argsort(row, descending=True)
            keep_idx = idx_sorted[:k]

            filtered = torch.zeros_like(row)
            filtered[keep_idx] = row[keep_idx]

            row = filtered / filtered.sum()

        samples[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return samples

