"""Reference implementation for top_p_sampling_from_probs_v151936.
Auto-extracted from flashinfer-bench definitions.
Op type: sampling
"""
import torch

@torch.no_grad()
def run(probs, top_p):
    batch_size, vocab_size = probs.shape
    device = probs.device

    # Check constants
    assert vocab_size == 151936

    probs = probs.to(torch.float32)
    out = torch.empty(batch_size, dtype=torch.int64, device=device)

    for i in range(batch_size):
        row = probs[i]
        p = float(top_p[i].item())
        
        if p <= 0.0:
            # Degenerate to argmax
            out[i] = torch.argmax(row).to(torch.int64)
            continue

        if p < 1.0:
            vals, idx = torch.sort(row, descending=True)
            cdf = torch.cumsum(vals, dim=0)

            # Shift mask to keep the first token that crosses p
            to_remove = cdf > p
            to_remove[1:] = to_remove[:-1].clone()
            to_remove[0] = False
            keep = ~to_remove
            keep_idx = idx[keep]

            # Build filtered distribution in original index space
            filtered = torch.zeros_like(row)
            filtered[keep_idx] = row[keep_idx]
            row = filtered / filtered.sum()

        out[i] = torch.multinomial(row, 1, replacement=True).squeeze(0)

    return out
