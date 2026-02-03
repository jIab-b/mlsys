# Grouped GEMM Plan (Concise)

## 1) Core issue
- Runtime OOB due to **full‑tile TMA** on irregular M/N (e.g., M=40/56/72/etc). Raw `tma_gmem2smem` (and packed TMA) assumes full tiles and **no OOB protection**.
- For packed FP4 tensor maps, CUDA rules require **dim0 alignment (multiple of 128)** and **no OOB fill** (packed types). So **pure TMA requires padding**.

## 2) Valid scale layouts (per PTX)
- f233 (SFA) and f242 (SFB) **are structurally identical**; only the **axis meaning** differs (rows of A vs columns of B).
- Scale layout is **per‑128 row/col block**; MMA expects those slots populated. Partial blocks must be **padded** with neutral scales.

## 3) Data movement hierarchy (host‑side selection)
1) **Regular swizzled tensor maps** for A/B (best path)
   - Requires M (and/or N) aligned to 128 for packed FP4.
   - Scales: either raw `tma_gmem2smem` from pointers or INT64 tmap (below).
2) **Scale factors via INT64 tensor map** (as in comp_subs/dual_gemm2/3)
   - Useful when scale layout/boxDim restrictions are awkward.
   - Still requires correct bounds/padding for A/B.
3) **cp.async.bulk (no tensor map)** for irregular shapes
   - Raw pointer bulk copies. **No OOB guard**; must pad or use a tail kernel with guarded loads.

## 4) Correctness strategies for irregular shapes
Option A — **Pad + pack (recommended)**
- Pad A/B to full tiles (M→128, N→BLOCK_N), fill zeros.
- Pad SFA/SFB to full 128 row/col blocks with neutral scales (1.0).
- Kernel runs full‑tile TMA safely; epilogue crops C back to original shape.

Option B — **Main + tail kernels**
- Main kernel: TMA full tiles only.
- Tail kernel: non‑TMA guarded loads for remainder tiles.
- Keeps main path fast; avoids full padding overhead.

## 5) Swapping A/B (“treat N as M”) — when/why
- Conceptually valid: compute **Cᵀ = Bᵀ · Aᵀ**.
- Must **swap scales** (SFB as scale_A, SFA as scale_B) and adjust indexing.
- Output is transposed; either accept **Cᵀ** or add transpose epilogue.
- Tradeoff: helps with small‑M tail, but makes **A huge**, often worse for cache/L2 reuse.

## 6) Output layout & epilogue
- Writing transposed output directly is fine if consumer accepts **Cᵀ**.
- If original layout needed, use **smem transpose epilogue** for coalesced stores.
- Pure reg‑transpose isn’t enough (needs cross‑thread exchange); smem is standard.

## 7) “Fuse M blocks” idea (e.g., 40 + 56)
- Possible **only if** you pack into one 128‑row tile and pad the rest.
- Must place each row’s scales into correct M0..M127 slots per f233/f242.
- Useful only when groups share **same N & K**; otherwise packing overhead dominates.

## 8) Validator notes (optional future work)
- Add static checks for: M/N alignment vs BLOCK_M/BLOCK_N and scale block layout.
- Flag kernels that issue full‑tile TMA without a declared remainder path or padding.

## 9) Practical next steps
- Implement host‑side decision table:
  - If M%128==0 and N%BLOCK_N==0 → full TMA path.
  - Else if padding allowed → pad+pack path.
  - Else → main+tail path (tail uses cp.async.bulk or guarded loads).
- Consider INT64 tmap for scales when boxDim restrictions bite.
