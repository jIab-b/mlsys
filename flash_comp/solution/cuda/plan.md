# DSA Indexer PTX Plan (LLM Handoff)

## Goal
Build `dsa_topk_indexer` in `flash_comp/solution/cuda/dsa_index.cu` for Blackwell with low-overhead DPS binding.

Inputs:
- `q_index_fp8 [B,64,128]`
- `k_index_cache_fp8 [num_pages,64,1,132]` (deep_gemm packed fp8+scale bytes)
- `weights [B,64]`
- `seq_lens [B]`
- `block_table [B,max_num_pages]`

Output:
- `topk_indices [B,2048] int32`, padded with `-1`.

## Ground Truth / Layout Clarifications
- `block_table[b, :]` is per-sequence page indirection, not shared globally across all batch rows.
- A page is physically contiguous in memory.
- Sequence pages are logically non-contiguous in global memory due to page indirection.
- Effective compute is dense on each loaded token tile, but the loaded token set itself is sparse via page table selection.
- No explicit matrix transpose is required if descriptors/idesc are encoded correctly.

## Target Kernel Shape
- Per sequence (`b`), compute score matrix:
  - `Q[b]`: `[64,128]`
  - `K_seq`: `[seq_len,128]`
  - `scores`: `[64,seq_len]`
- Head-reduced score:
  - `final[t] = sum_h(relu(scores[h,t]) * weights[b,h])`
- Top-k:
  - exact `k=2048`, output global token IDs (`page_id * 64 + offset`), fill tail with `-1`.

## Tile / Pipeline Design
- One CTA handles one sequence row `b` (initial design; tune later).
- `M = 64` fixed heads.
- `K = 128` head dim, split as 4 slices of 32 for MMA stepping.
- `Ntile = 64` (page-aligned).
- `num_stages = 10` on K-tile pipeline (tokens per stage = 64).
- Stage tail policy: zero-fill invalid tokens in partial last tile, mask in compute/topk.
- Keep `Q` staged for entire CTA lifetime (reuse across all `Ntile` iterations).

## PTX Instructions To Use (Current Library Names)
From `ptx_lib` (copy wrappers into solution files when integrating):

1. TMA / async tensor loads:
- `tma_2d_gmem2smem` (`ptx_lib/ptx_tma.cuh`)
- `tma_2d_gmem2smem_gather4` (`ptx_lib/ptx_tma.cuh`)
- Optional prefetch: `tma_prefetch_2d`
- Optional fallback for scalar metadata: regular `ld.global` inline PTX or CUDA loads

2. Tensor Core GEMM:
- `tcgen05_mma_f16_ss` / `tcgen05_mma_f16_ts` (`ptx_lib/ptx_tcgen05_mma.cuh`)
- Use encoded `idesc` for runtime-selectable `Ntile` and slice position.

3. TMEM load/store:
- `tcgen05_ld_*` family (`ptx_lib/ptx_tcgen05_ldst.cuh`) to read accum/results from TMEM.
- `tcgen05_st` only if needed for explicit TMEM writes.

4. Tensor Core pipeline sync/control:
- `tcgen05_commit`, `tcgen05_wait_ld`, `tcgen05_wait_st`
- `tcgen05_fence_before_thread_sync`, `tcgen05_fence_after_thread_sync`
- `tcgen05_alloc`, `tcgen05_dealloc` for TMEM management

5. Barriers:
- mbarrier/TMA completion flow (from `ptx_lib/ptx_mbarrier.cuh`)
- CTA sync (`bar.sync`) only where required for stage ownership.

## Gather Strategy
- Gather is needed when the next token rows map to non-consecutive page rows.
- Support effective gather widths `1..4` by policy:
  - Native path: `gather4`.
  - For `<4` rows, duplicate valid rows into unused gather slots and mask out those lanes/tokens in compute/topk.
- This keeps one codepath for gather control while handling small/tail cases.

## Main Loop Skeleton
1. Load metadata (`seq_len`, page span, block table base).
2. Stage `Q[b,:,:]` once into smem/tmem-friendly layout.
3. Prepare `final_scores` backing store (fp32):
   - if `seq_len <= seq_cap_smem`: use smem `final_scores_smem`.
   - else: use global spill buffer with L2 cache hint stores/loads.
4. Initialize running topk structure for this sequence.
5. For token tiles over sequence:
   - Build gather row list (or contiguous row coordinates).
   - Issue TMA async load into smem stage buffer (`ping/pong`).
   - Wait/commit previous stage as needed.
   - Run MMA over 4 `K` slices, accumulating in TMEM.
   - `tcgen05_ld` accumulator fragments.
   - Apply ReLU + head weights reduction to fp32 scalar per token.
   - Store scalar score to selected backing store (smem or L2-hinted global).
6. Final topk pass over stored fp32 scores:
   - read from smem or reload from global spill.
   - compute exact topk=2048 indices.
7. Write `topk_indices[b,:]`, pad with `-1`.

## Shared Memory / Buffer Budget (Planning Numbers)
Assume `smem_total = 228 KiB = 233,472 B`.

Fixed buffers:
- K stage buffers: `10 * (64 * 132) = 84,480 B`
- Q staged once: `64 * 128 = 8,192 B`
- Weights staged once: `64 * 4 = 256 B`
- Topk scratch (`scores fp32 + indices int32`, 2048 each): `16,384 B`
- Misc/barriers/metadata reserve: `4,096 B`

Fixed subtotal: `113,408 B`

Maximum preallocated fp32 final-score buffer in smem:
- `233,472 - 113,408 = 120,064 B`
- `120,064 / 4 = 30,016` fp32 tokens

Implication:
- For common DSA max `seq_len <= 16,384`, full fp32 final-score buffer fits in smem.
- Overflow path remains implemented for future larger `seq_len`.

## Final Score Dtype Policy
- Use fp32 for stored final scores and topk ranking path.
- Do not use fp8 for final score storage/ranking (too much quantization error for stable cutoff ordering).
- fp16 may be explored only as an optional fast mode, not default correctness mode.

## Global Spill Path (Overflow)
- If `seq_len > seq_cap_smem`, spill final scores to global workspace.
- Use L2-hinted global stores during producer pass; reload with L2-hinted loads for final topk pass.
- Workspace allocation policy:
  - allocate once outside hot path (module init / compile hook),
  - reuse across launches to avoid per-call alloc overhead.

## Binding / Integration Requirements
- Keep DPS-only call path in Python (`topk_indices` is input-output buffer).
- `binding.py` should only forward arguments and call launcher.
- Optional `compile_kernel()` should trigger one-time module build/warmup.
- Keep `dsa_index.cu` symbol stable:
  - `dsa_topk_indexer_launch(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)`

## Milestones
1. Correctness kernel (no tcgen/TMA optimization).
2. PTX staged load + MMA path with fixed `Ntile=64`.
3. Add variable `Ntile` (`64/128/256`) and gather policy.
4. Streaming topk + tail masking + global index mapping.
5. Tune by seq_len buckets, verify perf/correctness.
