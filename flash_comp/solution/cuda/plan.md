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
- `Ntile` variable by seq bucket: `64`, `128`, `256` (cap at 256).
- Double-buffer over `Ntile` (stage `s` compute while stage `s+1` loads).
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
3. Initialize running topk structure for this sequence.
4. For token tiles over sequence:
   - Build gather row list (or contiguous row coordinates).
   - Issue TMA async load into smem stage buffer (`ping/pong`).
   - Wait/commit previous stage as needed.
   - Run MMA over 4 `K` slices, accumulating in TMEM.
   - `tcgen05_ld` accumulator fragments.
   - Apply ReLU + head weights reduction.
   - Update streaming topk (score + global token index).
5. Finalize topk buffer, write `topk_indices[b,:]`, pad with `-1`.

## Shared Memory / Buffer Budget (Planning Numbers)
Assuming `Ntile=256` and packed K rows at `132 B/token`:
- K stage buffer (single): `256 * 132 = 33,792 B` (~33.0 KiB)
- K stage buffer (double): `67,584 B` (~66.0 KiB)
- Q tile staged once (fp8): `64 * 128 = 8,192 B` (~8.0 KiB)
- Weights: `64 * 4 = 256 B`
- Topk scratch estimate:
  - if storing `(score float32, idx int32)` for `k=2048`: `2048 * 8 = 16,384 B` (~16.0 KiB)
- Total planning footprint with full double-buffer + simple topk scratch is high; likely requires:
  - smaller `Ntile` on some paths, or
  - register/partial topk staging, or
  - reduced on-chip topk state.

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
