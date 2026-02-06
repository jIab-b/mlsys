// op_sig.h
// Reference for graph annotations used by the static validator.
// This file is documentation-only; it does not declare or define symbols.
//
// Annotation syntax (inline in CUDA):
//   // @op <op_name> key=value key=value ...
//   // @op key=value ...            (continuation for the previous @op)
//   // @buffer name=<id> space=tmem|smem|gmem|rmem cols=<int> dtype=<str> ...
//   // @barrier name=<id> scope=cta|cluster count=<int>
//   // @desc name=<id> buf=<buffer?> swizzle=<...> major=<...> sbo=<int> lbo=<int> ...
//   // @kernel name=<id> smem_bytes=<int> smem_static=<int> smem_dynamic=<int>
//   // @loop var=<id> iters=<int>
//   // @endloop
//
// Common metadata keys:
//   warp_id=<int or expr>   lane_id=<int|"elect">
//   issue / issue_scope=<one_thread|one_warp|all_warps|host>
//   when=<expr>             (validation-only conditional)
//
// NOTE: The validator enforces only what is specified in graph/static_validator.py
// and graph/state_machine/static.py. Keep annotations aligned with those constraints.
//
// ---------------------------------------------------------------------------
// tcgen05 (tmem) ops
// ---------------------------------------------------------------------------
// tcgen05_alloc
//   Required: tmem, cols
//   Optional: cta_group (1|2), scope=one_warp
//   Notes: cols in [32, 512], power-of-2, non-increasing across allocs.
//
// tcgen05_dealloc
//   Required: tmem, cols
//   Optional: cta_group (1|2), scope=one_warp
//   Notes: cols must match last alloc.
//
// tcgen05_cp (aliases: tcgen05_cp_*, tcgen05_cp_nvfp4)
//   Required: tmem
//   Optional: shape, tile, desc, smem_buf, tmem_offset, cols, cta_group, issue
//   Metadata: warp_id, lane_id (typically lane_id=elect)
//
// tcgen05_mma (aliases: tcgen05_mma_*, tcgen05_mma_nvfp4)
//   Required: tmem
//   Optional: shape, desc_a, desc_b, tile, cta_group, issue
//   Metadata: warp_id, lane_id (typically lane_id=elect)
//
// tcgen05_ld / tcgen05_st (aliases: tcgen05_ld_*, tcgen05_st)
//   Required: tmem
//   Optional: shape, num, cta_group
//   Metadata: warp_id, lane_id (lane_id cannot be "elect")
//
// tcgen05_commit / tcgen05_commit_mcast
//   Required: bar  (and cta_mask for commit_mcast)
//   Optional: cta_group, cta_mask
//   Metadata: warp_id, lane_id (typically lane_id=elect)
//
// tcgen05_wait / tcgen05_wait_ld / tcgen05_wait_st / tcgen05_fence*
//   No required args. Use to satisfy pending ld/st or ordering constraints.
//
// ---------------------------------------------------------------------------
// mbarrier / cluster ops
// ---------------------------------------------------------------------------
// mbarrier_init
//   Required: bar, count
//   Optional: scope=cta|cluster
//
// mbarrier_arrive_expect_tx / mbarrier_arrive_expect_tx_cta
//   Required: bar, size
//   Optional: scope=cta|cluster
//
// mbarrier_wait / mbarrier_wait_relaxed / mbarrier_wait_ticks
//   Required: bar, phase
//   Optional: scope=cta|cluster, ticks (for wait_ticks)
//
// mbarrier_fence_init_release
//   No required args.
//
// barrier_cluster_arrive / barrier_cluster_wait
//   No required args.
//
// ---------------------------------------------------------------------------
// TMA / cp.async bulk ops
// ---------------------------------------------------------------------------
// tma_gmem2smem
//   Required: bar, size
//   Optional: dst_align, src_align
//
// tma_1d_gmem2smem / tma_2d_gmem2smem / tma_3d_gmem2smem
//   Required: bar, tmap
//   Optional: dim, size
//
// tma_*_gmem2smem_mcast
//   Required: bar, tmap, cta_mask
//   Optional: dim, size
//
// cp_async_bulk_prefetch(_1d/_2d/_3d)
//   Required: addr+size (or tmap/x/y/z for *_1d/2d/3d)
//
// ---------------------------------------------------------------------------
// PTX helper ops
// ---------------------------------------------------------------------------
// ptx_laneid, ptx_activemask, ptx_elect_one_sync, ptx_elect_sync
//   No required args.
//
// ptx_bar_sync
//   Required: bar_id, count
//
// ---------------------------------------------------------------------------
// Host-side metadata ops
// ---------------------------------------------------------------------------
// cute_tmap
//   Required: name
//   Optional: rank, global_height, global_width, shared_height, shared_width
//
// ---------------------------------------------------------------------------
// Descriptor & buffer notes
// ---------------------------------------------------------------------------
// @desc and @buffer metadata is used by the validator for swizzle/major/LUT checks.
// Common keys:
//   swizzle, major, sbo, lbo, base_offset, stride, leading
//
// For allowed shapes, swizzles, and further constraints:
//   see graph/static_validator.py (TCGEN05_*, TMA_* sets).
