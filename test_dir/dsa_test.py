"""DSA index submission."""

import os
import re

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


_module = None
_DEBUG_PRINTED_ONCE = False


cuda_src = """
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <climits>
#include <cstdint>

// ---------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------
constexpr int kNumHeads = 64;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 64;

constexpr int kPayloadBytesPerToken = 128;
constexpr int kScaleBytesPerToken = 4;
constexpr int kRowBytes = 132;
constexpr int kPageBytes = kPageSize * kRowBytes;      // 8448
constexpr int kPackedFp8Bytes = kPageSize * kHeadDim;  // 8192

constexpr int kStageTokens = 64;
constexpr int kNumStages = 8;
constexpr int kNumTmemSlots = 8;
static_assert(kNumTmemSlots * kStageTokens <= 512, "TMEM columns exceed hardware limit");

constexpr int kNumEpTopkWarpgroups = 4;
constexpr int kWarpsPerWarpgroup = 4;
constexpr int kEpTopkWarpCount = kNumEpTopkWarpgroups * kWarpsPerWarpgroup;  // 16
constexpr int kNumEpilogueWarps = 8;
constexpr int kNumTopkWarps = 8;
constexpr int kProducerWarp = kEpTopkWarpCount;
constexpr int kMmaWarp = kProducerWarp + 1;
constexpr int kNumTopkConsumeWarps = 2;
constexpr int kTopkUpdateWarpBase = kNumTopkConsumeWarps;
constexpr int kNumTopkUpdateWarps = kNumTopkWarps - kNumTopkConsumeWarps;
constexpr int kNumWarps = kMmaWarp + 1;
constexpr int kThreadsPerBlock = kNumWarps * 32;

constexpr int kTopk = 2048;
constexpr int kPreDebugSamples = 4;
constexpr int kPostDebugSamples = 12;
constexpr int kDebugSamples = kPostDebugSamples;

constexpr int kMmaK = 32;
constexpr int kMmaIters = kHeadDim / kMmaK;  // 4
constexpr int kDesiredDynamicSmemBytes = 228 * 1024 - 2048;

// ---------------------------------------------------------------------------------
// Shared memory layout — all smem allocations in one struct
// ---------------------------------------------------------------------------------
struct __align__(1024) SmemLayout {
    // Per-stage K data
    uint8_t k_stage_payload[kNumStages * kStageTokens * kPayloadBytesPerToken];
    float k_stage_scale[kNumStages * kStageTokens];

    // Q + weights
    uint8_t q_stage[kNumHeads * kHeadDim];
    float w_stage[kNumHeads];

    // Stage metadata
    int stage_page_idx[kNumStages];
    int stage_valid_tokens[kNumStages];

    // Per-stage phase counters
    int tma_phase[kNumStages];
    int mma_phase[kNumStages];
    int topk_phase[kNumStages];
    int tmem_reuse_phase_mma[kNumTmemSlots];

    // Barrier arrays
    uint64_t tma_mbar[kNumStages];
    uint64_t mma_mbar[kNumStages];
    uint64_t epi_mbar[kNumStages];
    uint64_t topk_mbar[kNumStages];
    uint64_t tmem_reuse_mbar[kNumTmemSlots];
    uint64_t q_mbar;

    // Epilogue candidate buffers
    float stage_tile_scores[kNumStages * kStageTokens];
    float stage_tile_pair_partial[kNumStages * kStageTokens];
    int stage_tile_ids[kNumStages * kStageTokens];

    // Top-k buffers
    float topk_stage_compact_scores[kStageTokens];
    int topk_stage_compact_ids[kStageTokens];
    int topk_stage_compact_counts[2];
    float topk_local_heap_scores[kNumTopkUpdateWarps * kTopk];
    int topk_local_heap_ids[kNumTopkUpdateWarps * kTopk];
    int topk_local_heap_sizes[kNumTopkUpdateWarps];
    int topk_local_heap_ready[kNumTopkUpdateWarps];
    float topk_heap_scores[kTopk];
    int topk_heap_ids[kTopk];
    float topk_cutoff;

    // TMEM scratch
    int tmem_addr_scratch;
};

// Precomputed PTX smem addresses
struct SmemAddrs {
    int k_stage_payload;
    int k_stage_scale;
    int q_stage;
    int tma_mbar;
    int mma_mbar;
    int epi_mbar;
    int topk_mbar;
    int tmem_reuse_mbar;
    int q_mbar;
    int tmem_addr_scratch;
};

__device__ inline SmemAddrs init_smem_addrs(SmemLayout* s) {
    const int base = static_cast<int>(__cvta_generic_to_shared(s));
    auto off = [&](const void* field) {
        return base + static_cast<int>(
            reinterpret_cast<const unsigned char*>(field) -
            reinterpret_cast<const unsigned char*>(s));
    };
    SmemAddrs a;
    a.k_stage_payload = off(s->k_stage_payload);
    a.k_stage_scale   = off(s->k_stage_scale);
    a.q_stage         = off(s->q_stage);
    a.tma_mbar        = off(s->tma_mbar);
    a.mma_mbar        = off(s->mma_mbar);
    a.epi_mbar        = off(s->epi_mbar);
    a.topk_mbar       = off(s->topk_mbar);
    a.tmem_reuse_mbar = off(s->tmem_reuse_mbar);
    a.q_mbar          = off(&s->q_mbar);
    a.tmem_addr_scratch = off(&s->tmem_addr_scratch);
    return a;
}

// ---------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------
__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{ .reg .pred %%px; elect.sync _|%%px, %1; @%%px mov.s32 %0, 1; }"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

__device__ inline bool _is_ep_warp(int warp_id) {
    // EP uses the first 8 warps (0..7).
    return warp_id < kNumEpilogueWarps;
}

__device__ inline bool _is_topk_warp(int warp_id) {
    return (warp_id >= kNumEpilogueWarps) && (warp_id < kEpTopkWarpCount);
}

__device__ inline int _topk_warp_rank(int warp_id) {
    return warp_id - kNumEpilogueWarps;
}

__device__ inline constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_desc_kmajor_swizzle_128b(int smem_addr) {
    const int sbo = 8 * 128;
    return desc_encode(static_cast<uint64_t>(smem_addr)) |
           (desc_encode(static_cast<uint64_t>(sbo)) << 32ULL) |
           (1ULL << 46ULL) |
           (2ULL << 61ULL);
}

__device__ inline uint64_t make_desc_kmajor_noswizzle(int smem_addr) {
    //No-swizzle K-major candidates:
    const int lbo = 64 * 16;   // canonical-style (M=64 => 1024)
    const int sbo = 8 * 16;    // canonical-style (128)
    // const int lbo = 16;        // older interpretation (current)
    // const int sbo = 8 * 128;   // older interpretation (current)
    // const int lbo = 16;
    // const int sbo = 8 * 128;
    return desc_encode(static_cast<uint64_t>(smem_addr)) |
           (desc_encode(static_cast<uint64_t>(lbo)) << 16ULL) |
           (desc_encode(static_cast<uint64_t>(sbo)) << 32ULL) |
           (1ULL << 46ULL) |
           (0ULL << 61ULL);
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbar_addr), "r"(size_bytes)
                 : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(mbar_addr)
                 : "memory");
}

__device__ inline void mbarrier_wait_parity(int mbar_addr, int phase) {
    constexpr uint32_t kSuspendNs = 1000000U;  // 1 ms per try_wait attempt
    constexpr uint64_t kTimeoutNs = 5000000000ULL;  // 5 s total timeout
    uint64_t start_ns = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_ns));

    while (true) {
        uint32_t complete = 0;
        asm volatile(
            "{\\n\\t"
            ".reg .pred p;\\n\\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2, %3;\\n\\t"
            "selp.u32 %0, 1, 0, p;\\n\\t"
            "}"
            : "=r"(complete)
            : "r"(mbar_addr), "r"(phase), "r"(kSuspendNs)
            : "memory");
        if (complete) {
            return;
        }

        uint64_t now_ns = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now_ns));
        if (now_ns - start_ns > kTimeoutNs) {
            asm volatile("trap;");
        }
    }
}

__device__ inline void prepare_stage_metadata_and_scale_tail(
    int tile_id,
    int stage,
    int seq_len,
    int max_num_pages,
    int num_pages,
    const int* block_table_b,
    int* stage_page_idx,
    int* stage_valid_tokens,
    float* k_stage_scale,
    int& page_idx,
    int& valid_tokens
) {
    const int tile_seq_start = tile_id * kStageTokens;
    const int remain = seq_len - tile_seq_start;
    valid_tokens = (remain > 0) ? ((remain < kStageTokens) ? remain : kStageTokens) : 0;

    page_idx = -1;
    if (tile_id >= 0 && tile_id < max_num_pages) {
        page_idx = block_table_b[tile_id];
    }
    if (!(page_idx >= 0 && page_idx < num_pages)) {
        valid_tokens = 0;
    }

    stage_page_idx[stage] = page_idx;
    stage_valid_tokens[stage] = valid_tokens;

    // Zero out scale padding for invalid tokens.
    float* stage_scale = k_stage_scale + stage * kStageTokens;
    for (int tok = valid_tokens; tok < kStageTokens; ++tok) {
        stage_scale[tok] = 0.0f;
    }
}

__device__ inline void tma_3d_gmem2smem(
    int dst_smem_addr,
    const void* tmap_ptr,
    int x,
    int y,
    int z,
    int mbar_addr,
    uint64_t cache_policy
) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;"
        :: "r"(dst_smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
        : "memory");
}

__device__ inline void tma_2d_gmem2smem(
    int dst_smem_addr,
    const void* tmap_ptr,
    int x,
    int y,
    int mbar_addr,
    uint64_t cache_policy
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(dst_smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "l"(cache_policy)
        : "memory");
}

__device__ inline void tcgen05_alloc(int smem_addr, int num_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr), "r"(num_cols));
}

__device__ inline void tcgen05_dealloc(int base_tmem, int num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(base_tmem), "r"(num_cols));
}

__device__ inline void tcgen05_mma_f8f6f4(
    uint32_t tmem_d,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t idesc,
    int accumulate
) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %4, 0;\\n\\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\\n\\t"
        "}"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate),
           "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

__device__ inline void tcgen05_commit(int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_addr)
                 : "memory");
}

__device__ inline void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ inline void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ inline void tcgen05_ld_32x32b_64(int lane_base, int col_base, float out_vals[kStageTokens]) {
    const int addr = (lane_base << 16) | col_base;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
        : "=f"(out_vals[0]), "=f"(out_vals[1]), "=f"(out_vals[2]), "=f"(out_vals[3]),
          "=f"(out_vals[4]), "=f"(out_vals[5]), "=f"(out_vals[6]), "=f"(out_vals[7]),
          "=f"(out_vals[8]), "=f"(out_vals[9]), "=f"(out_vals[10]), "=f"(out_vals[11]),
          "=f"(out_vals[12]), "=f"(out_vals[13]), "=f"(out_vals[14]), "=f"(out_vals[15]),
          "=f"(out_vals[16]), "=f"(out_vals[17]), "=f"(out_vals[18]), "=f"(out_vals[19]),
          "=f"(out_vals[20]), "=f"(out_vals[21]), "=f"(out_vals[22]), "=f"(out_vals[23]),
          "=f"(out_vals[24]), "=f"(out_vals[25]), "=f"(out_vals[26]), "=f"(out_vals[27]),
          "=f"(out_vals[28]), "=f"(out_vals[29]), "=f"(out_vals[30]), "=f"(out_vals[31]),
          "=f"(out_vals[32]), "=f"(out_vals[33]), "=f"(out_vals[34]), "=f"(out_vals[35]),
          "=f"(out_vals[36]), "=f"(out_vals[37]), "=f"(out_vals[38]), "=f"(out_vals[39]),
          "=f"(out_vals[40]), "=f"(out_vals[41]), "=f"(out_vals[42]), "=f"(out_vals[43]),
          "=f"(out_vals[44]), "=f"(out_vals[45]), "=f"(out_vals[46]), "=f"(out_vals[47]),
          "=f"(out_vals[48]), "=f"(out_vals[49]), "=f"(out_vals[50]), "=f"(out_vals[51]),
          "=f"(out_vals[52]), "=f"(out_vals[53]), "=f"(out_vals[54]), "=f"(out_vals[55]),
          "=f"(out_vals[56]), "=f"(out_vals[57]), "=f"(out_vals[58]), "=f"(out_vals[59]),
          "=f"(out_vals[60]), "=f"(out_vals[61]), "=f"(out_vals[62]), "=f"(out_vals[63])
        : "r"(addr));
}

__device__ inline void tcgen05_ld_32x32b_8(int lane_base, int col_base, float out_vals[8]) {
    const int addr = (lane_base << 16) | col_base;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(out_vals[0]), "=f"(out_vals[1]), "=f"(out_vals[2]), "=f"(out_vals[3]),
          "=f"(out_vals[4]), "=f"(out_vals[5]), "=f"(out_vals[6]), "=f"(out_vals[7])
        : "r"(addr));
}

__device__ inline void topk_heap_sift_down(
    float* scores,
    int* ids,
    int size,
    int root
) {
    int p = root;
    while (true) {
        const int l = p * 2 + 1;
        if (l >= size) break;
        const int r = l + 1;
        int c = l;
        if (r < size && scores[r] < scores[l]) {
            c = r;
        }
        if (scores[c] >= scores[p]) {
            break;
        }
        const float ts = scores[p];
        scores[p] = scores[c];
        scores[c] = ts;
        const int ti = ids[p];
        ids[p] = ids[c];
        ids[c] = ti;
        p = c;
    }
}

__device__ inline void topk_heap_build(
    float* scores,
    int* ids,
    int size
) {
    for (int i = (size >> 1) - 1; i >= 0; --i) {
        topk_heap_sift_down(scores, ids, size, i);
    }
}

__device__ inline void topk_heap_push_candidate(
    float s,
    int id,
    float* heap_scores,
    int* heap_ids,
    int& heap_size,
    int topk,
    bool& heap_ready
) {
    if (heap_size < topk) {
        heap_scores[heap_size] = s;
        heap_ids[heap_size] = id;
        ++heap_size;
        if (heap_size == topk) {
            topk_heap_build(heap_scores, heap_ids, heap_size);
            heap_ready = true;
        }
        return;
    }

    if (!heap_ready) {
        topk_heap_build(heap_scores, heap_ids, heap_size);
        heap_ready = true;
    }
    if (s > heap_scores[0]) {
        heap_scores[0] = s;
        heap_ids[0] = id;
        topk_heap_sift_down(heap_scores, heap_ids, topk, 0);
    }
}

__device__ inline void topk_heap_emit_desc(
    float* heap_scores,
    int* heap_ids,
    int heap_size,
    int topk,
    int* out_b
) {
    if (heap_size <= 0) {
        return;
    }
    topk_heap_build(heap_scores, heap_ids, heap_size);
    int size = heap_size;
    for (int out = heap_size - 1; out >= 0; --out) {
        out_b[out] = heap_ids[0];
        --size;
        if (size <= 0) {
            break;
        }
        heap_scores[0] = heap_scores[size];
        heap_ids[0] = heap_ids[size];
        topk_heap_sift_down(heap_scores, heap_ids, size, 0);
    }
}

__device__ inline void topk_team_barrier_8warps() {
    asm volatile("bar.sync 2, 256;" ::: "memory");
}

__device__ inline void topk_compact_stage_two_warps(
    int topk_warp,
    int lane,
    int valid_tokens,
    const float* cand_scores,
    const int* cand_ids,
    float cutoff,
    float* compact_scores,
    int* compact_ids,
    int* compact_counts
) {
    if (topk_warp >= 2) {
        return;
    }

    const int token = topk_warp * 32 + lane;
    bool keep = false;
    float s = 0.0f;
    int id = -1;
    if (token < valid_tokens) {
        s = cand_scores[token];
        id = cand_ids[token];
        keep = (s > cutoff);
    }

    const unsigned int mask = __ballot_sync(0xFFFFFFFF, keep);
    const int count = __popc(mask);
    if (lane == 0) {
        compact_counts[topk_warp] = count;
    }

    if (keep) {
        const unsigned int lane_mask = (lane == 0) ? 0U : ((1U << lane) - 1U);
        const int rank = __popc(mask & lane_mask);
        const int out_idx = topk_warp * 32 + rank;
        compact_scores[out_idx] = s;
        compact_ids[out_idx] = id;
    }
}

__device__ inline void topk_process_compact_batch(
    int topk_warp,
    int lane,
    int compact_count0,
    int compact_count1,
    const float* compact_scores,
    const int* compact_ids,
    float* local_heap_scores,
    int* local_heap_ids,
    int* local_heap_sizes,
    int* local_heap_ready,
    int topk
) {
    if (topk_warp < kTopkUpdateWarpBase || topk_warp >= kTopkUpdateWarpBase + kNumTopkUpdateWarps) {
        return;
    }
    if (lane != 0) {
        return;
    }

    const int update_warp = topk_warp - kTopkUpdateWarpBase;
    float* heap_scores = local_heap_scores + update_warp * topk;
    int* heap_ids = local_heap_ids + update_warp * topk;
    int heap_size = local_heap_sizes[update_warp];
    bool heap_ready = (local_heap_ready[update_warp] != 0);

    for (int i = update_warp; i < compact_count0; i += kNumTopkUpdateWarps) {
        topk_heap_push_candidate(
            compact_scores[i],
            compact_ids[i],
            heap_scores,
            heap_ids,
            heap_size,
            topk,
            heap_ready);
    }
    for (int i = update_warp; i < compact_count1; i += kNumTopkUpdateWarps) {
        const int idx = 32 + i;
        topk_heap_push_candidate(
            compact_scores[idx],
            compact_ids[idx],
            heap_scores,
            heap_ids,
            heap_size,
            topk,
            heap_ready);
    }

    local_heap_sizes[update_warp] = heap_size;
    local_heap_ready[update_warp] = heap_ready ? 1 : 0;
}

__device__ inline void topk_update_global_cutoff_from_locals(
    int topk_warp,
    int lane,
    const float* local_heap_scores,
    const int* local_heap_sizes,
    const int* local_heap_ready,
    float* cutoff_ptr,
    int topk
) {
    if (topk_warp != 0 || lane != 0) {
        return;
    }

    bool all_ready = true;
    for (int w = 0; w < kNumTopkUpdateWarps; ++w) {
        if (!(local_heap_ready[w] != 0 && local_heap_sizes[w] >= topk)) {
            all_ready = false;
            break;
        }
    }

    if (!all_ready) {
        *cutoff_ptr = -3.402823466e+38F;
        return;
    }

    float cutoff = local_heap_scores[0];
    for (int w = 1; w < kNumTopkUpdateWarps; ++w) {
        const float root = local_heap_scores[w * topk];
        cutoff = fminf(cutoff, root);
    }
    *cutoff_ptr = cutoff;
}

__device__ inline void topk_merge_local_heaps_to_global(
    int topk_warp,
    int lane,
    const float* local_heap_scores,
    const int* local_heap_ids,
    const int* local_heap_sizes,
    float* global_heap_scores,
    int* global_heap_ids,
    int* out_b,
    int topk
) {
    if (topk_warp != 2 || lane != 0) {
        return;
    }

    int global_size = 0;
    bool global_ready = false;
    for (int w = 0; w < kNumTopkUpdateWarps; ++w) {
        const float* src_scores = local_heap_scores + w * topk;
        const int* src_ids = local_heap_ids + w * topk;
        const int n = local_heap_sizes[w];
        for (int i = 0; i < n; ++i) {
            topk_heap_push_candidate(
                src_scores[i],
                src_ids[i],
                global_heap_scores,
                global_heap_ids,
                global_size,
                topk,
                global_ready);
        }
    }

    topk_heap_emit_desc(global_heap_scores, global_heap_ids, global_size, topk, out_b);
}

__device__ inline void run_epilogue_warps(
    int warp_id,
    int lane,
    int b,
    int num_tiles,
    SmemLayout& s,
    const SmemAddrs& addr,
    const int* mma_debug_heads,
    const int* mma_debug_local_tok,
    float* mma_debug_vals,
    float* ep_debug_vals
) {
    // EP uses the first 8 warps, split as two 4-warp teams.
    // Each 4-warp team (slots 0..3) accumulates one 16-token chunk at a time.
    if (!_is_ep_warp(warp_id)) {
        return;
    }

    constexpr int kTokensPerChunk = 16;
    constexpr int kValsPerLd = 8;
    const int ep_team = warp_id >> 2;  // 0 or 1
    const int ep_slot = warp_id & 3;   // 0..3
    const int lane_base = ep_slot * 32;
    const int lane16 = lane & 15;
    const bool lane_active = (lane < 16);
    const int head = ep_slot * 16 + lane16;

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        const int stage = tile_id % kNumStages;
        const int tmem_slot = tile_id % kNumTmemSlots;

        if (warp_id == 0 && elect_sync()) {
            mbarrier_wait_parity(
                addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                s.mma_phase[stage]);
            s.mma_phase[stage] ^= 1;
        }

        asm volatile("bar.sync 1, 256;" ::: "memory");
        tcgen05_fence_after_thread_sync();

        const int valid_tokens = s.stage_valid_tokens[stage];
        if (valid_tokens > 0) {
            const int page_idx = s.stage_page_idx[stage];
            float* partial_scores = s.stage_tile_pair_partial + stage * kStageTokens;
            float* cand_scores = s.stage_tile_scores + stage * kStageTokens;
            int* cand_ids = s.stage_tile_ids + stage * kStageTokens;
            const int dbg_base = b * kDebugSamples;
            const int tile_tok_base = tile_id * kStageTokens;
            const float* stage_scale = s.k_stage_scale + stage * kStageTokens;
            const float w = s.w_stage[head];

            // Two passes: tokens {0..31} and {32..63}. Team 0 does low 16 of pass, team 1 does high 16.
            #pragma unroll
            for (int pass = 0; pass < 2; ++pass) {
                const int token_base = pass * 32 + ep_team * kTokensPerChunk;
                int chunk_valid = valid_tokens - token_base;
                if (chunk_valid < 0) chunk_valid = 0;
                if (chunk_valid > kTokensPerChunk) chunk_valid = kTokensPerChunk;

                float warp_partial[kTokensPerChunk];
                #pragma unroll
                for (int token = 0; token < kTokensPerChunk; ++token) {
                    warp_partial[token] = 0.0f;
                }

                if (chunk_valid > 0) {
                    const int tmem_col_base = tmem_slot * kStageTokens + token_base;
                    float lane_regs_0[kValsPerLd];
                    float lane_regs_1[kValsPerLd];
                    tcgen05_ld_32x32b_8(lane_base, tmem_col_base, lane_regs_0);
                    tcgen05_wait_ld();
                    tcgen05_ld_32x32b_8(lane_base, tmem_col_base + kValsPerLd, lane_regs_1);
                    tcgen05_wait_ld();

                    if (lane_active) {
                        #pragma unroll
                        for (int token = 0; token < kTokensPerChunk; ++token) {
                            const int tok = token_base + token;
                            float lane_val = 0.0f;
                            if (token < chunk_valid) {
                                const float raw = (token < kValsPerLd)
                                                    ? lane_regs_0[token]
                                                    : lane_regs_1[token - kValsPerLd];
                                lane_val = fmaxf(raw * stage_scale[tok], 0.0f) * w;

                                const int tile_tok = tile_tok_base + tok;
                                #pragma unroll
                                for (int sidx = 0; sidx < kPreDebugSamples; ++sidx) {
                                    if (mma_debug_heads[dbg_base + sidx] == head &&
                                        mma_debug_local_tok[dbg_base + sidx] == tile_tok) {
                                        mma_debug_vals[dbg_base + sidx] = raw;
                                    }
                                }
                            }
                            #pragma unroll
                            for (int off = 8; off > 0; off >>= 1) {
                                lane_val += __shfl_down_sync(0x0000FFFFU, lane_val, off);
                            }
                            warp_partial[token] = lane_val;
                        }
                    }
                }

                if (lane == 0 && ep_slot == 0) {
                    #pragma unroll
                    for (int token = 0; token < kTokensPerChunk; ++token) {
                        if (token < chunk_valid) {
                            partial_scores[token_base + token] = warp_partial[token];
                        }
                    }
                }
                asm volatile("bar.sync 1, 256;" ::: "memory");

                if (lane == 0 && ep_slot == 1) {
                    #pragma unroll
                    for (int token = 0; token < kTokensPerChunk; ++token) {
                        if (token < chunk_valid) {
                            partial_scores[token_base + token] += warp_partial[token];
                        }
                    }
                }
                asm volatile("bar.sync 1, 256;" ::: "memory");

                if (lane == 0 && ep_slot == 2) {
                    #pragma unroll
                    for (int token = 0; token < kTokensPerChunk; ++token) {
                        if (token < chunk_valid) {
                            partial_scores[token_base + token] += warp_partial[token];
                        }
                    }
                }
                asm volatile("bar.sync 1, 256;" ::: "memory");

                if (lane == 0 && ep_slot == 3) {
                    #pragma unroll
                    for (int token = 0; token < kTokensPerChunk; ++token) {
                        if (token < chunk_valid) {
                            const int tok = token_base + token;
                            const float final = partial_scores[tok] + warp_partial[token];
                            cand_scores[tok] = final;
                            cand_ids[tok] = page_idx * kPageSize + tok;

                            const int tile_tok = tile_tok_base + tok;
                            #pragma unroll
                            for (int sidx = 0; sidx < kDebugSamples; ++sidx) {
                                if (mma_debug_local_tok[dbg_base + sidx] == tile_tok) {
                                    ep_debug_vals[dbg_base + sidx] = final;
                                }
                            }
                        }
                    }
                }
                asm volatile("bar.sync 1, 256;" ::: "memory");
            }
        }

        if (warp_id == 0 && lane == 0) {
            mbarrier_arrive(addr.tmem_reuse_mbar + tmem_slot * static_cast<int>(sizeof(uint64_t)));
            mbarrier_arrive(addr.epi_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        }
    }
}

// ---------------------------------------------------------------------------------
// Kernel: 3-stage warp-specialized pipeline
// ---------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreadsPerBlock) void dsa_topk_indexer_kernel(
    const __grid_constant__ CUtensorMap q_fp8_tmap,
    const __grid_constant__ CUtensorMap k_fp8_tmap,
    const __grid_constant__ CUtensorMap k_scale_tmap,
    const uint8_t* q_index_bytes,   // [B,64,128], FP8 E4M3
    const float* weights,           // [B,64]
    const int* seq_lens,            // [B]
    const int* block_table,         // [B,max_num_pages]
    int* topk_indices,              // [B,topk]
    const int* mma_debug_heads,     // [B,12], pre uses first 4
    const int* mma_debug_local_tok, // [B,12]
    float* mma_debug_vals,          // [B,12], pre-ep qk_raw (first 4 used)
    float* ep_debug_vals,           // [B,12], post-ep final score
    int batch_size,
    int num_pages,
    int max_num_pages,
    int topk
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (b >= batch_size || warp_id >= kNumWarps) {
        return;
    }

    const int* block_table_b = block_table + static_cast<int64_t>(b) * max_num_pages;
    const float* weights_b = weights + static_cast<int64_t>(b) * kNumHeads;
    int* epi_tests_b = topk_indices + static_cast<int64_t>(b) * topk;

    int seq_len = seq_lens[b];
    if (seq_len < 0) seq_len = 0;
    const int max_seq_by_pages = max_num_pages * kPageSize;
    if (seq_len > max_seq_by_pages) seq_len = max_seq_by_pages;

    if (tid < kDebugSamples) {
        mma_debug_vals[static_cast<int64_t>(b) * kDebugSamples + tid] = 0.0f;
        ep_debug_vals[static_cast<int64_t>(b) * kDebugSamples + tid] = 0.0f;
    }

    for (int i = tid; i < topk; i += blockDim.x) {
        epi_tests_b[i] = -1;
    }
    __syncthreads();

    if (seq_len == 0 || topk <= 0) {
        return;
    }

    extern __shared__ __align__(1024) SmemLayout smem_storage[];
    SmemLayout& s = smem_storage[0];
    const SmemAddrs addr = init_smem_addrs(&s);

    // Init barriers and phases.
    if (warp_id == kProducerWarp && elect_sync()) {
        for (int i = 0; i < kNumStages; ++i) {
            mbarrier_init(addr.tma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.mma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.epi_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.topk_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            s.tma_phase[i] = 0;
            s.mma_phase[i] = 0;
            s.topk_phase[i] = 0;
        }
        for (int i = 0; i < kNumTmemSlots; ++i) {
            mbarrier_init(addr.tmem_reuse_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            s.tmem_reuse_phase_mma[i] = 0;
        }
        mbarrier_init(addr.q_mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    // tma load q so we get same swizzle as k_idx
    if (warp_id == kProducerWarp && elect_sync()) {
        constexpr int q_bytes = kNumHeads * kHeadDim;
        mbarrier_arrive_expect_tx(addr.q_mbar, q_bytes);
        tma_3d_gmem2smem(addr.q_stage, &q_fp8_tmap, 0, 0, b, addr.q_mbar, 0ULL);
        mbarrier_wait_parity(addr.q_mbar, 0);
    }
    __syncthreads();
    //asm volatile("bar.sync 1, 576;" ::: "memory");

    // #pragma unroll
    // for (int idx = tid; idx < kNumHeads * kHeadDim; idx += blockDim.x) {
    //     const int64_t q_off = static_cast<int64_t>(b) * kNumHeads * kHeadDim + idx;
    //     s.q_stage[idx] = q_index_bytes[q_off];
    // }


    #pragma unroll
    for (int idx = tid; idx < kNumHeads; idx += blockDim.x) {
        s.w_stage[idx] = weights_b[idx];
    }

    if (tid == 0) {
        s.tmem_addr_scratch = 0;
    }
    __syncthreads();


    constexpr uint32_t kIdesc = (0U << 7U)    // atype = E4M3
                              | (0U << 10U)   // btype = E4M3
                              | (1U << 4U)    // dtype = F32
                              | ((uint32_t)(kStageTokens >> 3U) << 17U)
                              | ((uint32_t)(kNumHeads >> 4U) << 24U);

    const int num_tiles = (seq_len + kStageTokens - 1) / kStageTokens;

    // ---------------- Producer warp ----------------
    if (warp_id == kProducerWarp && elect_sync()) {
        for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
            const int stage = tile_id % kNumStages;

            // Stage reuse requires prior top-k consumption on the same stage slot.
            if (tile_id >= kNumStages) {
                mbarrier_wait_parity(
                    addr.topk_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                    s.topk_phase[stage]);
                s.topk_phase[stage] ^= 1;
            }

            int page_idx, valid_tokens;
            prepare_stage_metadata_and_scale_tail(
                tile_id, stage, seq_len, max_num_pages, num_pages, block_table_b,
                s.stage_page_idx, s.stage_valid_tokens, s.k_stage_scale,
                page_idx, valid_tokens);

            if (valid_tokens > 0) {
                const int payload_dst = addr.k_stage_payload + stage * kStageTokens * kPayloadBytesPerToken;
                const int scale_dst = addr.k_stage_scale + stage * kStageTokens * static_cast<int>(sizeof(float));
                const int mbar = addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t));
                constexpr int payload_bytes = kStageTokens * kPayloadBytesPerToken;
                constexpr int scale_bytes = kStageTokens * kScaleBytesPerToken;

                mbarrier_arrive_expect_tx(mbar, payload_bytes + scale_bytes);

                tma_3d_gmem2smem(payload_dst, &k_fp8_tmap, 0, 0, page_idx, mbar, 0ULL);
                tma_2d_gmem2smem(scale_dst, &k_scale_tmap, 0, page_idx, mbar, 0ULL);
            } else {
                mbarrier_arrive(addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            }
        }
    }

    // ---------------- MMA warp ----------------
    if (warp_id == kMmaWarp) {
        tcgen05_alloc(addr.tmem_addr_scratch, kNumTmemSlots * kStageTokens);
    }
    if (warp_id == kMmaWarp && elect_sync()) {
        for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
            const int stage = tile_id % kNumStages;
            const int tmem_slot = tile_id % kNumTmemSlots;

            mbarrier_wait_parity(
                addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                s.tma_phase[stage]);
            s.tma_phase[stage] ^= 1;

            if (tile_id >= kNumTmemSlots) {
                mbarrier_wait_parity(
                    addr.tmem_reuse_mbar + tmem_slot * static_cast<int>(sizeof(uint64_t)),
                    s.tmem_reuse_phase_mma[tmem_slot]);
                s.tmem_reuse_phase_mma[tmem_slot] ^= 1;
            }

            const int valid_tokens = s.stage_valid_tokens[stage];
            if (valid_tokens > 0) {
                const int q_a = addr.q_stage;
                const int k_a = addr.k_stage_payload + stage * kStageTokens * kPayloadBytesPerToken;

                uint64_t q_desc = make_desc_kmajor_swizzle_128b(q_a);
                uint64_t k_desc = make_desc_kmajor_swizzle_128b(k_a);
                const uint32_t tmem_d = static_cast<uint32_t>(tmem_slot * kStageTokens);

                for (int ki = 0; ki < kMmaIters; ++ki) {
                    tcgen05_mma_f8f6f4(tmem_d, q_desc, k_desc, kIdesc, (ki > 0));
                    q_desc += (kMmaK >> 4);
                    k_desc += (kMmaK >> 4);
                }

                tcgen05_commit(addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            } else {
                mbarrier_arrive(addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            }
        }
    }

    // ---------------- Epilogue warps ----------------
    run_epilogue_warps(
        warp_id,
        lane,
        b,
        num_tiles,
        s,
        addr,
        mma_debug_heads,
        mma_debug_local_tok,
        mma_debug_vals,
        ep_debug_vals);

    // ---------------- Top-k warps ----------------
    if (_is_topk_warp(warp_id)) {
        const int topk_warp = _topk_warp_rank(warp_id);
        if (topk_warp == 0 && lane == 0) {
            s.topk_cutoff = -3.402823466e+38F;
            s.topk_stage_compact_counts[0] = 0;
            s.topk_stage_compact_counts[1] = 0;
            for (int w = 0; w < kNumTopkUpdateWarps; ++w) {
                s.topk_local_heap_sizes[w] = 0;
                s.topk_local_heap_ready[w] = 0;
            }
        }
        topk_team_barrier_8warps();

        for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
            const int stage = tile_id % kNumStages;
            const int phase = (tile_id / kNumStages) & 1;

            // Wait for epilogue to finish writing this stage's candidates.
            if (topk_warp == 0 && lane == 0) {
                mbarrier_wait_parity(
                    addr.epi_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                    phase);
            }
            topk_team_barrier_8warps();

            // Step 1: Compact — warps 0-1 filter 64 candidates against cutoff.
            const int valid_tokens = s.stage_valid_tokens[stage];
            const float cutoff = s.topk_cutoff;
            topk_compact_stage_two_warps(
                topk_warp, lane, valid_tokens,
                s.stage_tile_scores + stage * kStageTokens,
                s.stage_tile_ids + stage * kStageTokens,
                cutoff,
                s.topk_stage_compact_scores,
                s.topk_stage_compact_ids,
                s.topk_stage_compact_counts);

            topk_team_barrier_8warps();

            // Step 2: Update local heaps — warps 2-7 consume compacted survivors.
            const int cc0 = s.topk_stage_compact_counts[0];
            const int cc1 = s.topk_stage_compact_counts[1];
            topk_process_compact_batch(
                topk_warp, lane, cc0, cc1,
                s.topk_stage_compact_scores,
                s.topk_stage_compact_ids,
                s.topk_local_heap_scores,
                s.topk_local_heap_ids,
                s.topk_local_heap_sizes,
                s.topk_local_heap_ready,
                topk);

            // Step 3: Update global cutoff from local heap roots.
            topk_update_global_cutoff_from_locals(
                topk_warp, lane,
                s.topk_local_heap_scores,
                s.topk_local_heap_sizes,
                s.topk_local_heap_ready,
                &s.topk_cutoff,
                topk);

            topk_team_barrier_8warps();

            // Release stage for TMA reuse.
            if (topk_warp == 0 && lane == 0) {
                mbarrier_arrive(addr.topk_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            }
            topk_team_barrier_8warps();
        }

        // After all tiles: merge 6 local heaps into global heap, emit sorted result.
        topk_merge_local_heaps_to_global(
            topk_warp, lane,
            s.topk_local_heap_scores,
            s.topk_local_heap_ids,
            s.topk_local_heap_sizes,
            s.topk_heap_scores,
            s.topk_heap_ids,
            epi_tests_b,
            topk);
    }

    __syncthreads();
    if (warp_id == kMmaWarp) {
        tcgen05_dealloc(0, kNumTmemSlots * kStageTokens);
    }

}

// ---------------------------------------------------------------------------------
// Host-side tensor-map encode helpers
// ---------------------------------------------------------------------------------
static bool g_kernel_attrs_set = false;

static CUtensorMap make_q_fp8_tmap(const int8_t* q_ptr, int batch_size) {
    CUtensorMap tmap{};
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank] = {
        (uint64_t)kHeadDim,
        (uint64_t)kNumHeads,
        (uint64_t)batch_size,
    };
    uint64_t globalStrides[rank - 1] = {
        (uint64_t)kHeadDim,
        (uint64_t)(kNumHeads * kHeadDim),
    };
    uint32_t boxDim[rank] = {
        (uint32_t)kHeadDim,
        (uint32_t)kNumHeads,
        1U,
    };
    uint32_t elementStrides[rank] = {1U, 1U, 1U};
    auto st = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        (void*)q_ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(st == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for q");
    return tmap;
}

static CUtensorMap make_k_fp8_tmap(const int8_t* k_ptr, int num_pages) {
    CUtensorMap tmap{};
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank] = {
        (uint64_t)kPayloadBytesPerToken,
        (uint64_t)kPageSize,
        (uint64_t)num_pages,
    };
    uint64_t globalStrides[rank - 1] = {
        (uint64_t)kPayloadBytesPerToken,
        (uint64_t)kPageBytes,
    };
    uint32_t boxDim[rank] = {
        (uint32_t)kPayloadBytesPerToken,
        (uint32_t)kPageSize,
        1U,
    };
    uint32_t elementStrides[rank] = {1U, 1U, 1U};
    auto st = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        (void*)k_ptr,
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(st == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for payload");
    return tmap;
}

static CUtensorMap make_k_scale_tmap(const int8_t* k_ptr, int num_pages) {
    CUtensorMap tmap{};
    constexpr uint32_t srank = 2;
    const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(k_ptr) + kPackedFp8Bytes;
    uint64_t sglobalDim[srank] = {
        (uint64_t)(kPageSize * kScaleBytesPerToken),
        (uint64_t)num_pages,
    };
    uint64_t sglobalStrides[srank - 1] = {
        (uint64_t)kPageBytes,
    };
    uint32_t sboxDim[srank] = {
        (uint32_t)(kPageSize * kScaleBytesPerToken),
        1U,
    };
    uint32_t selementStrides[srank] = {1U, 1U};
    auto st = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        srank,
        (void*)scale_base,
        sglobalDim,
        sglobalStrides,
        sboxDim,
        selementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(st == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for scales");
    return tmap;
}

// ---------------------------------------------------------------------------------
// Launch entry
// ---------------------------------------------------------------------------------
void dsa_topk_indexer_launch(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices,
    torch::Tensor mma_debug_heads,
    torch::Tensor mma_debug_local_tok,
    torch::Tensor mma_debug_vals,
    torch::Tensor ep_debug_vals
) {
    const int batch_size = static_cast<int>(q_index_fp8.size(0));
    const int num_pages = static_cast<int>(k_index_cache_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int topk = static_cast<int>(topk_indices.size(1));

    if (batch_size == 0 || topk == 0) {
        return;
    }

    if (!g_kernel_attrs_set) {
        cudaFuncSetAttribute(
            dsa_topk_indexer_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDesiredDynamicSmemBytes
        );
        g_kernel_attrs_set = true;
    }

    const int8_t* q_ptr = reinterpret_cast<const int8_t*>(q_index_fp8.data_ptr());
    const int8_t* k_ptr = reinterpret_cast<const int8_t*>(k_index_cache_fp8.data_ptr());
    CUtensorMap q_fp8_tmap = make_q_fp8_tmap(q_ptr, batch_size);
    CUtensorMap k_fp8_tmap = make_k_fp8_tmap(k_ptr, num_pages);
    CUtensorMap k_scale_tmap = make_k_scale_tmap(k_ptr, num_pages);

    const int blocks = batch_size;
    dsa_topk_indexer_kernel<<<blocks, kThreadsPerBlock, kDesiredDynamicSmemBytes>>>(
        q_fp8_tmap,
        k_fp8_tmap,
        k_scale_tmap,
        reinterpret_cast<const uint8_t*>(q_index_fp8.data_ptr()),
        reinterpret_cast<const float*>(weights.data_ptr()),
        reinterpret_cast<const int*>(seq_lens.data_ptr()),
        reinterpret_cast<const int*>(block_table.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        reinterpret_cast<const int*>(mma_debug_heads.data_ptr()),
        reinterpret_cast<const int*>(mma_debug_local_tok.data_ptr()),
        reinterpret_cast<float*>(mma_debug_vals.data_ptr()),
        reinterpret_cast<float*>(ep_debug_vals.data_ptr()),
        batch_size,
        num_pages,
        max_num_pages,
        topk
    );
    cudaError_t launch_st = cudaGetLastError();
    TORCH_CHECK(launch_st == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(launch_st));
}
"""

cpp_decl_src = """
#include <torch/extension.h>
void dsa_topk_indexer_launch(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices,
    torch::Tensor mma_debug_heads,
    torch::Tensor mma_debug_local_tok,
    torch::Tensor mma_debug_vals,
    torch::Tensor ep_debug_vals);
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_indexer_ext",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            #verbose=True,
            no_implicit_headers=True,
            extra_cuda_cflags=[
                "-O1",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--split-compile=4",
                "--relocatable-device-code=false",
            ],
            extra_ldflags=["-lcuda"],
        )
    return _module


def compile_kernel():
    _get_module()


def _dsa_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
    mma_debug_heads: torch.Tensor,
    mma_debug_local_tok: torch.Tensor,
    mma_debug_vals: torch.Tensor,
    ep_debug_vals: torch.Tensor,
) -> torch.Tensor:
    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
        mma_debug_heads,
        mma_debug_local_tok,
        mma_debug_vals,
        ep_debug_vals,
    )
    return topk_indices


def _load_ref_indices_for_b0(device: torch.device, seq_lens: torch.Tensor, block_table: torch.Tensor):
    heads = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    local_tok = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    path = os.path.join(os.path.dirname(__file__), "outref.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        pairs = re.findall(r"(?:sample|pre_ep_sample)\d+:\s*head=(\d+)\s+local_tok=(\d+)", txt)
        if len(pairs) >= 4:
            heads = torch.tensor([int(pairs[i][0]) for i in range(4)], dtype=torch.int32, device=device)
            local_tok = torch.tensor([int(pairs[i][1]) for i in range(4)], dtype=torch.int32, device=device)
    except OSError:
        pass

    seq0 = int(seq_lens[0].item()) if seq_lens.numel() > 0 else 0
    if seq0 > 0:
        local_tok = torch.remainder(local_tok, seq0)
    else:
        local_tok = torch.zeros_like(local_tok)

    global_tok = torch.full((4,), -1, dtype=torch.int64, device=device)
    if block_table.size(0) > 0:
        max_pages = int(block_table.size(1))
        for i in range(4):
            t = int(local_tok[i].item())
            page_slot = t // 64
            if 0 <= page_slot < max_pages:
                global_page = int(block_table[0, page_slot].item())
                global_tok[i] = global_page * 64 + (t % 64)
    return heads, local_tok, global_tok


def _global_tok_from_local(
    block_table: torch.Tensor,
    batch_idx: int,
    local_tok: torch.Tensor,
) -> torch.Tensor:
    count = int(local_tok.numel())
    global_tok = torch.full((count,), -1, dtype=torch.int64, device=local_tok.device)
    if batch_idx < 0 or batch_idx >= int(block_table.size(0)):
        return global_tok

    max_pages = int(block_table.size(1))
    for i in range(count):
        t = int(local_tok[i].item())
        page_slot = t // 64
        if 0 <= page_slot < max_pages:
            global_page = int(block_table[batch_idx, page_slot].item())
            global_tok[i] = global_page * 64 + (t % 64)
    return global_tok


def _build_post_debug_local_tok(seq_len: int, device: torch.device) -> torch.Tensor:
    post_local_tok = torch.zeros((12,), dtype=torch.int32, device=device)
    if seq_len <= 0:
        return post_local_tok

    targets = (0, 1, 2, 3, 60, 61, 62, 63, 63, 63, 63, 63)
    last_tok = seq_len - 1
    for i, t in enumerate(targets):
        post_local_tok[i] = t if t <= last_tok else last_tok
    return post_local_tok


def custom_kernel(data: input_t) -> output_t:
    global _DEBUG_PRINTED_ONCE
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    batch = int(q_index_fp8.shape[0])
    k_pre_debug = 4
    k_post_debug = 12
    k_post_print = 8
    topk = 2048
    topk_indices = torch.full((batch, topk), -1, dtype=torch.int32, device=q_index_fp8.device)
    mma_debug_heads = torch.full((batch, k_post_debug), -1, dtype=torch.int32, device=q_index_fp8.device)
    mma_debug_local_tok = torch.zeros((batch, k_post_debug), dtype=torch.int32, device=q_index_fp8.device)
    mma_debug_vals = torch.full((batch, k_post_debug), float("nan"), dtype=torch.float32, device=q_index_fp8.device)
    ep_debug_vals = torch.full((batch, k_post_debug), float("nan"), dtype=torch.float32, device=q_index_fp8.device)

    if batch > 0:
        b0_heads, b0_pre_local_tok, _ = _load_ref_indices_for_b0(
            q_index_fp8.device, seq_lens, block_table
        )
        seq0 = int(seq_lens[0].item())
        b0_post_local_tok = _build_post_debug_local_tok(seq0, q_index_fp8.device)
        mma_debug_heads[0, :k_pre_debug] = b0_heads
        mma_debug_local_tok[0] = b0_post_local_tok
        b0_post_global_tok = _global_tok_from_local(block_table, 0, b0_post_local_tok)
    else:
        b0_heads = torch.zeros((4,), dtype=torch.int32, device=q_index_fp8.device)
        b0_pre_local_tok = torch.zeros((4,), dtype=torch.int32, device=q_index_fp8.device)
        b0_post_local_tok = torch.zeros((12,), dtype=torch.int32, device=q_index_fp8.device)
        b0_post_global_tok = torch.full((12,), -1, dtype=torch.int64, device=q_index_fp8.device)

    _dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
        mma_debug_heads,
        mma_debug_local_tok,
        mma_debug_vals,
        ep_debug_vals,
    )

    if batch > 0 and not _DEBUG_PRINTED_ONCE:
        torch.cuda.synchronize(q_index_fp8.device)
        post_vals = ep_debug_vals[0, :k_post_print].cpu().tolist()
        post_local_tok = b0_post_local_tok.cpu().tolist()
        post_global_tok = b0_post_global_tok.cpu().tolist()
        for i in range(k_post_print):
            print(
                f"post_ep_sample{i}: local_tok={int(post_local_tok[i])} "
                f"global_tok={int(post_global_tok[i])} post_ep_final_score={float(post_vals[i]):.7f}"
            )
        _DEBUG_PRINTED_ONCE = True
    return (topk_indices,)
