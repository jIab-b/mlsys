"""DSA index submission — multi-CTA, no in-kernel topk."""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


_module = None


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

constexpr int kWarpsPerWarpgroup = 4;
constexpr int kNumEpSlots = kWarpsPerWarpgroup;
constexpr int kNumEpTeams = 7;
constexpr int kNumEpilogueWarps = kNumEpTeams * kWarpsPerWarpgroup;  // 28
constexpr int kProducerWarp = kNumEpilogueWarps;                      // 28
constexpr int kMmaWarp = kProducerWarp + 1;                           // 29
constexpr int kNumWarps = kMmaWarp + 1;                               // 30
constexpr int kThreadsPerBlock = kNumWarps * 32;                      // 960

constexpr int kMmaK = 32;
constexpr int kMmaIters = kHeadDim / kMmaK;  // 4
constexpr int kDesiredDynamicSmemBytes = 120 * 1024;  // ~85 KB actual; >114 KB forces 1 CTA/SM
constexpr int kNumSMs = 148;  // B200

// ---------------------------------------------------------------------------------
// Shared memory layout
// ---------------------------------------------------------------------------------
struct __align__(1024) SmemLayout {
    uint8_t k_stage_payload[kNumStages * kStageTokens * kPayloadBytesPerToken];
    float k_stage_scale[kNumStages * kStageTokens];

    uint8_t q_stage[kNumHeads * kHeadDim];
    float w_stage[kNumHeads];

    int stage_page_idx[kNumStages];
    int stage_valid_tokens[kNumStages];

    int tma_phase[kNumStages];
    int tmem_reuse_phase_mma[kNumTmemSlots];

    uint64_t tma_mbar[kNumStages];
    uint64_t mma_mbar[kNumStages];
    uint64_t epi_mbar[kNumStages];
    uint64_t tmem_reuse_mbar[kNumTmemSlots];
    uint64_t q_mbar;

    float stage_tile_raw[kNumStages * kNumEpSlots * kStageTokens];

    int tmem_addr_scratch;
};

struct SmemAddrs {
    int k_stage_payload;
    int k_stage_scale;
    int q_stage;
    int tma_mbar;
    int mma_mbar;
    int epi_mbar;
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
    return warp_id < kNumEpilogueWarps;
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
    constexpr uint32_t kSuspendNs = 1000000U;
    constexpr uint64_t kTimeoutNs = 5000000000ULL;
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

    float* stage_scale = k_stage_scale + stage * kStageTokens;
    for (int tok = valid_tokens; tok < kStageTokens; ++tok) {
        stage_scale[tok] = 0.0f;
    }
}

__device__ inline void tma_3d_gmem2smem(
    int dst_smem_addr,
    const void* tmap_ptr,
    int x, int y, int z,
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
    int x, int y,
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

__device__ inline void tcgen05_ld_32x32b_16(int lane_base, int col_base, float out_vals[16]) {
    const int addr = (lane_base << 16) | col_base;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(out_vals[0]), "=f"(out_vals[1]), "=f"(out_vals[2]), "=f"(out_vals[3]),
          "=f"(out_vals[4]), "=f"(out_vals[5]), "=f"(out_vals[6]), "=f"(out_vals[7]),
          "=f"(out_vals[8]), "=f"(out_vals[9]), "=f"(out_vals[10]), "=f"(out_vals[11]),
          "=f"(out_vals[12]), "=f"(out_vals[13]), "=f"(out_vals[14]), "=f"(out_vals[15])
        : "r"(addr));
}

// ---------------------------------------------------------------------------------
// EP team barrier — 7 teams, bar IDs 3-9
// ---------------------------------------------------------------------------------
__device__ inline void ep_team_barrier_4warps(int ep_team) {
    if (ep_team == 0) {
        asm volatile("bar.sync 3, 128;" ::: "memory");
    } else if (ep_team == 1) {
        asm volatile("bar.sync 4, 128;" ::: "memory");
    } else if (ep_team == 2) {
        asm volatile("bar.sync 5, 128;" ::: "memory");
    } else if (ep_team == 3) {
        asm volatile("bar.sync 6, 128;" ::: "memory");
    } else if (ep_team == 4) {
        asm volatile("bar.sync 7, 128;" ::: "memory");
    } else if (ep_team == 5) {
        asm volatile("bar.sync 8, 128;" ::: "memory");
    } else {
        asm volatile("bar.sync 9, 128;" ::: "memory");
    }
}

// ---------------------------------------------------------------------------------
// Epilogue warps — multi-CTA: processes [tile_start, tile_start+num_my_tiles)
// ---------------------------------------------------------------------------------
__device__ inline void run_epilogue_warps(
    int warp_id,
    int lane,
    int tile_start,
    int num_my_tiles,
    SmemLayout& s,
    const SmemAddrs& addr,
    float* out_scores_b,
    int* out_ids_b,
    int out_stride
) {
    if (!_is_ep_warp(warp_id)) {
        return;
    }

    const int ep_team = warp_id >> 2;   // 0..6
    const int ep_slot = warp_id & 3;    // 0..3
    const int lane16 = lane & 15;
    const bool active_lane = (lane < 16);
    const int head = ep_slot * 16 + lane16;
    const int lane_base = ep_slot * 32;
    const int group_lane = ep_slot * 32 + lane;

    for (int local = ep_team; local < num_my_tiles; local += kNumEpTeams) {
        const int tile_id = tile_start + local;      // absolute — for global addressing
        const int stage = local % kNumStages;         // local — CTA resource
        const int tmem_slot = local % kNumTmemSlots;  // local — CTA resource
        const int phase = (local / kNumStages) & 1;

        // Wait for MMA to finish this stage.
        if (ep_slot == 0 && elect_sync()) {
            mbarrier_wait_parity(
                addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                phase);
        }
        ep_team_barrier_4warps(ep_team);
        tcgen05_fence_after_thread_sync();

        const int valid_tokens = s.stage_valid_tokens[stage];
        const int tile_seq_start = tile_id * kStageTokens;
        const int page_idx = s.stage_page_idx[stage];

        // Write global token ids.
        if (valid_tokens > 0 && group_lane < valid_tokens) {
            const int out_idx = tile_seq_start + group_lane;
            if (out_idx < out_stride) {
                out_ids_b[out_idx] = page_idx * kPageSize + group_lane;
            }
        }

        // TMEM read + ReLU + weight + shuffle-reduce across 16 heads.
        if (valid_tokens > 0) {
            const int tmem_col_base = tmem_slot * kStageTokens;
            const float* stage_scale = s.k_stage_scale + stage * kStageTokens;
            const float w = s.w_stage[head];
            constexpr int kChunk = 16;
            constexpr int kNumChunks = kStageTokens / kChunk;  // 4

            #pragma unroll
            for (int ci = 0; ci < kNumChunks; ++ci) {
                const int tok_off = ci * kChunk;
                float vals[kChunk];
                tcgen05_ld_32x32b_16(lane_base, tmem_col_base + tok_off, vals);
                tcgen05_wait_ld();

                if (active_lane) {
                    #pragma unroll
                    for (int t = 0; t < kChunk; ++t) {
                        const int tok = tok_off + t;
                        float v = 0.0f;
                        if (tok < valid_tokens) {
                            v = fmaxf(vals[t] * stage_scale[tok], 0.0f) * w;
                        }
                        vals[t] = v;
                    }

                    #pragma unroll
                    for (int off = 8; off > 0; off >>= 1) {
                        #pragma unroll
                        for (int t = 0; t < kChunk; ++t) {
                            vals[t] += __shfl_down_sync(0x0000FFFF, vals[t], off, 16);
                        }
                    }

                    if (lane16 == 0) {
                        #pragma unroll
                        for (int t = 0; t < kChunk; ++t) {
                            const int tok = tok_off + t;
                            if (tok < valid_tokens) {
                                const int part_idx =
                                    ((stage * kNumEpSlots + ep_slot) * kStageTokens) + tok;
                                s.stage_tile_raw[part_idx] = vals[t];
                            }
                        }
                    }
                }
            }
        }

        // Cross-slot reduction: sum 4 partials, write score to global.
        ep_team_barrier_4warps(ep_team);

        if (valid_tokens > 0 && group_lane < valid_tokens) {
            const int tok = group_lane;
            const int out_idx = tile_seq_start + tok;
            if (out_idx < out_stride) {
                const int part_base = stage * kNumEpSlots * kStageTokens + tok;
                float sum = 0.0f;
                #pragma unroll
                for (int sidx = 0; sidx < kNumEpSlots; ++sidx) {
                    sum += s.stage_tile_raw[part_base + sidx * kStageTokens];
                }
                out_scores_b[out_idx] = sum;
            }
        }

        ep_team_barrier_4warps(ep_team);

        // Signal stage + TMEM reuse.
        if (ep_slot == 0 && elect_sync()) {
            mbarrier_arrive(addr.tmem_reuse_mbar + tmem_slot * static_cast<int>(sizeof(uint64_t)));
            mbarrier_arrive(addr.epi_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        }
    }
}

// ---------------------------------------------------------------------------------
// Kernel — multi-CTA: grid(ctas_per_batch, batch_size)
// ---------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreadsPerBlock) void dsa_topk_indexer_kernel(
    const __grid_constant__ CUtensorMap q_fp8_tmap,
    const __grid_constant__ CUtensorMap k_fp8_tmap,
    const __grid_constant__ CUtensorMap k_scale_tmap,
    const float* weights,
    const int* seq_lens,
    const int* block_table,
    float* out_scores,
    int* out_ids,
    int batch_size,
    int num_pages,
    int max_num_pages,
    int out_stride,
    int tiles_per_cta
) {
    const int b = blockIdx.y;
    const int chunk_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (b >= batch_size || warp_id >= kNumWarps) {
        return;
    }

    // Batch pointers.
    const int* block_table_b = block_table + static_cast<int64_t>(b) * max_num_pages;
    const float* weights_b = weights + static_cast<int64_t>(b) * kNumHeads;
    float* out_scores_b = out_scores + static_cast<int64_t>(b) * out_stride;
    int* out_ids_b = out_ids + static_cast<int64_t>(b) * out_stride;

    extern __shared__ __align__(1024) SmemLayout smem_storage[];
    SmemLayout& s = smem_storage[0];
    const SmemAddrs addr = init_smem_addrs(&s);

    // Compute this CTA's tile range.
    int seq_len = seq_lens[b];
    if (seq_len < 0) seq_len = 0;
    const int max_seq_by_pages = max_num_pages * kPageSize;
    if (seq_len > max_seq_by_pages) seq_len = max_seq_by_pages;

    const int num_tiles_total = (seq_len + kStageTokens - 1) / kStageTokens;
    const int tile_start = chunk_id * tiles_per_cta;
    int tile_end = tile_start + tiles_per_cta;
    if (tile_end > num_tiles_total) tile_end = num_tiles_total;
    const int num_my_tiles = tile_end - tile_start;

    if (seq_len == 0 || num_my_tiles <= 0) {
        return;
    }

    // Init barriers and phases.
    if (warp_id == kProducerWarp && elect_sync()) {
        for (int i = 0; i < kNumStages; ++i) {
            mbarrier_init(addr.tma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.mma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.epi_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            s.tma_phase[i] = 0;
        }
        for (int i = 0; i < kNumTmemSlots; ++i) {
            mbarrier_init(addr.tmem_reuse_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            s.tmem_reuse_phase_mma[i] = 0;
        }
        mbarrier_init(addr.q_mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    // TMA load Q.
    if (warp_id == kProducerWarp && elect_sync()) {
        constexpr int q_bytes = kNumHeads * kHeadDim;
        mbarrier_arrive_expect_tx(addr.q_mbar, q_bytes);
        tma_3d_gmem2smem(addr.q_stage, &q_fp8_tmap, 0, 0, b, addr.q_mbar, 0ULL);
        mbarrier_wait_parity(addr.q_mbar, 0);
    }

    // Load weights into smem.
    if (tid < kNumHeads) {
        s.w_stage[tid] = weights_b[tid];
    }
    if (tid == 0) {
        s.tmem_addr_scratch = 0;
    }
    __syncthreads();

    constexpr uint32_t kIdesc = (0U << 7U)
                              | (0U << 10U)
                              | (1U << 4U)
                              | ((uint32_t)(kStageTokens >> 3U) << 17U)
                              | ((uint32_t)(kNumHeads >> 4U) << 24U);

    // ---------------- Producer warp ----------------
    if (warp_id == kProducerWarp && elect_sync()) {
        for (int local = 0; local < num_my_tiles; ++local) {
            const int tile_id = tile_start + local;
            const int stage = local % kNumStages;

            if (local >= kNumStages) {
                const int phase = ((local - kNumStages) / kNumStages) & 1;
                mbarrier_wait_parity(
                    addr.epi_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                    phase);
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
        for (int local = 0; local < num_my_tiles; ++local) {
            const int stage = local % kNumStages;
            const int tmem_slot = local % kNumTmemSlots;

            mbarrier_wait_parity(
                addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                s.tma_phase[stage]);
            s.tma_phase[stage] ^= 1;

            if (local >= kNumTmemSlots) {
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
        warp_id, lane, tile_start, num_my_tiles,
        s, addr, out_scores_b, out_ids_b, out_stride);

    __syncthreads();
    if (warp_id == kMmaWarp) {
        tcgen05_dealloc(0, kNumTmemSlots * kStageTokens);
    }
}

// ---------------------------------------------------------------------------------
// Host-side tensor-map helpers
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
// Launch entry — static scheduler
// ---------------------------------------------------------------------------------
void dsa_topk_indexer_launch(
    torch::Tensor q_index_fp8,
    torch::Tensor k_index_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor out_scores,
    torch::Tensor out_ids
) {
    const int batch_size = static_cast<int>(q_index_fp8.size(0));
    const int num_pages = static_cast<int>(k_index_cache_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int out_stride = static_cast<int>(out_scores.size(1));

    if (batch_size == 0 || out_stride == 0 || max_num_pages == 0) {
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

    // Static scheduler: spread tiles across SMs.
    int ctas_per_batch;
    if (max_num_pages * batch_size <= kNumSMs) {
        ctas_per_batch = max_num_pages;
    } else {
        ctas_per_batch = kNumSMs / batch_size;
        if (ctas_per_batch < 1) ctas_per_batch = 1;
        if (ctas_per_batch > max_num_pages) ctas_per_batch = max_num_pages;
    }
    const int tiles_per_cta = (max_num_pages + ctas_per_batch - 1) / ctas_per_batch;

    dim3 grid(ctas_per_batch, batch_size);
    dsa_topk_indexer_kernel<<<grid, kThreadsPerBlock, kDesiredDynamicSmemBytes>>>(
        q_fp8_tmap,
        k_fp8_tmap,
        k_scale_tmap,
        reinterpret_cast<const float*>(weights.data_ptr()),
        reinterpret_cast<const int*>(seq_lens.data_ptr()),
        reinterpret_cast<const int*>(block_table.data_ptr()),
        reinterpret_cast<float*>(out_scores.data_ptr()),
        reinterpret_cast<int*>(out_ids.data_ptr()),
        batch_size,
        num_pages,
        max_num_pages,
        out_stride,
        tiles_per_cta
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
    torch::Tensor out_scores,
    torch::Tensor out_ids);
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_indexer_ext_no_topk",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
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
    out_scores: torch.Tensor,
    out_ids: torch.Tensor,
):
    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        out_scores,
        out_ids,
    )
    return out_scores, out_ids


def custom_kernel(data: input_t) -> output_t:
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    batch = int(q_index_fp8.shape[0])
    topk = 2048
    out_topk_indices = torch.full((batch, topk), -1, dtype=torch.int32, device=q_index_fp8.device)
    if batch == 0:
        return (out_topk_indices,)

    max_seq = min(int(seq_lens.max().item()), int(block_table.shape[1]) * 64)
    if max_seq <= 0:
        return (out_topk_indices,)

    out_scores = torch.full((batch, max_seq), float('-inf'), dtype=torch.float32, device=q_index_fp8.device)
    out_ids = torch.full((batch, max_seq), -1, dtype=torch.int32, device=q_index_fp8.device)

    _dsa_topk_indexer(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        out_scores,
        out_ids,
    )

    # Skip topk — just return -1s to measure kernel-only time
    return (out_topk_indices,)
