"""DSA sparse-attention v2 — dual-MMA pipelined architecture.

Architecture:
- 1 producer warp: TMA loads Kc[64,512] + Kp[64,64] into double-buffered SMEM
- 2 MMA warps:
    score MMA warp: Kc[64,512] @ Q_nope[16,512]^T + Kp[64,64] @ Q_pe[16,64]^T → [64,16] in TMEM
    value MMA warp: 4x Kc_slice[128,64] @ weights[16,64]^T → [128,16] in TMEM
- 2 score-epilogue warpgroups (2x4=8 warps): stage-ring schedule, read [64,16] scores, online softmax, write bf16 weights
- 4 value-epilogue warpgroups (4x4=16 warps): one warpgroup per 128-d value tile, online rescale and output accumulation
- Total: 1 + 2 + 8 + 16 = 27 warps = 864 threads

Pipeline per stage (64 KV tokens):
  Producer TMA → score MMA → score epilogue (softmax → weights to SMEM) → value MMA → value epilogue
  Double-buffered: while score epilogue processes stage N, producer loads stage N+1.

Grid: num_tokens blocks (1 CTA per token for now; multi-CTA split-K is a future optimization).
"""

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


_module = None

NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 64
TOPK = 2048


cuda_src = """
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>

// ─── Constants ───────────────────────────────────────────────────────────────
constexpr int kNumHeads = 16;
constexpr int kHeadDimCkv = 512;
constexpr int kHeadDimKpe = 64;
constexpr int kPageSize = 64;
constexpr int kTopK = 2048;

constexpr int kStageTokens = 64;
constexpr int kNumStages = 2;  // double-buffer

constexpr int kMmaK = 32;
constexpr int kMmaItersCkv = kHeadDimCkv / kMmaK;  // 16
constexpr int kMmaItersKpe = kHeadDimKpe / kMmaK;   // 2

// Value MMA: 4 tiles of [128, 16] with K=64 (2 K-iters of 32)
constexpr int kValMmaMTile = 128;
constexpr int kValMmaMTiles = kHeadDimCkv / kValMmaMTile;  // 4
constexpr int kValMmaKIters = kStageTokens / kMmaK;        // 2

constexpr int kCkvRowBytes = kHeadDimCkv * static_cast<int>(sizeof(__nv_bfloat16));  // 1024
constexpr int kKpeRowBytes = kHeadDimKpe * static_cast<int>(sizeof(__nv_bfloat16));  // 128
constexpr int kCkvChunkBytes = 128;
constexpr int kCkvChunksPerRow = kCkvRowBytes / kCkvChunkBytes;  // 8

// ─── Warp layout ─────────────────────────────────────────────────────────────
// 2 score-epilogue warpgroups (8 warps) + 4 value-epilogue warpgroups (16 warps)
// + 2 MMA warps + 1 producer warp = 27 warps
constexpr int kNumScoreEpiWarpgroups = 2;
constexpr int kNumValueEpiWarpgroups = 4;
constexpr int kWarpsPerWarpgroup = 4;
constexpr int kScoreEpiWarps = kNumScoreEpiWarpgroups * kWarpsPerWarpgroup;  // 8
constexpr int kValueEpiWarps = kNumValueEpiWarpgroups * kWarpsPerWarpgroup;  // 16
// Warp IDs: [0..7] = score epilogue, [8..23] = value epilogue, 24 = score MMA, 25 = value MMA, 26 = producer
constexpr int kScoreMmaWarp = kScoreEpiWarps + kValueEpiWarps;  // 24
constexpr int kValueMmaWarp = kScoreMmaWarp + 1;                // 25
constexpr int kProducerWarp = kValueMmaWarp + 1;                // 26
constexpr int kNumWarps = kProducerWarp + 1;                    // 27
constexpr int kThreadsPerBlock = kNumWarps * 32;                // 864

// TMEM allocation:
// Score output [M=64, N=16] is read back with tcgen05.ld in 16-token chunks,
// so reserve 64 columns per stage slot.
// Value output path uses one 128-column scratch tile consumed by value epilogue.
// Keep a 512-column reservation for now to avoid allocator edge cases while iterating.
constexpr int kScoreTmemCols = kStageTokens;                          // 64
constexpr int kScoreTmemSlots = kNumStages;                           // 2
constexpr int kScoreTmemTotal = kScoreTmemSlots * kScoreTmemCols;     // 128
constexpr int kValueTmemCols = kValMmaMTile;                          // 128
constexpr int kValueTmemBase = kScoreTmemTotal;                       // 128
constexpr int kTotalTmemCols = 512;

// ─── SMEM layout ─────────────────────────────────────────────────────────────
struct __align__(1024) SmemLayout {
    // Query (loaded once per token)
    __nv_bfloat16 q_nope[kNumHeads * kHeadDimCkv];      // 16*512*2 = 16KB
    __nv_bfloat16 q_pe[kNumHeads * kHeadDimKpe];         // 16*64*2 = 2KB

    // Double-buffered KV stages
    __nv_bfloat16 kc_stage[kNumStages * kStageTokens * kHeadDimCkv];  // 2*64*512*2 = 128KB
    __nv_bfloat16 kp_stage[kNumStages * kStageTokens * kHeadDimKpe];  // 2*64*64*2 = 16KB

    // Index tracking
    int stage_indices[kNumStages * kStageTokens];
    int stage_valid[kNumStages];

    // Softmax weights after score epilogue, written as bf16 for value MMA
    // Shape: [stage, tokens, heads] = [2, 64, 16] — token-major for value MMA B operand
    __nv_bfloat16 attn_weights[kNumStages * kStageTokens * kNumHeads];  // 2*64*16*2 = 4KB

    // Stage score scratch and per-stage output rescale (online softmax).
    float score_buf[kNumStages * kStageTokens * kNumHeads];
    float stage_alpha[kNumStages * kNumHeads];

    // Online softmax state (per head)
    float m_state[kNumHeads];
    float l_state[kNumHeads];
    float o_accum[kNumHeads * kHeadDimCkv];

    // Barriers
    uint64_t tma_mbar[kNumStages];
    uint64_t score_mma_mbar[kNumStages];
    uint64_t score_epi_mbar[kNumStages];   // score epilogue done → value MMA can proceed
    uint64_t score_tmem_reuse_mbar[kNumStages];  // score epilogue done reading score TMEM slot
    uint64_t value_mma_mbar[kNumStages * kValMmaMTiles];   // value MMA tile ready
    uint64_t value_tile_mbar[kNumStages * kValMmaMTiles];  // value epilogue tile consumed
    uint64_t value_epi_mbar[kNumStages];   // value epilogue done → producer can reuse SMEM

    int tmem_addr_scratch;
};

constexpr int kDesiredDynamicSmemBytes = static_cast<int>(sizeof(SmemLayout));

// ─── SMEM address helper ─────────────────────────────────────────────────────
struct SmemAddrs {
    int q_nope;
    int q_pe;
    int kc_stage;
    int kp_stage;
    int attn_weights;
    int tma_mbar;
    int score_mma_mbar;
    int score_epi_mbar;
    int score_tmem_reuse_mbar;
    int value_mma_mbar;
    int value_tile_mbar;
    int value_epi_mbar;
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
    a.q_nope = off(s->q_nope);
    a.q_pe = off(s->q_pe);
    a.kc_stage = off(s->kc_stage);
    a.kp_stage = off(s->kp_stage);
    a.attn_weights = off(s->attn_weights);
    a.tma_mbar = off(s->tma_mbar);
    a.score_mma_mbar = off(s->score_mma_mbar);
    a.score_epi_mbar = off(s->score_epi_mbar);
    a.score_tmem_reuse_mbar = off(s->score_tmem_reuse_mbar);
    a.value_mma_mbar = off(s->value_mma_mbar);
    a.value_tile_mbar = off(s->value_tile_mbar);
    a.value_epi_mbar = off(s->value_epi_mbar);
    a.tmem_addr_scratch = off(&s->tmem_addr_scratch);
    return a;
}

// ─── PTX intrinsics ──────────────────────────────────────────────────────────

__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{ .reg .pred %%px; elect.sync _|%%px, %1; @%%px mov.s32 %0, 1; }"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ inline constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_desc_kmajor_swizzle_128b(int smem_addr) {
    const int sbo = 8 * 128;
    const uint64_t base_offset = (static_cast<uint32_t>(smem_addr) >> 7U) & 0x7ULL;
    return desc_encode(static_cast<uint64_t>(smem_addr)) |
           (desc_encode(static_cast<uint64_t>(sbo)) << 32ULL) |
           (base_offset << 49ULL) |
           (1ULL << 46ULL) |
           (2ULL << 61ULL);
}

// K-major no-swizzle descriptor for bf16 uses:
// LBO = height * 16 bytes, SBO = 8 * 16 bytes (tcgen tutorial + PTX canonical layout).
__device__ inline uint64_t make_desc_kmajor_no_swizzle(int smem_addr, int height_rows) {
    const int lbo = height_rows * 16;
    const int sbo = 8 * 16;
    return desc_encode(static_cast<uint64_t>(smem_addr)) |
           (desc_encode(static_cast<uint64_t>(lbo)) << 16ULL) |
           (desc_encode(static_cast<uint64_t>(sbo)) << 32ULL) |
           (1ULL << 46ULL);  // no swizzle (bits 61:62 = 0)
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(mbar_addr), "r"(size_bytes)
        : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :
                 : "r"(mbar_addr)
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
        if (complete) return;
        uint64_t now_ns = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now_ns));
        if (now_ns - start_ns > kTimeoutNs) {
            asm volatile("trap;");
        }
    }
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
        :
        : "r"(dst_smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "l"(cache_policy)
        : "memory");
}

__device__ inline void tma_2d_prefetch_gather4(
    const void* tmap_ptr,
    int x,
    int y0,
    int y1,
    int y2,
    int y3
) {
    asm volatile(
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [%0, {%1, %2, %3, %4, %5}];"
        :
        : "l"(tmap_ptr), "r"(x), "r"(y0), "r"(y1), "r"(y2), "r"(y3)
        : "memory");
}

__device__ inline void tma_2d_gmem2smem_gather4(
    int dst_smem_addr,
    const void* tmap_ptr,
    int col,
    int row0,
    int row1,
    int row2,
    int row3,
    int mbar_addr,
    uint64_t cache_policy
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :
        : "r"(dst_smem_addr), "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(mbar_addr), "l"(cache_policy)
        : "memory");
}

__device__ inline void tcgen05_alloc(int smem_addr, int num_cols) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(smem_addr), "r"(num_cols));
}

__device__ inline void tcgen05_dealloc(int base_tmem, int num_cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(base_tmem), "r"(num_cols));
}

__device__ inline void tcgen05_mma_f16(
    uint32_t tmem_d,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t idesc,
    int enable_input_d
) {
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %4, 0;\\n\\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\\n\\t"
        "}"
        :
        : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(enable_input_d),
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
}

__device__ inline void tcgen05_commit(int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :
                 : "r"(mbar_addr)
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
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(out_vals[0]), "=f"(out_vals[1]), "=f"(out_vals[2]), "=f"(out_vals[3]),
          "=f"(out_vals[4]), "=f"(out_vals[5]), "=f"(out_vals[6]), "=f"(out_vals[7]),
          "=f"(out_vals[8]), "=f"(out_vals[9]), "=f"(out_vals[10]), "=f"(out_vals[11]),
          "=f"(out_vals[12]), "=f"(out_vals[13]), "=f"(out_vals[14]), "=f"(out_vals[15])
        : "r"(addr));
}

__device__ inline float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ inline __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ inline void init_query_and_state(
    SmemLayout& s,
    int tid,
    int token,
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe
) {
    const __nv_bfloat16* qn_src = q_nope + static_cast<int64_t>(token) * kNumHeads * kHeadDimCkv;
    const __nv_bfloat16* qp_src = q_pe + static_cast<int64_t>(token) * kNumHeads * kHeadDimKpe;
    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        s.q_nope[i] = qn_src[i];
    }
    for (int i = tid; i < kNumHeads * kHeadDimKpe; i += kThreadsPerBlock) {
        s.q_pe[i] = qp_src[i];
    }

    if (tid < kNumHeads) {
        s.m_state[tid] = -INFINITY;
        s.l_state[tid] = 0.0f;
    }
    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        s.o_accum[i] = 0.0f;
    }
    for (int i = tid; i < kNumStages * kNumHeads; i += kThreadsPerBlock) {
        s.stage_alpha[i] = 1.0f;
    }
}

__device__ inline void init_pipeline_barriers(
    SmemLayout& s,
    const SmemAddrs& addr,
    int warp_id
) {
    if (warp_id == kProducerWarp && elect_sync()) {
        for (int i = 0; i < kNumStages; ++i) {
            mbarrier_init(addr.tma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.score_mma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.score_epi_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.score_tmem_reuse_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.value_epi_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            for (int tile = 0; tile < kValMmaMTiles; ++tile) {
                const int st = i * kValMmaMTiles + tile;
                mbarrier_init(addr.value_mma_mbar + st * static_cast<int>(sizeof(uint64_t)), 1);
                mbarrier_init(addr.value_tile_mbar + st * static_cast<int>(sizeof(uint64_t)), 1);
            }
        }
        s.tmem_addr_scratch = 0;
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
}

__device__ inline void run_producer_warp(
    SmemLayout& s,
    const SmemAddrs& addr,
    int lane,
    int warp_id,
    int token,
    const int* sparse_indices,
    int total_kv_tokens,
    const CUtensorMap* kc_tmap,
    const CUtensorMap* kp_tmap,
    int stages_total
) {
    if (warp_id != kProducerWarp) return;

    for (int local = 0; local < stages_total; ++local) {
        const int stage = local % kNumStages;
        const int stage_start = local * kStageTokens;

        if (lane == 0 && local >= kNumStages) {
            const int phase = ((local - kNumStages) / kNumStages) & 1;
            mbarrier_wait_parity(
                addr.value_epi_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                phase);
        }

        for (int i = lane; i < kStageTokens; i += 32) {
            const int idx = sparse_indices[token * kTopK + stage_start + i];
            const bool valid = (idx >= 0 && idx < total_kv_tokens);
            s.stage_indices[stage * kStageTokens + i] = valid ? idx : -1;
            if (!valid) {
                __nv_bfloat16* kc_row = s.kc_stage + (stage * kStageTokens + i) * kHeadDimCkv;
                __nv_bfloat16* kp_row = s.kp_stage + (stage * kStageTokens + i) * kHeadDimKpe;
                for (int d = lane; d < kHeadDimCkv; d += 32) kc_row[d] = float_to_bf16(0.0f);
                for (int d = lane; d < kHeadDimKpe; d += 32) kp_row[d] = float_to_bf16(0.0f);
            }
        }
        __syncwarp();

        if (lane == 0) {
            int valid_count = 0;
            for (int i = 0; i < kStageTokens; ++i) {
                valid_count += (s.stage_indices[stage * kStageTokens + i] >= 0) ? 1 : 0;
            }
            s.stage_valid[stage] = valid_count;

            const int mbar = addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t));
            const int bytes = valid_count * (kCkvRowBytes + kKpeRowBytes);
            if (bytes > 0) {
                mbarrier_arrive_expect_tx(mbar, bytes);

                for (int g = 0; g < kStageTokens; g += 4) {
                    const int r0 = s.stage_indices[stage * kStageTokens + g + 0];
                    const int r1 = s.stage_indices[stage * kStageTokens + g + 1];
                    const int r2 = s.stage_indices[stage * kStageTokens + g + 2];
                    const int r3 = s.stage_indices[stage * kStageTokens + g + 3];
                    const int kc_dst_group = addr.kc_stage + ((stage * kStageTokens + g) * kCkvRowBytes);
                    if (r0 >= 0 && r1 >= 0 && r2 >= 0 && r3 >= 0) {
                        #pragma unroll
                        for (int c = 0; c < kCkvChunksPerRow; ++c) {
                            tma_2d_gmem2smem_gather4(
                                kc_dst_group + c * kCkvChunkBytes,
                                kc_tmap,
                                c * kCkvChunkBytes,
                                r0, r1, r2, r3,
                                mbar,
                                0ULL);
                        }
                    } else {
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) {
                            const int idx = s.stage_indices[stage * kStageTokens + g + t];
                            if (idx < 0) continue;
                            const int kc_dst_row = kc_dst_group + t * kCkvRowBytes;
                            #pragma unroll
                            for (int c = 0; c < kCkvChunksPerRow; ++c) {
                                tma_2d_gmem2smem(
                                    kc_dst_row + c * kCkvChunkBytes,
                                    kc_tmap,
                                    c * kCkvChunkBytes,
                                    idx,
                                    mbar,
                                    0ULL);
                            }
                        }
                    }
                }

                for (int i = 0; i < kStageTokens; ++i) {
                    const int idx = s.stage_indices[stage * kStageTokens + i];
                    if (idx < 0) continue;
                    const int kp_dst = addr.kp_stage + ((stage * kStageTokens + i) * kKpeRowBytes);
                    tma_2d_gmem2smem(kp_dst, kp_tmap, 0, idx, mbar, 0ULL);
                }
            } else {
                mbarrier_arrive(mbar);
            }
        }
    }
}

__device__ inline void run_mma_warps(
    SmemLayout& s,
    const SmemAddrs& addr,
    int lane,
    int warp_id,
    int stages_total
) {
    if (warp_id != kScoreMmaWarp && warp_id != kValueMmaWarp) return;

    constexpr uint32_t kScoreIdesc = (0U << 7U) |
                                     (0U << 10U) |
                                     (1U << 4U) |
                                     ((uint32_t)(kNumHeads >> 3U) << 17U) |
                                     ((uint32_t)(kStageTokens >> 4U) << 24U);
    constexpr uint32_t kValueIdesc = (0U << 7U) |
                                     (0U << 10U) |
                                     (1U << 4U) |
                                     ((uint32_t)(kNumHeads >> 3U) << 17U) |
                                     ((uint32_t)(kValMmaMTile >> 4U) << 24U);

    if (warp_id == kScoreMmaWarp) {tcgen05_alloc(addr.tmem_addr_scratch, kTotalTmemCols);}
    if (warp_id == kScoreMmaWarp && lane == 0) {
        const uint32_t tmem_base = static_cast<uint32_t>(s.tmem_addr_scratch);
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int tmem_slot = local % kNumStages;
            const int phase = (local / kNumStages) & 1;

            mbarrier_wait_parity(
                addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                phase);

            if (local >= kNumStages) {
                const int reuse_phase = ((local - kNumStages) / kNumStages) & 1;
                mbarrier_wait_parity(
                    addr.score_tmem_reuse_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                    reuse_phase);
            }

            if (s.stage_valid[stage] > 0) {
                const uint32_t tmem_d = tmem_base + static_cast<uint32_t>(tmem_slot * kScoreTmemCols);
                uint64_t kc_desc = make_desc_kmajor_swizzle_128b(
                    addr.kc_stage + stage * kStageTokens * kCkvRowBytes);
                uint64_t qn_desc = make_desc_kmajor_no_swizzle(addr.q_nope, kNumHeads);
                for (int ki = 0; ki < kMmaItersCkv; ++ki) {
                    tcgen05_mma_f16(tmem_d, kc_desc, qn_desc, kScoreIdesc, (ki > 0) ? 1 : 0);
                    kc_desc += (kMmaK >> 4);
                    qn_desc += (kMmaK >> 4);
                }

                uint64_t kp_desc = make_desc_kmajor_no_swizzle(
                    addr.kp_stage + stage * kStageTokens * kKpeRowBytes,
                    kStageTokens);
                uint64_t qp_desc = make_desc_kmajor_no_swizzle(addr.q_pe, kNumHeads);
                for (int ki = 0; ki < kMmaItersKpe; ++ki) {
                    tcgen05_mma_f16(tmem_d, kp_desc, qp_desc, kScoreIdesc, 1);
                    kp_desc += (kMmaK >> 4);
                    qp_desc += (kMmaK >> 4);
                }

                tcgen05_commit(addr.score_mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            } else {
                mbarrier_arrive(addr.score_mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            }
        }
    }

    if (warp_id == kValueMmaWarp && lane == 0) {
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int phase = (local / kNumStages) & 1;
            mbarrier_wait_parity(
                addr.score_epi_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                phase);
            const uint32_t tmem_base = static_cast<uint32_t>(s.tmem_addr_scratch);

            const bool valid_stage = (s.stage_valid[stage] > 0);
            const uint64_t w_desc_base = make_desc_kmajor_no_swizzle(
                addr.attn_weights + stage * kStageTokens * kNumHeads * static_cast<int>(sizeof(__nv_bfloat16)),
                kNumHeads);

            for (int tile = 0; tile < kValMmaMTiles; ++tile) {
                const int st = stage * kValMmaMTiles + tile;
                if (valid_stage) {
                    const uint32_t tmem_d = tmem_base + static_cast<uint32_t>(kValueTmemBase);
                    uint64_t kc_col_desc = make_desc_kmajor_swizzle_128b(
                        addr.kc_stage + stage * kStageTokens * kCkvRowBytes +
                        tile * kValMmaMTile * static_cast<int>(sizeof(__nv_bfloat16)));
                    uint64_t w_desc = w_desc_base;

                    for (int ki = 0; ki < kValMmaKIters; ++ki) {
                        tcgen05_mma_f16(tmem_d, kc_col_desc, w_desc, kValueIdesc, (ki > 0) ? 1 : 0);
                        kc_col_desc += (kMmaK >> 4);
                        w_desc += (kMmaK >> 4);
                    }
                    tcgen05_commit(addr.value_mma_mbar + st * static_cast<int>(sizeof(uint64_t)));
                } else {
                    mbarrier_arrive(addr.value_mma_mbar + st * static_cast<int>(sizeof(uint64_t)));
                }

                mbarrier_wait_parity(
                    addr.value_tile_mbar + st * static_cast<int>(sizeof(uint64_t)),
                    phase);
            }
        }
    }
}

__device__ inline void run_score_epilogue_warps(
    SmemLayout& s,
    const SmemAddrs& addr,
    int lane,
    int warp_id,
    int stages_total,
    float sm_scale_log2e
) {
    if (warp_id >= kScoreEpiWarps) return;
    const int ep_wg = warp_id / kWarpsPerWarpgroup;
    const int ep_warp_lane = warp_id % kWarpsPerWarpgroup;
    // Keep one active warp per score-ep warpgroup for now.
    if (ep_warp_lane != 0) return;

    for (int local = 0; local < stages_total; ++local) {
        const int stage = local % kNumStages;
        const int tmem_slot = local % kNumStages;
        const int phase = (local / kNumStages) & 1;
        if ((stage % kNumScoreEpiWarpgroups) != ep_wg) continue;

        if (lane == 0) {
            mbarrier_wait_parity(
                addr.score_mma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                phase);
        }
        __syncwarp();
        tcgen05_fence_after_thread_sync();

        if (s.stage_valid[stage] > 0) {
            float vals[16];
            const int lane_base = 0;
            const int tmem_col_base =
                static_cast<int>(static_cast<uint32_t>(s.tmem_addr_scratch)) + tmem_slot * kScoreTmemCols;
            #pragma unroll
            for (int tok_off = 0; tok_off < kStageTokens; tok_off += 16) {
                tcgen05_ld_32x32b_16(lane_base, tmem_col_base + tok_off, vals);
                tcgen05_wait_ld();
                if (lane < kNumHeads) {
                    const int head = lane;
                    #pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        s.score_buf[(stage * kStageTokens + tok_off + t) * kNumHeads + head] =
                            vals[t] * sm_scale_log2e;
                    }
                }
            }
        } else if (lane < kNumHeads) {
            const int head = lane;
            for (int tok = 0; tok < kStageTokens; ++tok) {
                s.score_buf[(stage * kStageTokens + tok) * kNumHeads + head] = -INFINITY;
            }
        }
        __syncwarp();

        if (lane < kNumHeads) {
            const int head = lane;
            float m_tile = -INFINITY;
            for (int tok = 0; tok < kStageTokens; ++tok) {
                const int sparse_idx = s.stage_indices[stage * kStageTokens + tok];
                if (sparse_idx < 0) continue;
                const float score = s.score_buf[(stage * kStageTokens + tok) * kNumHeads + head];
                m_tile = fmaxf(m_tile, score);
            }

            float m_prev = s.m_state[head];
            float l_prev = s.l_state[head];
            if (m_tile == -INFINITY) {
                s.stage_alpha[stage * kNumHeads + head] = 1.0f;
                for (int tok = 0; tok < kStageTokens; ++tok) {
                    s.attn_weights[(stage * kStageTokens + tok) * kNumHeads + head] = float_to_bf16(0.0f);
                }
            } else {
                const float m_new = fmaxf(m_prev, m_tile);
                const float alpha = (m_prev == -INFINITY) ? 0.0f : exp2f(m_prev - m_new);
                float l_tile = 0.0f;
                for (int tok = 0; tok < kStageTokens; ++tok) {
                    const int sparse_idx = s.stage_indices[stage * kStageTokens + tok];
                    float w = 0.0f;
                    if (sparse_idx >= 0) {
                        const float score = s.score_buf[(stage * kStageTokens + tok) * kNumHeads + head];
                        w = exp2f(score - m_new);
                        l_tile += w;
                    }
                    s.attn_weights[(stage * kStageTokens + tok) * kNumHeads + head] = float_to_bf16(w);
                }
                s.stage_alpha[stage * kNumHeads + head] = alpha;
                s.m_state[head] = m_new;
                s.l_state[head] = l_prev * alpha + l_tile;
            }
        }
        __syncwarp();

        if (lane == 0) {
            mbarrier_arrive(addr.score_epi_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            mbarrier_arrive(addr.score_tmem_reuse_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        }
    }
}

__device__ inline void run_value_epilogue_warps(
    SmemLayout& s,
    const SmemAddrs& addr,
    int lane,
    int warp_id,
    int stages_total
) {
    if (warp_id < kScoreEpiWarps || warp_id >= kScoreEpiWarps + kValueEpiWarps) return;
    const int value_wg = (warp_id - kScoreEpiWarps) / kWarpsPerWarpgroup;
    const int value_ep_warp_lane = (warp_id - kScoreEpiWarps) % kWarpsPerWarpgroup;
    // Keep one active warp per value-ep warpgroup for now.
    if (value_ep_warp_lane != 0) return;
    if (value_wg >= kValMmaMTiles) return;

    for (int local = 0; local < stages_total; ++local) {
        const int stage = local % kNumStages;
        const int tile = value_wg;
        const int st = stage * kValMmaMTiles + tile;
        const int phase = (local / kNumStages) & 1;
        if (lane == 0) {
            mbarrier_wait_parity(
                addr.value_mma_mbar + st * static_cast<int>(sizeof(uint64_t)),
                phase);
        }
        __syncwarp();
        tcgen05_fence_after_thread_sync();

        const bool active_head_lane = (lane < kNumHeads);
        const int head = lane;
        const float alpha = active_head_lane ? s.stage_alpha[stage * kNumHeads + head] : 0.0f;
        const int base_idx = active_head_lane ? (head * kHeadDimCkv + tile * kValMmaMTile) : 0;
        if (s.stage_valid[stage] > 0) {
            float vals[16];
            const int lane_base = 0;
            const int tmem_col_base =
                static_cast<int>(static_cast<uint32_t>(s.tmem_addr_scratch)) + kValueTmemBase;
            #pragma unroll
            for (int dim_off = 0; dim_off < kValMmaMTile; dim_off += 16) {
                tcgen05_ld_32x32b_16(lane_base, tmem_col_base + dim_off, vals);
                tcgen05_wait_ld();
                if (active_head_lane) {
                    #pragma unroll
                    for (int d = 0; d < 16; ++d) {
                        const int out_idx = base_idx + dim_off + d;
                        s.o_accum[out_idx] = s.o_accum[out_idx] * alpha + vals[d];
                    }
                }
            }
        }
        __syncwarp();

        if (lane == 0) {
            mbarrier_arrive(addr.value_tile_mbar + st * static_cast<int>(sizeof(uint64_t)));
        }

        if (lane == 0 && tile == (kValMmaMTiles - 1)) {
            mbarrier_arrive(addr.value_epi_mbar + stage * static_cast<int>(sizeof(uint64_t)));
        }
    }
}

__device__ inline void writeback_outputs(
    SmemLayout& s,
    int tid,
    int token,
    __nv_bfloat16* out,
    float* out_lse
) {
    __nv_bfloat16* out_token = out + static_cast<int64_t>(token) * kNumHeads * kHeadDimCkv;
    float* lse_token = out_lse + static_cast<int64_t>(token) * kNumHeads;

    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        const int head = i / kHeadDimCkv;
        const float l = s.l_state[head];
        const float v = (l > 0.0f) ? (s.o_accum[i] / l) : 0.0f;
        out_token[i] = float_to_bf16(v);
    }
    if (tid < kNumHeads) {
        const float l = s.l_state[tid];
        const float m = s.m_state[tid];
        if (l > 0.0f && m != -INFINITY) {
            lse_token[tid] = m + log2f(l);
        } else {
            lse_token[tid] = -INFINITY;
        }
    }
}

// ─── Kernel ──────────────────────────────────────────────────────────────────

__global__ __launch_bounds__(kThreadsPerBlock) void dsa_sparse_attn_kernel_v2(
    const __grid_constant__ CUtensorMap kc_tmap,
    const __grid_constant__ CUtensorMap kp_tmap,
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe,
    const __nv_bfloat16* ckv_rows,
    const int* sparse_indices,
    __nv_bfloat16* out,
    float* out_lse,
    int num_tokens,
    int total_kv_tokens,
    float sm_scale_log2e
) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;
    (void)ckv_rows;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int stages_total = (kTopK + kStageTokens - 1) / kStageTokens;

    extern __shared__ __align__(1024) SmemLayout smem_storage[];
    SmemLayout& s = smem_storage[0];
    const SmemAddrs addr = init_smem_addrs(&s);

    init_query_and_state(s, tid, token, q_nope, q_pe);
    init_pipeline_barriers(s, addr, warp_id);

    run_producer_warp(
        s, addr, lane, warp_id, token, sparse_indices, 
        total_kv_tokens, &kc_tmap, &kp_tmap, stages_total);

    run_mma_warps(s, addr, lane, warp_id, stages_total);
    run_score_epilogue_warps(s, addr, lane, warp_id, stages_total, sm_scale_log2e);
    run_value_epilogue_warps(s, addr, lane, warp_id, stages_total);
    __syncthreads();
    writeback_outputs(s, tid, token, out, out_lse);

    __syncthreads();
    if (warp_id == kScoreMmaWarp) {
        tcgen05_dealloc(static_cast<int>(static_cast<uint32_t>(s.tmem_addr_scratch)), kTotalTmemCols);
    }
}

static bool g_kernel_attrs_set = false;

static inline CUtensorMap make_tmap(
    void* ptr,
    uint32_t rank,
    const uint64_t* global_dim,
    const uint64_t* global_strides,
    const uint32_t* box_dim,
    const uint32_t* element_strides,
    CUtensorMapSwizzle swizzle,
    const char* err_msg
) {
    CUtensorMap tmap{};
    auto st = cuTensorMapEncodeTiled(
        &tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        ptr,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(st == CUDA_SUCCESS, err_msg);
    return tmap;
}

static inline void maybe_set_kernel_attrs() {
    if (!g_kernel_attrs_set) {
        cudaFuncSetAttribute(
            dsa_sparse_attn_kernel_v2,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kDesiredDynamicSmemBytes
        );
        g_kernel_attrs_set = true;
    }
}

void dsa_sparse_attn_launch(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale,
    torch::Tensor out,
    torch::Tensor lse
) {
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA");
    TORCH_CHECK(ckv_cache.is_cuda(), "ckv_cache must be CUDA");
    TORCH_CHECK(kpe_cache.is_cuda(), "kpe_cache must be CUDA");
    TORCH_CHECK(sparse_indices.is_cuda(), "sparse_indices must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lse must be CUDA");

    const int num_tokens = static_cast<int>(q_nope.size(0));
    if (num_tokens == 0) return;

    const int num_pages = static_cast<int>(ckv_cache.size(0));
    const int total_kv_tokens = num_pages * kPageSize;
    maybe_set_kernel_attrs();

    const uint8_t* kc_base = reinterpret_cast<const uint8_t*>(ckv_cache.data_ptr());
    const uint8_t* kp_base = reinterpret_cast<const uint8_t*>(kpe_cache.data_ptr());

    constexpr uint32_t rank2 = 2;
    uint32_t e2[rank2] = {1U, 1U};

    uint64_t kc_dim[rank2] = {
        static_cast<uint64_t>(kCkvRowBytes),
        static_cast<uint64_t>(total_kv_tokens),
    };
    uint64_t kc_strides[rank2 - 1] = {
        static_cast<uint64_t>(kCkvRowBytes),
    };
    uint32_t kc_box[rank2] = {
        static_cast<uint32_t>(kCkvChunkBytes),
        1U,
    };
    CUtensorMap kc_tmap = make_tmap(
        const_cast<uint8_t*>(kc_base),
        rank2,
        kc_dim,
        kc_strides,
        kc_box,
        e2,
        CU_TENSOR_MAP_SWIZZLE_128B,
        "cuTensorMapEncodeTiled failed for ckv"
    );

    uint64_t kp_dim[rank2] = {
        static_cast<uint64_t>(kKpeRowBytes),
        static_cast<uint64_t>(total_kv_tokens),
    };
    uint64_t kp_strides[rank2 - 1] = {
        static_cast<uint64_t>(kKpeRowBytes),
    };
    uint32_t kp_box[rank2] = {
        static_cast<uint32_t>(kKpeRowBytes),
        1U,
    };
    CUtensorMap kp_tmap = make_tmap(
        const_cast<uint8_t*>(kp_base),
        rank2,
        kp_dim,
        kp_strides,
        kp_box,
        e2,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        "cuTensorMapEncodeTiled failed for kpe"
    );

    const float sm_scale_log2e = static_cast<float>(sm_scale) * 1.4426950408889634f;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dsa_sparse_attn_kernel_v2<<<num_tokens, kThreadsPerBlock, kDesiredDynamicSmemBytes, stream>>>(
        kc_tmap,
        kp_tmap,
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
        reinterpret_cast<const int*>(sparse_indices.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<float*>(lse.data_ptr()),
        num_tokens,
        total_kv_tokens,
        sm_scale_log2e
    );

    cudaError_t st = cudaGetLastError();
    TORCH_CHECK(st == cudaSuccess, "dsa_sparse_attn_kernel_v2 launch failed: ", cudaGetErrorString(st));
}
"""

cpp_decl_src = """
#include <torch/extension.h>
void dsa_sparse_attn_launch(
    torch::Tensor q_nope,
    torch::Tensor q_pe,
    torch::Tensor ckv_cache,
    torch::Tensor kpe_cache,
    torch::Tensor sparse_indices,
    float sm_scale,
    torch::Tensor out,
    torch::Tensor lse);
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_sparse_attn_v2_ext",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_sparse_attn_launch"],
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


def custom_kernel(data: input_t) -> output_t:
    q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale = data
    num_tokens = int(q_nope.shape[0])
    if num_tokens == 0:
        out = torch.empty((0, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((0, NUM_QO_HEADS), dtype=torch.float32, device=q_nope.device)
        return out, lse

    out = torch.empty((num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty((num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=q_nope.device)

    mod = _get_module()
    mod.dsa_sparse_attn_launch(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        sparse_indices,
        sm_scale,
        out,
        lse,
    )
    return out, lse
