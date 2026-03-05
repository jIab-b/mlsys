"""DSA sparse-attention v3 — single-CTA per token, no split/combine.

Changes from v2:
- Value MMA: 4 tiles write to separate TMEM col regions (16 cols each), no per-tile serialization
- 1 score-ep wargroup + 1 value-ep wargroup (single owner, stage-ordered)
- No split-K: one CTA handles full TOPK for one token
- Barrier cleanup: removed per-tile value_mma_mbar/value_tile_mbar, replaced with per-stage

Architecture:
- 1 producer warp: direct coalesced gmem->smem loads for Kc[64,512] + Kp[64,64]
- 2 MMA warps:
    score MMA warp: Kc[64,512] @ Q_nope[16,512]^T + Kp[64,64] @ Q_pe[16,64]^T → [64,16] in TMEM
    value MMA warp: 4x Kc_slice[128,16] back-to-back → 4 separate 16-col TMEM regions
- 1 score-epilogue warpgroup (4 warps): direct TMEM->regs, online softmax, write bf16 weights
- 1 value-epilogue warpgroup (4 warps): read all 4 value tiles, online rescale + accum
- Total: 1 + 2 + 4 + 4 = 11 warps = 352 threads

Grid: one CTA per token, each CTA processes full TOPK window.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_module = None

def _cutlass_include_dir():
    """Find the directory to pass to -I so that #include <cute/tensor.hpp> works."""
    import glob, site
    for sp in site.getsitepackages():
        for match in glob.glob(os.path.join(sp, "**/cute/tensor.hpp"), recursive=True):
            return os.path.dirname(os.path.dirname(match))
    raise RuntimeError("cute/tensor.hpp not found in site-packages")

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

#include <cute/tensor.hpp>
constexpr int kNumHeads = 16;
constexpr int kHeadDimCkv = 512;
constexpr int kHeadDimKpe = 64;
constexpr int kPageSize = 64;
constexpr int kTopK = 2048;

constexpr int kStageTokens = 64;
constexpr int kNumStages = 2;

constexpr int kMmaK = 16;
constexpr int kMmaItersCkv = kHeadDimCkv / kMmaK;
constexpr int kMmaItersKpe = kHeadDimKpe / kMmaK;
constexpr int kValMmaMTile = 128;
constexpr int kValMmaMTiles = kHeadDimCkv / kValMmaMTile;
constexpr int kValMmaKIters = kStageTokens / kMmaK;

constexpr int kCkvRowBytes = kHeadDimCkv * sizeof(__nv_bfloat16);
constexpr int kKpeRowBytes = kHeadDimKpe * sizeof(__nv_bfloat16);
constexpr int kCkvChunkBytes = 128;
constexpr int kCkvChunksPerRow = kCkvRowBytes / kCkvChunkBytes;
constexpr int kNumScoreEpiWarpgroups = 2;
constexpr int kNumValueEpiWarpgroups = 1;
constexpr int kScoreEpiWarps = kNumScoreEpiWarpgroups * 4;
constexpr int kValueEpiWarps = kNumValueEpiWarpgroups * 4;
constexpr int kScoreMmaWarp = kScoreEpiWarps + kValueEpiWarps;
constexpr int kValueMmaWarp = kScoreMmaWarp + 1;
constexpr int kProducerWarp = kValueMmaWarp + 1;
constexpr int kNumWarps = kProducerWarp + 1;
constexpr int kThreadsPerBlock = kNumWarps * 32;
constexpr int kScoreTmemTotal = kNumStages * kStageTokens;
constexpr int kValueTmemBase = kScoreTmemTotal;
constexpr int kTotalTmemCols = 512;
struct __align__(1024) SmemLayout {
    __nv_bfloat16 q_nope[kNumHeads * kHeadDimCkv];
    __nv_bfloat16 q_pe[kNumHeads * kHeadDimKpe];
    __nv_bfloat16 kc_stage[kNumStages * kStageTokens * kHeadDimCkv];
    __nv_bfloat16 kp_stage[kNumStages * kStageTokens * kHeadDimKpe];
    int stage_indices[kNumStages * kStageTokens];
    __nv_bfloat16 attn_weights[kNumStages * kStageTokens * kNumHeads];
    __nv_bfloat16 kc_val_tile[kStageTokens * kValMmaMTile];
    float score_buf[kNumStages * kStageTokens * kNumHeads];
    float score_m_partial[4 * kNumHeads];
    float score_l_partial[4 * kNumHeads];
    float score_m_new[kNumHeads];
    float stage_alpha[kNumStages * kNumHeads];
    float m_state[kNumHeads];
    float l_state[kNumHeads];
    float o_accum[kNumHeads * kHeadDimCkv];
    uint64_t tma_mbar[kNumStages];
    uint64_t score_mma_mbar[kNumStages];
    uint64_t score_epi_mbar[kNumStages];
    uint64_t score_tmem_reuse_mbar[kNumStages];
    uint64_t value_mma_mbar[kNumStages];
    uint64_t value_epi_mbar[kNumStages];

    int tmem_addr_scratch;
};

constexpr int kDesiredDynamicSmemBytes = sizeof(SmemLayout);
struct SmemAddrs {
    int q_nope;
    int q_pe;
    int kc_stage;
    int kc_val_tile;
    int kp_stage;
    int attn_weights;
    int tma_mbar;
    int score_mma_mbar;
    int score_epi_mbar;
    int score_tmem_reuse_mbar;
    int value_mma_mbar;
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
    a.kc_val_tile = off(s->kc_val_tile);
    a.kp_stage = off(s->kp_stage);
    a.attn_weights = off(s->attn_weights);
    a.tma_mbar = off(s->tma_mbar);
    a.score_mma_mbar = off(s->score_mma_mbar);
    a.score_epi_mbar = off(s->score_epi_mbar);
    a.score_tmem_reuse_mbar = off(s->score_tmem_reuse_mbar);
    a.value_mma_mbar = off(s->value_mma_mbar);
    a.value_epi_mbar = off(s->value_epi_mbar);
    a.tmem_addr_scratch = off(&s->tmem_addr_scratch);
    return a;
}

__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{ .reg .pred %%px; elect.sync _|%%px, %1; @%%px mov.s32 %0, 1; }"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

template <cute::UMMA::Major Major, typename SwizzleAtom, int Rows, int Cols>
__device__ inline uint64_t make_umma_desc_from_smem(int smem_addr) {
    auto layout = cute::tile_to_shape(SwizzleAtom{}, cute::Shape<cute::Int<Rows>, cute::Int<Cols>>{});
    auto tensor = cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<__nv_bfloat16*>(smem_addr)),
        layout
    );
    return static_cast<uint64_t>(cute::UMMA::make_umma_desc<Major>(tensor));
}


__device__ inline uint64_t make_smem_desc_manual(int smem_addr, int lbo_bytes, int sbo_bytes,
    int matrix_base_offset, int swizzle) {
    // PTX spec 9.7.16.4.1: matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
    uint64_t d = 0;
    const uint64_t start = (static_cast<uint64_t>(static_cast<uint32_t>(smem_addr) & 0x3FFFFu) >> 4);
    const uint64_t lbo = (static_cast<uint64_t>(static_cast<uint32_t>(lbo_bytes) & 0x3FFFFu) >> 4);
    const uint64_t sbo = (static_cast<uint64_t>(static_cast<uint32_t>(sbo_bytes) & 0x3FFFFu) >> 4);

    const uint64_t base_off = static_cast<uint64_t>(matrix_base_offset & 0x7);
    const int swz_in = (swizzle & 0x7);
    const int swz = (swz_in == 0 || swz_in == 1 || swz_in == 2 || swz_in == 4 || swz_in == 6)
        ? swz_in : 0;

    d |= start;                           // bits 0-13: matrix start address (encoded)
    d |= lbo << 16;                       // bits 16-29: leading-dimension byte offset/address (encoded)
    d |= sbo << 32;                       // bits 32-45: stride-dimension byte offset (encoded)
    d |= uint64_t{1} << 46;               // bits 46-48: version field = 0b001
    d |= ((swz == 0) ? 0ull : base_off) << 49;  // bits 49-51: base offset (unused for no-swizzle)

    d |= static_cast<uint64_t>(swz) << 61;      // bits 61-63: swizzle mode

    return d;
}

__device__ inline int smem_idx_kmajor_inter(int rows, int row, int col) {
    // Swizzle<0,4,3> K-major interleaved canonical index (bf16 elements).
    return ((row & 7) * 8) +
           ((row >> 3) * 64) +
           (col & 7) +
           ((col >> 3) * rows * 8);
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
        :
        : "r"(dst_smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
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
    int tmem_d,
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

__device__ inline void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ inline void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
    int tid
) {
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
            mbarrier_init(addr.tma_mbar + i * sizeof(uint64_t), 1);
            mbarrier_init(addr.score_mma_mbar + i * sizeof(uint64_t), 1);
            mbarrier_init(addr.score_epi_mbar + i * sizeof(uint64_t), 1);
            mbarrier_init(addr.score_tmem_reuse_mbar + i * sizeof(uint64_t), 1);
            mbarrier_init(addr.value_mma_mbar + i * sizeof(uint64_t), 1);
            mbarrier_init(addr.value_epi_mbar + i * sizeof(uint64_t), 1);
        }
        s.tmem_addr_scratch = 0;
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();
}

__device__ inline void load_q_data(
    SmemLayout& s,
    int tid,
    int token,
    const __nv_bfloat16* q_nope_gmem,
    const __nv_bfloat16* q_pe_gmem
) {
    const __nv_bfloat16* qn_token =
        q_nope_gmem + 1LL * token * kNumHeads * kHeadDimCkv;
    const __nv_bfloat16* qp_token =
        q_pe_gmem + 1LL * token * kNumHeads * kHeadDimKpe;

    // Pack Q into K-major-interleaved [N,K] tiles: rows=kNumHeads, cols=dim.
    for (int idx = tid; idx < kNumHeads * kHeadDimCkv; idx += kThreadsPerBlock) {
        const int h = idx / kHeadDimCkv;
        const int k = idx - h * kHeadDimCkv;
        const int dst = smem_idx_kmajor_inter(kNumHeads, h, k);
        s.q_nope[dst] = qn_token[idx];
    }
    for (int idx = tid; idx < kNumHeads * kHeadDimKpe; idx += kThreadsPerBlock) {
        const int h = idx / kHeadDimKpe;
        const int k = idx - h * kHeadDimKpe;
        const int dst = smem_idx_kmajor_inter(kNumHeads, h, k);
        s.q_pe[dst] = qp_token[idx];
    }

    // Make generic shared writes visible to tcgen05 async proxy before score MMA.
    fence_proxy_async_shared_cta();
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
    const __nv_bfloat16* kc_rows,
    const __nv_bfloat16* kp_rows,
    int stages_total,
    int block_start
) {
    if (warp_id != kProducerWarp) return;

    for (int local = 0; local < stages_total; ++local) {
        const int stage = local % kNumStages;
        const int stage_start = (block_start + local) * kStageTokens;

        if (lane == 0 && local >= kNumStages) {
            const int phase = ((local - kNumStages) / kNumStages) & 1;
            mbarrier_wait_parity(
                addr.value_epi_mbar + stage * sizeof(uint64_t),
                phase);
        }

        for (int i = lane; i < kStageTokens; i += 32) {
            const int idx = sparse_indices[token * kTopK + stage_start + i];
            const bool valid = (idx >= 0 && idx < total_kv_tokens);
            s.stage_indices[stage * kStageTokens + i] = valid ? idx : -1;
        }
        __syncwarp();

        for (int i = 0; i < kStageTokens; ++i) {
            const int idx = s.stage_indices[stage * kStageTokens + i];
            __nv_bfloat16* kc_dst = s.kc_stage + stage * kStageTokens * kHeadDimCkv;
            __nv_bfloat16* kp_dst = s.kp_stage + stage * kStageTokens * kHeadDimKpe;
            if (idx >= 0) {
                const __nv_bfloat16* kc_src = kc_rows + 1LL * idx * kHeadDimCkv;
                const __nv_bfloat16* kp_src = kp_rows + 1LL * idx * kHeadDimKpe;
                for (int d = lane; d < kHeadDimCkv; d += 32) {
                    const int dst = smem_idx_kmajor_inter(kStageTokens, i, d);
                    kc_dst[dst] = kc_src[d];
                }
                for (int d = lane; d < kHeadDimKpe; d += 32) {
                    const int dst = smem_idx_kmajor_inter(kStageTokens, i, d);
                    kp_dst[dst] = kp_src[d];
                }
            } else {
                for (int d = lane; d < kHeadDimCkv; d += 32) {
                    const int dst = smem_idx_kmajor_inter(kStageTokens, i, d);
                    kc_dst[dst] = float_to_bf16(0.0f);
                }
                for (int d = lane; d < kHeadDimKpe; d += 32) {
                    const int dst = smem_idx_kmajor_inter(kStageTokens, i, d);
                    kp_dst[dst] = float_to_bf16(0.0f);
                }
            }
        }
        __syncwarp();

        // Make producer-warp generic shared writes visible to tcgen05 async proxy.
        fence_proxy_async_shared_cta();
        // Order producer shared writes before stage-ready signaling to consumer MMA warp.
        tcgen05_fence_before_thread_sync();
        __syncwarp();

        if (lane == 0) {
            const int mbar = addr.tma_mbar + stage * sizeof(uint64_t);
            mbarrier_arrive(mbar);
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

    constexpr uint32_t kScoreIdesc = (1U << 7U) |
                                     (1U << 10U) |
                                 //    (1U << 16U) |
                                     (1U << 4U) |
                                     ((uint32_t)(kNumHeads >> 3U) << 17U) |
                                     ((uint32_t)(kStageTokens >> 4U) << 24U);
    constexpr uint32_t kValueIdesc = (1U << 7U) |
                                     (1U << 10U) |
                                //     (1U << 15U) |
                                     (1U << 4U) |
                                     ((uint32_t)(kNumHeads >> 3U) << 17U) |
                                     ((uint32_t)(kValMmaMTile >> 4U) << 24U);

    if (warp_id == kScoreMmaWarp) {tcgen05_alloc(addr.tmem_addr_scratch, kTotalTmemCols);}

    if (warp_id == kScoreMmaWarp && lane == 0) {
        const int tmem_base = s.tmem_addr_scratch;
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int tmem_slot = local % kNumStages;
            const int phase = (local / kNumStages) & 1;

            mbarrier_wait_parity(
                addr.tma_mbar + stage * sizeof(uint64_t),
                phase);
            // Synchronize stage handoff before issuing MMA that reads shared-memory descriptors.
            tcgen05_fence_after_thread_sync();

            if (local >= kNumStages) {
                const int reuse_phase = ((local - kNumStages) / kNumStages) & 1;
                mbarrier_wait_parity(
                    addr.score_tmem_reuse_mbar + stage * sizeof(uint64_t),
                    reuse_phase);
            }

            const int tmem_d = tmem_base + tmem_slot * kStageTokens;

            // uint64_t kc_desc = make_umma_desc_from_smem<
            //     cute::UMMA::Major::K, cute::UMMA::Layout_K_INTER_Atom<__nv_bfloat16>,
            //     kStageTokens, kMmaK>(
            //     addr.kc_stage + stage * kStageTokens * kCkvRowBytes);
            uint64_t kc_desc = make_smem_desc_manual(
                addr.kc_stage + stage * kStageTokens * kCkvRowBytes,
                kStageTokens * 8 * sizeof(__nv_bfloat16),
                8 * 8 * sizeof(__nv_bfloat16),
                0, 0);

            // uint64_t qn_desc = make_umma_desc_from_smem<
            //     cute::UMMA::Major::K, cute::UMMA::Layout_K_INTER_Atom<__nv_bfloat16>,
            //     kNumHeads, kMmaK>(addr.q_nope);
            uint64_t qn_desc = make_smem_desc_manual(
                addr.q_nope,
                kNumHeads * 8 * sizeof(__nv_bfloat16),
                8 * 8 * sizeof(__nv_bfloat16),
                0, 0);
            #pragma unroll
            for (int ki = 0; ki < kMmaItersCkv; ++ki) {
                tcgen05_mma_f16(tmem_d, kc_desc, qn_desc, kScoreIdesc, (ki > 0) ? 1 : 0);
                kc_desc += kStageTokens * kMmaK * sizeof(__nv_bfloat16) / 16;
                qn_desc += kNumHeads * kMmaK * sizeof(__nv_bfloat16) / 16;
            }

            // Kp: K-major, no swizzle, tile [kStageTokens, kMmaK]
            // uint64_t kp_desc = make_umma_desc_from_smem<
            //     cute::UMMA::Major::K, cute::UMMA::Layout_K_INTER_Atom<__nv_bfloat16>,
            //     kStageTokens, kMmaK>(
            //     addr.kp_stage + stage * kStageTokens * kKpeRowBytes);
            uint64_t kp_desc = make_smem_desc_manual(
                addr.kp_stage + stage * kStageTokens * kKpeRowBytes,
                kStageTokens * 8 * sizeof(__nv_bfloat16),
                8 * 8 * sizeof(__nv_bfloat16),
                0, 0);

            // Q_pe: K-major, tile [kNumHeads, kMmaK]
            // uint64_t qp_desc = make_umma_desc_from_smem<
            //     cute::UMMA::Major::K, cute::UMMA::Layout_K_INTER_Atom<__nv_bfloat16>,
            //     kNumHeads, kMmaK>(addr.q_pe);
            uint64_t qp_desc = make_smem_desc_manual(
                addr.q_pe,
                kNumHeads * 8 * sizeof(__nv_bfloat16),
                8 * 8 * sizeof(__nv_bfloat16),
                0, 0);
            #pragma unroll
            for (int ki = 0; ki < kMmaItersKpe; ++ki) {
                tcgen05_mma_f16(tmem_d, kp_desc, qp_desc, kScoreIdesc, 1);
                kp_desc += kStageTokens * kMmaK * sizeof(__nv_bfloat16) / 16;
                qp_desc += kNumHeads * kMmaK * sizeof(__nv_bfloat16) / 16;
            }

            tcgen05_commit(addr.score_mma_mbar + stage * sizeof(uint64_t));
        }
    }
    if (warp_id == kValueMmaWarp && lane == 0) {
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int phase = (local / kNumStages) & 1;
            mbarrier_wait_parity(
                addr.score_epi_mbar + stage * sizeof(uint64_t),
                phase);
            // score_ep wrote attn_weights through generic proxy; order before value MMA consume.
            tcgen05_fence_after_thread_sync();
            const int tmem_base = s.tmem_addr_scratch;

            // attn_weights: K-major, no swizzle, tile [kNumHeads, kMmaK]
            // const uint64_t w_desc_base = make_umma_desc_from_smem<
            //     cute::UMMA::Major::K, cute::UMMA::Layout_K_INTER_Atom<__nv_bfloat16>,
            //     kNumHeads, kMmaK>(
            //     addr.attn_weights + stage * kStageTokens * kNumHeads * sizeof(__nv_bfloat16));
            const uint64_t w_desc_base = make_smem_desc_manual(
                addr.attn_weights + stage * kStageTokens * kNumHeads * sizeof(__nv_bfloat16),
                kNumHeads * 8 * sizeof(__nv_bfloat16),
                8 * 8 * sizeof(__nv_bfloat16),
                0, 0);

            for (int tile = 0; tile < kValMmaMTiles; ++tile) {
                const int tmem_d = tmem_base + kValueTmemBase + tile * kNumHeads;
                uint64_t kc_col_desc = make_smem_desc_manual(
                    addr.kc_val_tile,
                    kValMmaMTile * 8 * sizeof(__nv_bfloat16),
                    8 * 8 * sizeof(__nv_bfloat16),
                    0, 0);
                uint64_t w_desc = w_desc_base;

                for (int ki = 0; ki < kValMmaKIters; ++ki) {
                    tcgen05_mma_f16(tmem_d, kc_col_desc, w_desc, kValueIdesc, (ki > 0) ? 1 : 0);
                    kc_col_desc += kValMmaMTile * kMmaK * sizeof(__nv_bfloat16) / 16;
                    w_desc += kNumHeads * kMmaK * sizeof(__nv_bfloat16) / 16;
                }
            }
            tcgen05_commit(addr.value_mma_mbar + stage * sizeof(uint64_t));
        }
    }
}

__device__ inline void wg_sync(int bar_slot) {
    asm volatile("bar.sync %0, 128;" :: "r"(bar_slot) : "memory");
}

__device__ inline void score_ep_dualwg_sync() {
    asm volatile("bar.sync 13, 256;" ::: "memory");
}

__device__ inline float warp_reduce_max(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

__device__ inline float warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

__device__ inline float half_warp_reduce_max(float v) {
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
    return v;
}

__device__ inline float half_warp_reduce_sum(float v) {
    #pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, mask);
    return v;
}

__device__ inline void score_ep_stage_regs(
    SmemLayout& s,
    int lane,
    int warp_id,
    int stage,
    int tmem_slot,
    float sm_scale_log2e,
    bool is_last_stage,
    float* lse_ptr
) {
    // tcgen05.ld.32x32b.x16: lanes 0-15 from each warp hold data.
    // tok = warp_id * 16 + lane, active when lane < 16.
    const bool active = (lane < 16);
    const int tok = warp_id * 16 + lane;
    const int tmem_col_base = s.tmem_addr_scratch + tmem_slot * kStageTokens;
    float vals[16];
    tcgen05_ld_32x32b_16(warp_id * 32, tmem_col_base, vals);
    tcgen05_wait_ld();

    // --- per-head max within each warp (half-warp reduce over 16 active lanes) ---
    float m_local[kNumHeads];
    #pragma unroll
    for (int head = 0; head < kNumHeads; ++head) {
        m_local[head] = active ? vals[head] * sm_scale_log2e : -INFINITY;
    }
    #pragma unroll
    for (int head = 0; head < kNumHeads; ++head) {
        m_local[head] = half_warp_reduce_max(m_local[head]);
    }

    // --- merge max across 4 warps, compute alpha (warp 0, lanes 0-15, one head per lane) ---
    // Lane h in warp w writes its warp's max for head h.
    if (lane < kNumHeads) {
        s.score_m_partial[warp_id * kNumHeads + lane] = m_local[lane];
    }
    wg_sync(10);

    float m_new_h, alpha_h;
    if (warp_id == 0 && lane < kNumHeads) {
        float m_tile = s.score_m_partial[lane];
        #pragma unroll
        for (int w = 1; w < 4; ++w)
            m_tile = fmaxf(m_tile, s.score_m_partial[w * kNumHeads + lane]);
        const float m_prev = s.m_state[lane];
        m_new_h = fmaxf(m_prev, m_tile);
        alpha_h = (m_prev == -INFINITY) ? 0.0f : exp2f(m_prev - m_new_h);
        s.score_m_new[lane] = m_new_h;
        s.stage_alpha[stage * kNumHeads + lane] = alpha_h;
    }
    wg_sync(10);

    // --- softmax weights + l reduction ---
    float l_local[kNumHeads];
    #pragma unroll
    for (int head = 0; head < kNumHeads; ++head) {
        float w = 0.0f;
        if (active) {
            w = exp2f(vals[head] * sm_scale_log2e - s.score_m_new[head]);
            const int base = stage * kStageTokens * kNumHeads;
            const int dst = smem_idx_kmajor_inter(kNumHeads, head, tok);
            s.attn_weights[base + dst] = float_to_bf16(w);
        }
        l_local[head] = w;
    }
    #pragma unroll
    for (int head = 0; head < kNumHeads; ++head) {
        l_local[head] = half_warp_reduce_sum(l_local[head]);
    }

    // --- merge l across 4 warps, update state (warp 0, lanes 0-15) ---
    if (lane < kNumHeads) {
        s.score_l_partial[warp_id * kNumHeads + lane] = l_local[lane];
    }
    wg_sync(10);

    if (warp_id == 0 && lane < kNumHeads) {
        const float alpha = s.stage_alpha[stage * kNumHeads + lane];
        const float l_prev = s.l_state[lane];
        float l_tile = s.score_l_partial[lane];
        #pragma unroll
        for (int w = 1; w < 4; ++w)
            l_tile += s.score_l_partial[w * kNumHeads + lane];
        const float l_new = fmaf(l_prev, alpha, l_tile);
        const float m_new = s.score_m_new[lane];
        s.m_state[lane] = m_new;
        s.l_state[lane] = l_new;
        if (is_last_stage) {
            lse_ptr[lane] = (l_new > 0.0f && m_new != -INFINITY)
                ? m_new + log2f(l_new) : -INFINITY;
        }
    }
    wg_sync(10);
}

__device__ inline void run_score_epilogue_warps(
    SmemLayout& s,
    const SmemAddrs& addr,
    int lane,
    int warp_id,
    int stages_total,
    float sm_scale_log2e,
    float* lse_ptr
) {
    if (warp_id >= kScoreEpiWarps) return;
    const int score_main_warps = kNumScoreEpiWarpgroups == 0 ? 0 : 4;
    if (warp_id < score_main_warps) {
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int phase = (local / kNumStages) & 1;

            if (lane == 0) {
                mbarrier_wait_parity(
                    addr.score_mma_mbar + stage * sizeof(uint64_t),
                    phase);
            }
            wg_sync(10);
            tcgen05_fence_after_thread_sync();

            const bool is_last_stage = (local + 1 >= stages_total);
            score_ep_stage_regs(
                s, lane, warp_id, stage, stage, sm_scale_log2e, is_last_stage, lse_ptr);
            // Make attn_weights writes visible before signaling score->value handoff.
            fence_proxy_async_shared_cta();
            wg_sync(10);
            score_ep_dualwg_sync();

            if (warp_id == 0 && lane == 0) {
                mbarrier_arrive(addr.score_epi_mbar + stage * sizeof(uint64_t));
                mbarrier_arrive(addr.score_tmem_reuse_mbar + stage * sizeof(uint64_t));
            }
        }
        return;
    }

    // Secondary score-ep warpgroup: transpose kc_stage tile-by-tile into kc_val_tile
    // so value MMA can use standard K-major, non-transposed A descriptors.
    const int tx_warp = warp_id - score_main_warps;
    const int tx_tid = tx_warp * 32 + lane;
    for (int local = 0; local < stages_total; ++local) {
        const int stage = local % kNumStages;
        const int phase = (local / kNumStages) & 1;
        if (lane == 0) {
            mbarrier_wait_parity(
                addr.score_mma_mbar + stage * sizeof(uint64_t),
                phase);
        }
        wg_sync(11);
        tcgen05_fence_after_thread_sync();

        const int src_stage_base = stage * kStageTokens * kHeadDimCkv;
        for (int tile = 0; tile < kValMmaMTiles; ++tile) {
            wg_sync(11);

            for (int idx = tx_tid; idx < kValMmaMTile * kStageTokens; idx += 128) {
                const int dim_in_tile = idx / kStageTokens;
                const int tok = idx - dim_in_tile * kStageTokens;
                const int src_dim = tile * kValMmaMTile + dim_in_tile;
                const int src = smem_idx_kmajor_inter(kStageTokens, tok, src_dim);
                const int dst = smem_idx_kmajor_inter(kValMmaMTile, dim_in_tile, tok);
                s.kc_val_tile[dst] = s.kc_stage[src_stage_base + src];
            }
            wg_sync(11);
        }
        // Make kc_val_tile writes visible to value MMA (tcgen05 async proxy).
        fence_proxy_async_shared_cta();
        wg_sync(11);
        score_ep_dualwg_sync();
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
    const int value_epi_wg_id = warp_id - kScoreEpiWarps;

    for (int local = 0; local < stages_total; ++local) {
        const int stage = local % kNumStages;
        const int phase = (local / kNumStages) & 1;

        if (lane == 0) {
            mbarrier_wait_parity(
                addr.value_mma_mbar + stage * sizeof(uint64_t),
                phase);
        }
        wg_sync(12);

        tcgen05_fence_after_thread_sync();
        const int lane_base = value_epi_wg_id * 32;
        const int dim_in_tile = value_epi_wg_id * 32 + lane;

        if (dim_in_tile < kValMmaMTile) {
            for (int tile = 0; tile < kValMmaMTiles; ++tile) {
                float vals[16];
                const int tmem_col_base =
                    s.tmem_addr_scratch + kValueTmemBase + tile * kNumHeads;
                tcgen05_ld_32x32b_16(lane_base, tmem_col_base, vals);
                tcgen05_wait_ld();
                #pragma unroll
                for (int head = 0; head < kNumHeads; ++head) {
                    const float alpha = s.stage_alpha[stage * kNumHeads + head];
                    const int out_idx = head * kHeadDimCkv + tile * kValMmaMTile + dim_in_tile;
                    s.o_accum[out_idx] = s.o_accum[out_idx] * alpha + vals[head];
                }
            }
        }
        wg_sync(12);
        if (warp_id == kScoreEpiWarps && lane == 0) {
            mbarrier_arrive(addr.value_epi_mbar + stage * sizeof(uint64_t));
        }
    }
}

__device__ inline void writeback_outputs_final(
    SmemLayout& s,
    int tid,
    int token,
    __nv_bfloat16* out
) {
    __nv_bfloat16* o_token = out + 1LL * token * kNumHeads * kHeadDimCkv;

    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        const int head = i / kHeadDimCkv;
        const float l = s.l_state[head];
        const float v = s.o_accum[i];
        o_token[i] = __float2bfloat16((l > 0.0f) ? (v / l) : 0.0f);
    }
}

__global__ __launch_bounds__(kThreadsPerBlock) void dsa_sparse_attn_kernel_v3(
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe,
    const __nv_bfloat16* ckv_rows,
    const __nv_bfloat16* kpe_rows,
    const int* sparse_indices,
    __nv_bfloat16* out,
    float* lse,
    int num_tokens,
    int total_kv_tokens,
    float sm_scale_log2e
) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;
    constexpr int stages_total = kTopK / kStageTokens;
    constexpr int block_start = 0;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    extern __shared__ __align__(1024) SmemLayout smem_storage[];
    SmemLayout& s = smem_storage[0];
    const SmemAddrs addr = init_smem_addrs(&s);

    init_query_and_state(s, tid);
    init_pipeline_barriers(s, addr, warp_id);
    load_q_data(s, tid, token, q_nope, q_pe);

    run_producer_warp(
        s, addr, lane, warp_id, token, sparse_indices,
        total_kv_tokens, ckv_rows, kpe_rows, stages_total, block_start);

    run_mma_warps(s, addr, lane, warp_id, stages_total);
    float* lse_ptr = lse + 1LL * token * kNumHeads;
    run_score_epilogue_warps(s, addr, lane, warp_id, stages_total, sm_scale_log2e, lse_ptr);
    run_value_epilogue_warps(s, addr, lane, warp_id, stages_total);
    __syncthreads();

    writeback_outputs_final(s, tid, token, out);

    __syncthreads();
    if (warp_id == kScoreMmaWarp) {
        tcgen05_dealloc(s.tmem_addr_scratch, kTotalTmemCols);
    }
}

static bool g_kernel_attrs_set = false;

static inline void maybe_set_kernel_attrs() {
    if (!g_kernel_attrs_set) {
        cudaFuncSetAttribute(
            dsa_sparse_attn_kernel_v3,
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

    const int num_tokens = q_nope.size(0);
    if (num_tokens == 0) return;

    const int num_pages = ckv_cache.size(0);
    const int total_kv_tokens = num_pages * kPageSize;
    maybe_set_kernel_attrs();

    const float sm_scale_log2e = sm_scale * 1.4426950408889634f;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dsa_sparse_attn_kernel_v3<<<num_tokens, kThreadsPerBlock, kDesiredDynamicSmemBytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(ckv_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(kpe_cache.data_ptr()),
        reinterpret_cast<const int*>(sparse_indices.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<float*>(lse.data_ptr()),
        num_tokens,
        total_kv_tokens,
        sm_scale_log2e
    );

    {
        cudaError_t st = cudaGetLastError();
        TORCH_CHECK(st == cudaSuccess, "dsa_sparse_attn_kernel_v3 launch failed: ", cudaGetErrorString(st));
    }
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
            name="dsa_sparse_attn_v3_ext",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_sparse_attn_launch"],
            no_implicit_headers=True,
            extra_cuda_cflags=[
                "-O1",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--split-compile=8",
                "--threads 4",
                "--relocatable-device-code=false",
                f"-I{_cutlass_include_dir()}",
            ],
            extra_ldflags=["-lcuda"],
        )
    return _module


def compile_kernel():
    _get_module()


def custom_kernel(data):
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
