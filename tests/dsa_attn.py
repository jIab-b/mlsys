"""DSA sparse-attention submission — single CUDA launch, no graph, no second kernel.

Kernel design implemented in CUDA:
- Producer warp: staged sparse-KV loading (TMA row copies) into shared memory
- MMA warp: tcgen05 `.kind::f16` pipelined matmul (Kc@Q_nope^T + Kp@Q_pe^T)
- Epilogue warps: online softmax + value accumulation, final output/lse writeback
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

constexpr int kNumHeads = 16;
constexpr int kHeadDimCkv = 512;
constexpr int kHeadDimKpe = 64;
constexpr int kPageSize = 64;
constexpr int kTopK = 2048;

constexpr int kStageTokens = 64;
constexpr int kNumStages = 2;
constexpr int kNumTmemSlots = 2;
static_assert(kNumTmemSlots * kStageTokens <= 512, "TMEM columns exceed hardware limit");

constexpr int kNumEpilogueWarps = 4;
constexpr int kProducerWarp = kNumEpilogueWarps;
constexpr int kMmaWarp = kProducerWarp + 1;
constexpr int kNumWarps = kMmaWarp + 1;
constexpr int kThreadsPerBlock = kNumWarps * 32;

constexpr int kMmaK = 32;
constexpr int kMmaItersCkv = kHeadDimCkv / kMmaK;  // 16
constexpr int kMmaItersKpe = kHeadDimKpe / kMmaK;  // 2

constexpr int kCkvRowBytes = kHeadDimCkv * static_cast<int>(sizeof(__nv_bfloat16));  // 1024
constexpr int kKpeRowBytes = kHeadDimKpe * static_cast<int>(sizeof(__nv_bfloat16));  // 128

struct __align__(1024) SmemLayout {
    __nv_bfloat16 q_nope[kNumHeads * kHeadDimCkv];
    __nv_bfloat16 q_pe[kNumHeads * kHeadDimKpe];

    __nv_bfloat16 kc_stage[kNumStages * kStageTokens * kHeadDimCkv];
    __nv_bfloat16 kp_stage[kNumStages * kStageTokens * kHeadDimKpe];

    int stage_indices[kNumStages * kStageTokens];
    int stage_valid[kNumStages];

    // Per-stage logits in log2-domain, shape [stage, head, tok]
    float logits_stage[kNumStages * kNumHeads * kStageTokens];

    // Online softmax state (per head), and accumulated numerator (per head, dim)
    float m_state[kNumHeads];
    float l_state[kNumHeads];
    float o_state[kNumHeads * kHeadDimCkv];

    int tma_phase[kNumStages];
    int mma_phase[kNumStages];
    int epi_phase[kNumStages];

    uint64_t tma_mbar[kNumStages];
    uint64_t mma_mbar[kNumStages];
    uint64_t epi_mbar[kNumStages];

    int tmem_addr_scratch;
};

constexpr int kDesiredDynamicSmemBytes = static_cast<int>(sizeof(SmemLayout));

struct SmemAddrs {
    int q_nope;
    int q_pe;
    int kc_stage;
    int kp_stage;
    int tma_mbar;
    int mma_mbar;
    int epi_mbar;
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
    a.tma_mbar = off(s->tma_mbar);
    a.mma_mbar = off(s->mma_mbar);
    a.epi_mbar = off(s->epi_mbar);
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

__device__ inline void ep_barrier() {
    asm volatile("bar.sync 3, 128;" ::: "memory");
}

__global__ __launch_bounds__(kThreadsPerBlock) void dsa_sparse_attn_kernel(
    const __grid_constant__ CUtensorMap kc_tmap,
    const __grid_constant__ CUtensorMap kp_tmap,
    const __nv_bfloat16* q_nope,
    const __nv_bfloat16* q_pe,
    const int* sparse_indices,
    __nv_bfloat16* out,
    float* out_lse,
    int num_tokens,
    int total_kv_tokens,
    float sm_scale_log2e
) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    extern __shared__ __align__(1024) SmemLayout smem_storage[];
    SmemLayout& s = smem_storage[0];
    const SmemAddrs addr = init_smem_addrs(&s);

    // Load query once per token into shared memory.
    const __nv_bfloat16* qn_src = q_nope + static_cast<int64_t>(token) * kNumHeads * kHeadDimCkv;
    const __nv_bfloat16* qp_src = q_pe + static_cast<int64_t>(token) * kNumHeads * kHeadDimKpe;
    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        s.q_nope[i] = qn_src[i];
    }
    for (int i = tid; i < kNumHeads * kHeadDimKpe; i += kThreadsPerBlock) {
        s.q_pe[i] = qp_src[i];
    }

    // Initialize online softmax/output state.
    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        s.o_state[i] = 0.0f;
    }
    if (tid < kNumHeads) {
        s.m_state[tid] = -INFINITY;
        s.l_state[tid] = 0.0f;
    }
    __syncthreads();

    if (warp_id == kProducerWarp && elect_sync()) {
        for (int i = 0; i < kNumStages; ++i) {
            mbarrier_init(addr.tma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.mma_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(addr.epi_mbar + i * static_cast<int>(sizeof(uint64_t)), 1);
            s.tma_phase[i] = 0;
            s.mma_phase[i] = 0;
            s.epi_phase[i] = 0;
        }
        s.tmem_addr_scratch = 0;
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    constexpr uint32_t kIdescMN = (0U << 7U) |
                                  (0U << 10U) |
                                  (1U << 4U) |
                                  ((uint32_t)(kStageTokens >> 3U) << 17U) |
                                  ((uint32_t)(kNumHeads >> 4U) << 24U);

    const int stages_total = (kTopK + kStageTokens - 1) / kStageTokens;  // 32

    // Producer: load sparse stages into shared memory.
    if (warp_id == kProducerWarp) {
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int stage_start = local * kStageTokens;

            if (lane == 0 && local >= kNumStages) {
                mbarrier_wait_parity(
                    addr.epi_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                    s.epi_phase[stage]);
                s.epi_phase[stage] ^= 1;
            }
            __syncwarp();

            // Index decode and zero-fill invalid rows.
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
                    for (int i = 0; i < kStageTokens; ++i) {
                        const int idx = s.stage_indices[stage * kStageTokens + i];
                        if (idx < 0) continue;
                        const int kc_dst = addr.kc_stage + ((stage * kStageTokens + i) * kCkvRowBytes);
                        const int kp_dst = addr.kp_stage + ((stage * kStageTokens + i) * kKpeRowBytes);
                        tma_2d_gmem2smem(kc_dst, &kc_tmap, 0, idx, mbar, 0ULL);
                        tma_2d_gmem2smem(kp_dst, &kp_tmap, 0, idx, mbar, 0ULL);
                    }
                } else {
                    mbarrier_arrive(mbar);
                }
            }
            __syncwarp();
        }
    }

    if (warp_id == kMmaWarp) {
        tcgen05_alloc(addr.tmem_addr_scratch, kNumTmemSlots * kStageTokens);
    }
    __syncthreads();

    // MMA warp: Kc@Q_nope^T + Kp@Q_pe^T
    if (warp_id == kMmaWarp && elect_sync()) {
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int tmem_slot = local % kNumTmemSlots;

            mbarrier_wait_parity(
                addr.tma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                s.tma_phase[stage]);
            s.tma_phase[stage] ^= 1;

            if (s.stage_valid[stage] > 0) {
                const uint32_t tmem_d = static_cast<uint32_t>(tmem_slot * kStageTokens);

                // Pass 1: noPE branch, K=512.
                uint64_t qn_desc = make_desc_kmajor_swizzle_128b(addr.q_nope);
                uint64_t kc_desc = make_desc_kmajor_swizzle_128b(
                    addr.kc_stage + stage * kStageTokens * kCkvRowBytes);
                for (int ki = 0; ki < kMmaItersCkv; ++ki) {
                    tcgen05_mma_f16(tmem_d, qn_desc, kc_desc, kIdescMN, (ki > 0) ? 1 : 0);
                    qn_desc += (kMmaK >> 4);
                    kc_desc += (kMmaK >> 4);
                }

                // Pass 2: PE branch, K=64, accumulate into D.
                uint64_t qp_desc = make_desc_kmajor_swizzle_128b(addr.q_pe);
                uint64_t kp_desc = make_desc_kmajor_swizzle_128b(
                    addr.kp_stage + stage * kStageTokens * kKpeRowBytes);
                for (int ki = 0; ki < kMmaItersKpe; ++ki) {
                    tcgen05_mma_f16(tmem_d, qp_desc, kp_desc, kIdescMN, 1);
                    qp_desc += (kMmaK >> 4);
                    kp_desc += (kMmaK >> 4);
                }

                tcgen05_commit(addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            } else {
                mbarrier_arrive(addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            }
        }
    }

    // Epilogue warpgroup: read logits, online softmax, output accumulation.
    if (warp_id < kNumEpilogueWarps) {
        for (int local = 0; local < stages_total; ++local) {
            const int stage = local % kNumStages;
            const int tmem_slot = local % kNumTmemSlots;

            if (warp_id == 0 && elect_sync()) {
                mbarrier_wait_parity(
                    addr.mma_mbar + stage * static_cast<int>(sizeof(uint64_t)),
                    s.mma_phase[stage]);
                s.mma_phase[stage] ^= 1;
            }
            ep_barrier();
            tcgen05_fence_after_thread_sync();

            // Warp 0 reads MMA tile from TMEM into shared logits buffer.
            if (warp_id == 0 && lane < kNumHeads) {
                const int head = lane;
                const int lane_base = 0;
                const int tmem_col_base = tmem_slot * kStageTokens;
                float vals[16];
                #pragma unroll
                for (int tok_off = 0; tok_off < kStageTokens; tok_off += 16) {
                    tcgen05_ld_32x32b_16(lane_base, tmem_col_base + tok_off, vals);
                    tcgen05_wait_ld();
                    #pragma unroll
                    for (int t = 0; t < 16; ++t) {
                        s.logits_stage[(stage * kNumHeads + head) * kStageTokens + tok_off + t] =
                            vals[t] * sm_scale_log2e;
                    }
                }
            }
            ep_barrier();

            // One lane per ep-warp updates 4 heads (keeps code deterministic/simple).
            if (lane == 0) {
                const int h0 = warp_id * 4;
                const int h1 = h0 + 4;
                for (int head = h0; head < h1; ++head) {
                    float m_prev = s.m_state[head];
                    float l_prev = s.l_state[head];
                    float m_tile = -INFINITY;

                    for (int tok = 0; tok < kStageTokens; ++tok) {
                        const int idx = s.stage_indices[stage * kStageTokens + tok];
                        if (idx < 0) continue;
                        const float x = s.logits_stage[(stage * kNumHeads + head) * kStageTokens + tok];
                        m_tile = fmaxf(m_tile, x);
                    }
                    if (m_tile == -INFINITY) {
                        continue;
                    }

                    const float m_new = fmaxf(m_prev, m_tile);
                    const float alpha = (m_prev == -INFINITY) ? 0.0f : exp2f(m_prev - m_new);

                    float* o = s.o_state + head * kHeadDimCkv;
                    for (int d = 0; d < kHeadDimCkv; ++d) {
                        o[d] *= alpha;
                    }

                    float l_new = l_prev * alpha;
                    for (int tok = 0; tok < kStageTokens; ++tok) {
                        const int idx = s.stage_indices[stage * kStageTokens + tok];
                        if (idx < 0) continue;

                        const float x = s.logits_stage[(stage * kNumHeads + head) * kStageTokens + tok];
                        const float beta = exp2f(x - m_new);
                        l_new += beta;

                        const __nv_bfloat16* kc_row =
                            s.kc_stage + (stage * kStageTokens + tok) * kHeadDimCkv;
                        for (int d = 0; d < kHeadDimCkv; ++d) {
                            o[d] += beta * bf16_to_float(kc_row[d]);
                        }
                    }

                    s.m_state[head] = m_new;
                    s.l_state[head] = l_new;
                }
            }
            ep_barrier();

            if (warp_id == 0 && elect_sync()) {
                mbarrier_arrive(addr.epi_mbar + stage * static_cast<int>(sizeof(uint64_t)));
            }
            ep_barrier();
        }
    }

    __syncthreads();
    if (warp_id == kMmaWarp) {
        tcgen05_dealloc(0, kNumTmemSlots * kStageTokens);
    }
    __syncthreads();

    // Final normalize and write outputs/lse.
    __nv_bfloat16* out_token = out + static_cast<int64_t>(token) * kNumHeads * kHeadDimCkv;
    float* lse_token = out_lse + static_cast<int64_t>(token) * kNumHeads;

    for (int i = tid; i < kNumHeads * kHeadDimCkv; i += kThreadsPerBlock) {
        const int head = i / kHeadDimCkv;
        const float l = s.l_state[head];
        const float v = (l > 0.0f) ? (s.o_state[i] / l) : 0.0f;
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
            dsa_sparse_attn_kernel,
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

    // Flatten token rows as [row_bytes, total_tokens] on uint8 tensor-map.
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
        static_cast<uint32_t>(kCkvRowBytes),
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
    dsa_sparse_attn_kernel<<<num_tokens, kThreadsPerBlock, kDesiredDynamicSmemBytes, stream>>>(
        kc_tmap,
        kp_tmap,
        reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
        reinterpret_cast<const int*>(sparse_indices.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        reinterpret_cast<float*>(lse.data_ptr()),
        num_tokens,
        total_kv_tokens,
        sm_scale_log2e
    );

    cudaError_t st = cudaGetLastError();
    TORCH_CHECK(st == cudaSuccess, "dsa_sparse_attn_kernel launch failed: ", cudaGetErrorString(st));
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
            name="dsa_sparse_attn_ext_single_launch",
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
