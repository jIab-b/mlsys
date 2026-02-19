"""DSA index submission."""

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

constexpr int kEpilogueWarpBase = 0;
constexpr int kNumEpilogueWarps = 2;
constexpr int kProducerWarp = 2;
constexpr int kMmaWarp = 3;
constexpr int kNumWarps = 2 + kNumEpilogueWarps;
constexpr int kThreadsPerBlock = kNumWarps * 32;

constexpr int kMmaK = 32;
constexpr int kMmaIters = kHeadDim / kMmaK;  // 4
constexpr int kDesiredDynamicSmemBytes = 228 * 1024 - 2048;

// ---------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------
__device__ inline int smem_addr_from_base(const void* base_ptr, int base_addr, const void* ptr) {
    return base_addr + static_cast<int>(
        reinterpret_cast<const unsigned char*>(ptr) - reinterpret_cast<const unsigned char*>(base_ptr));
}

__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{ .reg .pred %%px; elect.sync _|%%px, %1; @%%px mov.s32 %0, 1; }"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
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

__device__ inline uint64_t make_desc_kmajor_noswizzle(int smem_addr) {
    const int sbo = 8 * 128;
    return desc_encode(static_cast<uint64_t>(smem_addr)) |
           (desc_encode(static_cast<uint64_t>(sbo)) << 32ULL) |
           (1ULL << 46ULL);
}

template <typename T>
__device__ inline T* smem_alloc(unsigned char*& ptr, int n) {
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    constexpr uintptr_t kAlign = alignof(T);
    p = (p + kAlign - 1) & ~(kAlign - 1);
    T* out = reinterpret_cast<T*>(p);
    ptr = reinterpret_cast<unsigned char*>(out + n);
    return out;
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

__device__ inline void ep_compute_all_stage_after_ld64(
    int epi_warp,
    int lane,
    int valid_tokens,
    int page_idx,
    const float* stage_scale,
    const float lane_regs[kStageTokens],
    const float* w_stage,
    float* cand_scores,
    int* cand_ids
) {
    const int tmem_row_base = epi_warp * 32;
    const float w = w_stage[tmem_row_base + lane];
    float warp_partial[kStageTokens];

    #pragma unroll
    for (int token = 0; token < kStageTokens; ++token) {
        float lane_val = 0.0f;
        if (token < valid_tokens) {
            lane_val = fmaxf(lane_regs[token] * stage_scale[token], 0.0f) * w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lane_val += __shfl_down_sync(0xFFFFFFFF, lane_val, off);
        }
        warp_partial[token] = lane_val;
    }

    if (epi_warp == 1 && lane == 0) {
        #pragma unroll
        for (int token = 0; token < kStageTokens; ++token) {
            if (token < valid_tokens) {
                cand_scores[token] = warp_partial[token];
            }
        }
    }
    asm volatile("bar.sync 1, 64;" ::: "memory");

    if (epi_warp == 0 && lane == 0) {
        #pragma unroll
        for (int token = 0; token < kStageTokens; ++token) {
            if (token < valid_tokens) {
                cand_scores[token] = warp_partial[token] + cand_scores[token];
                cand_ids[token] = page_idx * kPageSize + token;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Kernel: 3-stage warp-specialized pipeline
// ---------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreadsPerBlock) void dsa_topk_indexer_kernel(
    const __grid_constant__ CUtensorMap k_fp8_tmap,
    const __grid_constant__ CUtensorMap k_scale_tmap,
    const uint8_t* q_index_bytes,   // [B,64,128], FP8 E4M3
    const float* weights,           // [B,64]
    const int* seq_lens,            // [B]
    const int* block_table,         // [B,max_num_pages]
    int* topk_indices,              // [B,topk]
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
    int* out_b = topk_indices + static_cast<int64_t>(b) * topk;

    int seq_len = seq_lens[b];
    if (seq_len < 0) seq_len = 0;
    const int max_seq_by_pages = max_num_pages * kPageSize;
    if (seq_len > max_seq_by_pages) seq_len = max_seq_by_pages;

    for (int i = tid; i < topk; i += blockDim.x) {
        out_b[i] = -1;
    }
    __syncthreads();

    if (seq_len == 0 || topk <= 0) {
        return;
    }

    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const int smem_base_addr = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    unsigned char* smem_ptr = smem_raw;

    // Per-stage K data.
    uint8_t* k_stage_payload = smem_alloc<uint8_t>(smem_ptr, kNumStages * kStageTokens * kPayloadBytesPerToken);
    float* k_stage_scale = smem_alloc<float>(smem_ptr, kNumStages * kStageTokens);

    // Q + weights.
    uint8_t* q_stage = smem_alloc<uint8_t>(smem_ptr, kNumHeads * kHeadDim);
    float* w_stage = smem_alloc<float>(smem_ptr, kNumHeads);

    // Stage metadata.
    int* stage_page_idx = smem_alloc<int>(smem_ptr, kNumStages);
    int* stage_valid_tokens = smem_alloc<int>(smem_ptr, kNumStages);

    // Per-stage phase counters.
    int* tma_phase = smem_alloc<int>(smem_ptr, kNumStages);
    int* mma_phase = smem_alloc<int>(smem_ptr, kNumStages);
    int* epi_phase = smem_alloc<int>(smem_ptr, kNumStages);
    int* tmem_reuse_phase_mma = smem_alloc<int>(smem_ptr, kNumTmemSlots);

    // Three mbar arrays (tma_done, mma_done, epi_done).
    uint64_t* tma_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);
    uint64_t* mma_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);
    uint64_t* epi_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);
    uint64_t* tmem_reuse_mbar = smem_alloc<uint64_t>(smem_ptr, kNumTmemSlots);

    // Per-stage epilogue candidates (requested per-stage top-k buffer).
    float* stage_tile_scores = smem_alloc<float>(smem_ptr, kNumStages * kStageTokens);
    int* stage_tile_ids = smem_alloc<int>(smem_ptr, kNumStages * kStageTokens);

    // TMEM scratch.
    int* tmem_addr_scratch = smem_alloc<int>(smem_ptr, 1);

    const int k_stage_payload_addr = smem_addr_from_base(smem_raw, smem_base_addr, k_stage_payload);
    const int k_stage_scale_addr = smem_addr_from_base(smem_raw, smem_base_addr, k_stage_scale);
    const int q_stage_addr = smem_addr_from_base(smem_raw, smem_base_addr, q_stage);
    const int tma_mbar_addr = smem_addr_from_base(smem_raw, smem_base_addr, tma_mbar);
    const int mma_mbar_addr = smem_addr_from_base(smem_raw, smem_base_addr, mma_mbar);
    const int epi_mbar_addr = smem_addr_from_base(smem_raw, smem_base_addr, epi_mbar);
    const int tmem_reuse_mbar_addr = smem_addr_from_base(smem_raw, smem_base_addr, tmem_reuse_mbar);
    const int tmem_addr_scratch_addr = smem_addr_from_base(smem_raw, smem_base_addr, tmem_addr_scratch);

    // Init barriers and phases.
    if (warp_id == kProducerWarp && elect_sync()) {
        for (int i = 0; i < kNumStages; ++i) {
            mbarrier_init(tma_mbar_addr + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(mma_mbar_addr + i * static_cast<int>(sizeof(uint64_t)), 1);
            mbarrier_init(epi_mbar_addr + i * static_cast<int>(sizeof(uint64_t)), 1);
            tma_phase[i] = 0;
            mma_phase[i] = 0;
            epi_phase[i] = 0;
        }
        for (int i = 0; i < kNumTmemSlots; ++i) {
            mbarrier_init(tmem_reuse_mbar_addr + i * static_cast<int>(sizeof(uint64_t)), 1);
            tmem_reuse_phase_mma[i] = 0;
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
   // __syncthreads();

    // Stage Q + weights.
    #pragma unroll
    for (int idx = tid; idx < kNumHeads * kHeadDim; idx += blockDim.x) {
        const int64_t q_off = static_cast<int64_t>(b) * kNumHeads * kHeadDim + idx;
        q_stage[idx] = q_index_bytes[q_off];
    }
    #pragma unroll
    for (int idx = tid; idx < kNumHeads; idx += blockDim.x) {
        w_stage[idx] = weights_b[idx];
    }

    if (tid == 0) {
        tmem_addr_scratch[0] = 0;
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

            // Stage reuse requires prior epilogue completion on the same stage slot.
            if (tile_id >= kNumStages) {
                mbarrier_wait_parity(
                    epi_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)),
                    epi_phase[stage]);
                epi_phase[stage] ^= 1;
            }

            int page_idx, valid_tokens;
            prepare_stage_metadata_and_scale_tail(
                tile_id, stage, seq_len, max_num_pages, num_pages, block_table_b,
                stage_page_idx, stage_valid_tokens, k_stage_scale,
                page_idx, valid_tokens);
            


            if (valid_tokens > 0) {
                const int payload_dst = k_stage_payload_addr + stage * kStageTokens * kPayloadBytesPerToken;
                const int scale_dst = k_stage_scale_addr + stage * kStageTokens * static_cast<int>(sizeof(float));
                const int mbar_addr = tma_mbar_addr + stage * static_cast<int>(sizeof(uint64_t));
                constexpr int payload_bytes = kStageTokens * kPayloadBytesPerToken;
                constexpr int scale_bytes = kStageTokens * kScaleBytesPerToken;

                mbarrier_arrive_expect_tx(mbar_addr, payload_bytes + scale_bytes);
                tma_3d_gmem2smem(payload_dst, &k_fp8_tmap, 0, 0, page_idx, mbar_addr, 0ULL);
                tma_2d_gmem2smem(scale_dst, &k_scale_tmap, 0, page_idx, mbar_addr, 0ULL);
            } else {
                mbarrier_arrive(tma_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)));
            }

        }
    }

    // ---------------- MMA warp ----------------
    if (warp_id == kMmaWarp) {
        tcgen05_alloc(tmem_addr_scratch_addr, kNumTmemSlots * kStageTokens);
    }
    if (warp_id == kMmaWarp && elect_sync()) {
        for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
            const int stage = tile_id % kNumStages;
            const int tmem_slot = tile_id % kNumTmemSlots;

 
            mbarrier_wait_parity(
                tma_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)),
                tma_phase[stage]);
            tma_phase[stage] ^= 1;

            if (tile_id >= kNumTmemSlots) {
                mbarrier_wait_parity(
                    tmem_reuse_mbar_addr + tmem_slot * static_cast<int>(sizeof(uint64_t)),
                    tmem_reuse_phase_mma[tmem_slot]);
                tmem_reuse_phase_mma[tmem_slot] ^= 1;
            }
            
        

            const int valid_tokens = stage_valid_tokens[stage];
            if (valid_tokens > 0) {
                const int q_addr = q_stage_addr;
                const int k_addr = k_stage_payload_addr + stage * kStageTokens * kPayloadBytesPerToken;

                uint64_t q_desc = make_desc_kmajor_noswizzle(q_addr);
                uint64_t k_desc = make_desc_kmajor_swizzle_128b(k_addr);
                const uint32_t tmem_d = static_cast<uint32_t>(tmem_slot * kStageTokens);

                for (int ki = 0; ki < kMmaIters; ++ki) {
                    tcgen05_mma_f8f6f4(tmem_d, q_desc, k_desc, kIdesc, (ki > 0));
                    q_desc += (kMmaK >> 4);
                    k_desc += (kMmaK >> 4);
                }

                tcgen05_commit(mma_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)));
            } else {
                mbarrier_arrive(mma_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)));
            }
        }
    }

    // ---------------- Epilogue warps ----------------
    if (warp_id >= kEpilogueWarpBase && warp_id < kEpilogueWarpBase + kNumEpilogueWarps) {
        const int epi_warp = warp_id - kEpilogueWarpBase;

        for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
            const int stage = tile_id % kNumStages;
            const int tmem_slot = tile_id % kNumTmemSlots;

            if (epi_warp == 0 && lane == 0) {
                mbarrier_wait_parity(
                    mma_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)),
                    mma_phase[stage]);
                mma_phase[stage] ^= 1;
            }

            // Both EP warps synchronize with warp0's mma_mbar wait.
            asm volatile("bar.sync 1, 64;" ::: "memory");
            tcgen05_fence_after_thread_sync();

            const int valid_tokens = stage_valid_tokens[stage];
            if (valid_tokens > 0) {
                const int tmem_row_base = epi_warp * 32;
                const int tmem_col_base = tmem_slot * kStageTokens;
                float lane_regs[kStageTokens];
                tcgen05_ld_32x32b_64(tmem_row_base, tmem_col_base, lane_regs);
                tcgen05_wait_ld();

                const int page_idx = stage_page_idx[stage];
                const float* stage_scale = k_stage_scale + stage * kStageTokens;
                float* cand_scores = stage_tile_scores + stage * kStageTokens;
                int* cand_ids = stage_tile_ids + stage * kStageTokens;
                ep_compute_all_stage_after_ld64(
                    epi_warp,
                    lane,
                    valid_tokens,
                    page_idx,
                    stage_scale,
                    lane_regs,
                    w_stage,
                    cand_scores,
                    cand_ids
                );
            }
            if (epi_warp == 0 && lane == 0) {
                mbarrier_arrive(tmem_reuse_mbar_addr + tmem_slot * static_cast<int>(sizeof(uint64_t)));
                mbarrier_arrive(epi_mbar_addr + stage * static_cast<int>(sizeof(uint64_t)));
            }
        }
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
    torch::Tensor topk_indices
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

    const int8_t* k_ptr = reinterpret_cast<const int8_t*>(k_index_cache_fp8.data_ptr());
    CUtensorMap k_fp8_tmap = make_k_fp8_tmap(k_ptr, num_pages);
    CUtensorMap k_scale_tmap = make_k_scale_tmap(k_ptr, num_pages);

    const int blocks = batch_size;
    dsa_topk_indexer_kernel<<<blocks, kThreadsPerBlock, kDesiredDynamicSmemBytes>>>(
        k_fp8_tmap,
        k_scale_tmap,
        reinterpret_cast<const uint8_t*>(q_index_fp8.data_ptr()),
        reinterpret_cast<const float*>(weights.data_ptr()),
        reinterpret_cast<const int*>(seq_lens.data_ptr()),
        reinterpret_cast<const int*>(block_table.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
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
    torch::Tensor topk_indices);
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_indexer_ext",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            verbose=True,
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
) -> torch.Tensor:
    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        topk_indices,
    )
    return topk_indices


def custom_kernel(data: input_t) -> output_t:
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    batch = int(q_index_fp8.shape[0])
    topk_indices = torch.empty((batch, 2048), dtype=torch.int32, device=q_index_fp8.device)
    out = _dsa_topk_indexer(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
    return (out,)
