"""Single-file DSA index submission with inlined CUDA source."""

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
#include <cfloat>
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
constexpr int kNumStages = 10;

constexpr int kProducerWarp = 0;
constexpr int kMmaWarp = 1;
constexpr int kEpilogueWarpBase = 2;
constexpr int kNumEpilogueWarps = 2;
constexpr int kNumWarps = 2 + kNumEpilogueWarps;
constexpr int kThreadsPerBlock = kNumWarps * 32;

constexpr int kMmaK = 32;
constexpr int kMmaIters = kHeadDim / kMmaK;  // 4
constexpr int kDesiredDynamicSmemBytes = 228 * 1024;

// ---------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------
__device__ inline uint32_t lane_id() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
    return lane;
}

__device__ inline uint32_t cvta_to_shared_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ inline bool elect_lane0() {
    return lane_id() == 0;
}

__device__ inline constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_desc_kmajor_128b(int smem_addr) {
    const int sbo = 8 * 128;
    return desc_encode(static_cast<uint64_t>(smem_addr)) |
           (desc_encode(static_cast<uint64_t>(sbo)) << 32ULL) |
           (1ULL << 46ULL) |
           (2ULL << 61ULL);
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
    const uint32_t ticks = 0x989680;
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "L_WAIT_%=: \\n\\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\\n\\t"
        "@p bra.uni L_DONE_%=;\\n\\t"
        "bra.uni L_WAIT_%=;\\n\\t"
        "L_DONE_%=: \\n\\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
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
    if (!elect_lane0()) {
        return;
    }
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

__device__ inline float tcgen05_ld_32x32b_1(int row, int col) {
    float val;
    int addr = (row << 16) | col;
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=f"(val) : "r"(addr));
    return val;
}

__device__ inline uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

__device__ inline void insert_topk_desc(float score, int idx, float* scores, int* ids, int& count, int k) {
    if (k <= 0 || idx < 0) {
        return;
    }

    if (count < k) {
        int pos = count;
        while (pos > 0 && score > scores[pos - 1]) {
            scores[pos] = scores[pos - 1];
            ids[pos] = ids[pos - 1];
            --pos;
        }
        scores[pos] = score;
        ids[pos] = idx;
        ++count;
        return;
    }

    if (score <= scores[k - 1]) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && score > scores[pos - 1]) {
        scores[pos] = scores[pos - 1];
        ids[pos] = ids[pos - 1];
        --pos;
    }
    scores[pos] = score;
    ids[pos] = idx;
}

// ---------------------------------------------------------------------------------
// Kernel: 3-stage warp-specialized pipeline
// ---------------------------------------------------------------------------------
__global__ void dsa_topk_indexer_kernel(
    const uint8_t* q_index_bytes,   // [B,64,128], FP8 E4M3
    const int8_t* k_index_cache,    // [num_pages,64,1,132], deep_gemm packed
    const float* weights,           // [B,64]
    const int* seq_lens,            // [B]
    const int* block_table,         // [B,max_num_pages]
    const void* k_fp8_tmap,         // payload tensor map
    const void* k_scale_tmap,       // scale tensor map
    int* topk_indices,              // [B,topk]
    int batch_size,
    int num_pages,
    int max_num_pages,
    int topk
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

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

    const int actual_topk = (topk < seq_len) ? topk : seq_len;

    extern __shared__ __align__(1024) unsigned char smem_raw[];
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

    // Three mbar arrays (tma_done, mma_done, epi_done).
    uint64_t* tma_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);
    uint64_t* mma_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);
    uint64_t* epi_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);

    // Per-stage epilogue candidates (requested per-stage top-k buffer).
    float* stage_tile_scores = smem_alloc<float>(smem_ptr, kNumStages * kStageTokens);
    int* stage_tile_ids = smem_alloc<int>(smem_ptr, kNumStages * kStageTokens);

    // Global rolling top-k.
    float* topk_scores = smem_alloc<float>(smem_ptr, topk);
    int* topk_ids = smem_alloc<int>(smem_ptr, topk);
    int* topk_count_ptr = smem_alloc<int>(smem_ptr, 1);

    // TMEM scratch.
    int* tmem_addr_scratch = smem_alloc<int>(smem_ptr, 1);

    // Init barriers and phases.
    if (warp_id == kProducerWarp && lane < kNumStages) {
        mbarrier_init(cvta_to_shared_u32(&tma_mbar[lane]), 1);
        mbarrier_init(cvta_to_shared_u32(&mma_mbar[lane]), 1);
        mbarrier_init(cvta_to_shared_u32(&epi_mbar[lane]), 1);
        tma_phase[lane] = 0;
        mma_phase[lane] = 0;
        epi_phase[lane] = 0;
    }
    if (warp_id == kProducerWarp && lane == 0) {
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    // Stage Q + weights.
    for (int idx = tid; idx < kNumHeads * kHeadDim; idx += blockDim.x) {
        const int64_t q_off = static_cast<int64_t>(b) * kNumHeads * kHeadDim + idx;
        q_stage[idx] = q_index_bytes[q_off];
    }
    for (int idx = tid; idx < kNumHeads; idx += blockDim.x) {
        w_stage[idx] = weights_b[idx];
    }

    // Init global rolling top-k.
    for (int i = tid; i < topk; i += blockDim.x) {
        topk_scores[i] = -FLT_MAX;
        topk_ids[i] = -1;
    }
    if (tid == 0) {
        topk_count_ptr[0] = 0;
        tmem_addr_scratch[0] = 0;
    }
    __syncthreads();

    if (warp_id == kMmaWarp) {
        tcgen05_alloc(cvta_to_shared_u32(tmem_addr_scratch), kStageTokens);
    }
    __syncthreads();

    constexpr uint32_t kIdesc = (0U << 7U)    // atype = E4M3
                              | (0U << 10U)   // btype = E4M3
                              | (1U << 4U)    // dtype = F32
                              | ((uint32_t)(kStageTokens >> 3U) << 17U)
                              | ((uint32_t)(kNumHeads >> 4U) << 24U);

    const int num_tiles = (seq_len + kStageTokens - 1) / kStageTokens;

    // Pipeline schedule:
    // iter: producer(tile=iter), mma(tile=iter-1), epilogue(tile=iter-2).
    for (int iter = 0; iter < num_tiles + 2; ++iter) {
        // ---------------- Producer warp ----------------
        if (warp_id == kProducerWarp) {
            const int tile_id = iter;
            if (tile_id < num_tiles) {
                const int stage = tile_id % kNumStages;

                // Stage reuse requires prior epilogue completion on the same stage slot.
                if (tile_id >= kNumStages && lane == 0) {
                    mbarrier_wait_parity(cvta_to_shared_u32(&epi_mbar[stage]), epi_phase[stage]);
                    epi_phase[stage] ^= 1;
                }
                __syncwarp();

                const int tile_seq_start = tile_id * kStageTokens;
                const int remain = seq_len - tile_seq_start;
                int valid_tokens = (remain > 0) ? ((remain < kStageTokens) ? remain : kStageTokens) : 0;

                int page_idx = -1;
                if (tile_id >= 0 && tile_id < max_num_pages) {
                    page_idx = block_table_b[tile_id];
                }
                if (!(page_idx >= 0 && page_idx < num_pages)) {
                    valid_tokens = 0;
                }

                if (lane == 0) {
                    stage_page_idx[stage] = page_idx;
                    stage_valid_tokens[stage] = valid_tokens;
                }
                __syncwarp();

                // Zero out scale padding for invalid tokens.
                float* stage_scale = k_stage_scale + stage * kStageTokens;
                for (int tok = lane + valid_tokens; tok < kStageTokens; tok += 32) {
                    stage_scale[tok] = 0.0f;
                }
                __syncwarp();

                // TMA both payload and scales to the same mbarrier.
                if (lane == 0) {
                    if (valid_tokens > 0) {
                        const int payload_dst = cvta_to_shared_u32(
                            k_stage_payload + stage * kStageTokens * kPayloadBytesPerToken);
                        const int scale_dst = cvta_to_shared_u32(stage_scale);
                        const int mbar_addr = cvta_to_shared_u32(&tma_mbar[stage]);
                        constexpr int payload_bytes = kStageTokens * kPayloadBytesPerToken;
                        constexpr int scale_bytes = kStageTokens * kScaleBytesPerToken;
                        mbarrier_arrive_expect_tx(mbar_addr, payload_bytes + scale_bytes);
                        tma_3d_gmem2smem(payload_dst, k_fp8_tmap, 0, 0, page_idx, mbar_addr, 0ULL);
                        tma_2d_gmem2smem(scale_dst, k_scale_tmap, 0, page_idx, mbar_addr, 0ULL);
                    } else {
                        mbarrier_arrive(cvta_to_shared_u32(&tma_mbar[stage]));
                    }
                }
            }
        }

        // ---------------- MMA warp ----------------
        if (warp_id == kMmaWarp) {
            const int tile_id = iter - 1;
            if (tile_id >= 0 && tile_id < num_tiles) {
                const int stage = tile_id % kNumStages;

                if (lane == 0) {
                    mbarrier_wait_parity(cvta_to_shared_u32(&tma_mbar[stage]), tma_phase[stage]);
                    tma_phase[stage] ^= 1;
                }
                __syncwarp();

                const int valid_tokens = stage_valid_tokens[stage];
                if (valid_tokens > 0 && lane == 0) {
                    const int q_addr = cvta_to_shared_u32(q_stage);
                    const int k_addr = cvta_to_shared_u32(
                        k_stage_payload + stage * kStageTokens * kPayloadBytesPerToken);

                    uint64_t a_desc = make_desc_kmajor_128b(q_addr);
                    uint64_t b_desc = make_desc_kmajor_128b(k_addr);

                    for (int ki = 0; ki < kMmaIters; ++ki) {
                        tcgen05_mma_f8f6f4(0, a_desc, b_desc, kIdesc, (ki > 0));
                        a_desc += (kMmaK >> 4);
                        b_desc += (kMmaK >> 4);
                    }

                    tcgen05_commit(cvta_to_shared_u32(&mma_mbar[stage]));
                } else if (lane == 0) {
                    mbarrier_arrive(cvta_to_shared_u32(&mma_mbar[stage]));
                }
            }
        }

        // ---------------- Epilogue warps ----------------
        if (warp_id >= kEpilogueWarpBase && warp_id < kEpilogueWarpBase + kNumEpilogueWarps) {
            const int epi_warp = warp_id - kEpilogueWarpBase;
            const int tile_id = iter - 2;

            if (tile_id >= 0 && tile_id < num_tiles) {
                const int stage = tile_id % kNumStages;

                if (epi_warp == 0 && lane == 0) {
                    mbarrier_wait_parity(cvta_to_shared_u32(&mma_mbar[stage]), mma_phase[stage]);
                    mma_phase[stage] ^= 1;
                }

                // Epilogue warps synchronize before/after TMEM reads and candidate writes.
                asm volatile("bar.sync 1, 64;" ::: "memory");

                tcgen05_fence_after_thread_sync();

                const int valid_tokens = stage_valid_tokens[stage];
                const int page_idx = stage_page_idx[stage];
                const float* stage_scale = k_stage_scale + stage * kStageTokens;
                float* cand_scores = stage_tile_scores + stage * kStageTokens;
                int* cand_ids = stage_tile_ids + stage * kStageTokens;

                // epi_warp 0 reads TMEM rows 0-31 (heads 0..31)
                // epi_warp 1 reads TMEM rows 32-63 (heads 32..63)
                const int tmem_row_base = epi_warp * 32;

                for (int t = 0; t < valid_tokens; ++t) {
                    // Each warp loads its 32 heads for token t
                    float d = tcgen05_ld_32x32b_1(tmem_row_base, t);
                    tcgen05_wait_ld();

                    const float scale = stage_scale[t];
                    const float v = d * scale;

                    // weight for this lane's head
                    const float w = w_stage[tmem_row_base + lane];

                    // ReLU + weighted
                    float my_sum = (v > 0.0f) ? (v * w) : 0.0f;

                    // Warp reduction: partial sum over 32 heads
                    for (int off = 16; off > 0; off >>= 1) {
                        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, off);
                    }

                    // epi_warp 1 writes partial sum to smem for epi_warp 0 to read
                    if (epi_warp == 1 && lane == 0) {
                        cand_scores[t] = my_sum;
                    }
                    asm volatile("bar.sync 1, 64;" ::: "memory");

                    if (epi_warp == 0 && lane == 0) {
                        float total = my_sum + cand_scores[t];
                        cand_scores[t] = total;
                        cand_ids[t] = page_idx * kPageSize + t;
                    }
                    asm volatile("bar.sync 1, 64;" ::: "memory");
                }

                if (epi_warp == 0 && lane == 0) {
                    int topk_count = topk_count_ptr[0];
                    for (int t = 0; t < valid_tokens; ++t) {
                        insert_topk_desc(cand_scores[t], cand_ids[t], topk_scores, topk_ids, topk_count, actual_topk);
                    }
                    topk_count_ptr[0] = topk_count;
                    mbarrier_arrive(cvta_to_shared_u32(&epi_mbar[stage]));
                }
            }
        }
    }

    __syncthreads();

    if (warp_id == kEpilogueWarpBase) {
        const int topk_count = topk_count_ptr[0];
        for (int i = lane; i < actual_topk; i += 32) {
            out_b[i] = (i < topk_count) ? topk_ids[i] : -1;
        }
    }

    __syncthreads();
    if (warp_id == kMmaWarp) {
        tcgen05_dealloc(0, kStageTokens);
    }
}

// ---------------------------------------------------------------------------------
// Host-side tensor-map encode helpers
// ---------------------------------------------------------------------------------
static void* g_k_fp8_tmap_dev = nullptr;
static void* g_k_scale_tmap_dev = nullptr;

static inline size_t align_up_size(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

static size_t required_dynamic_smem_bytes(int topk) {
    size_t off = 0;

    off = align_up_size(off, alignof(uint8_t));
    off += static_cast<size_t>(kNumStages) * kStageTokens * kPayloadBytesPerToken;

    off = align_up_size(off, alignof(float));
    off += static_cast<size_t>(kNumStages) * kStageTokens * sizeof(float);

    off = align_up_size(off, alignof(uint8_t));
    off += static_cast<size_t>(kNumHeads) * kHeadDim;

    off = align_up_size(off, alignof(float));
    off += static_cast<size_t>(kNumHeads) * sizeof(float);

    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(kNumStages) * sizeof(int);  // stage_page_idx
    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(kNumStages) * sizeof(int);  // stage_valid_tokens
    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(kNumStages) * sizeof(int);  // tma_phase
    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(kNumStages) * sizeof(int);  // mma_phase
    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(kNumStages) * sizeof(int);  // epi_phase

    off = align_up_size(off, alignof(uint64_t));
    off += static_cast<size_t>(kNumStages) * sizeof(uint64_t);  // tma_mbar
    off = align_up_size(off, alignof(uint64_t));
    off += static_cast<size_t>(kNumStages) * sizeof(uint64_t);  // mma_mbar
    off = align_up_size(off, alignof(uint64_t));
    off += static_cast<size_t>(kNumStages) * sizeof(uint64_t);  // epi_mbar

    off = align_up_size(off, alignof(float));
    off += static_cast<size_t>(kNumStages) * kStageTokens * sizeof(float);  // stage_tile_scores
    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(kNumStages) * kStageTokens * sizeof(int);    // stage_tile_ids

    off = align_up_size(off, alignof(float));
    off += static_cast<size_t>(topk) * sizeof(float);  // topk_scores
    off = align_up_size(off, alignof(int));
    off += static_cast<size_t>(topk) * sizeof(int);    // topk_ids
    off = align_up_size(off, alignof(int));
    off += sizeof(int);                                 // topk_count_ptr
    off = align_up_size(off, alignof(int));
    off += sizeof(int);                                 // tmem_addr_scratch

    return off;
}

static void ensure_k_fp8_tmap_on_device(const int8_t* k_index_cache_ptr, int num_pages) {
    if (g_k_fp8_tmap_dev == nullptr) {
        cudaError_t alloc_st = cudaMalloc(&g_k_fp8_tmap_dev, sizeof(CUtensorMap));
        TORCH_CHECK(alloc_st == cudaSuccess, "cudaMalloc(CUtensorMap) failed");
    }

    CUtensorMap k_fp8_tmap{};
    cuInit(0);

    constexpr cuuint32_t rank = 3;
    const cuuint64_t globalDim[rank] = {
        static_cast<cuuint64_t>(kPayloadBytesPerToken),
        static_cast<cuuint64_t>(kPageSize),
        static_cast<cuuint64_t>(num_pages),
    };
    const cuuint64_t globalStrides[rank - 1] = {
        static_cast<cuuint64_t>(kPayloadBytesPerToken),
        static_cast<cuuint64_t>(kPageBytes),
    };
    const cuuint32_t boxDim[rank] = {
        static_cast<cuuint32_t>(kPayloadBytesPerToken),
        static_cast<cuuint32_t>(kPageSize),
        1U,
    };
    const cuuint32_t elementStrides[rank] = {1U, 1U, 1U};

    const CUresult st = cuTensorMapEncodeTiled(
        &k_fp8_tmap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,
        const_cast<int8_t*>(k_index_cache_ptr),
        globalDim,
        globalStrides,
        boxDim,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(st == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for payload");

    cudaError_t cp_st = cudaMemcpy(g_k_fp8_tmap_dev, &k_fp8_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    TORCH_CHECK(cp_st == cudaSuccess, "cudaMemcpy(payload CUtensorMap) failed");

    // Scale tensor map: 1D over the flat scale region.
    // Scales are at offset kPackedFp8Bytes within each page, contiguous kPageSize * 4 bytes.
    // We treat the entire cache as a flat byte array and index by page_idx * kPageSize * 4.
    if (g_k_scale_tmap_dev == nullptr) {
        cudaError_t alloc_st2 = cudaMalloc(&g_k_scale_tmap_dev, sizeof(CUtensorMap));
        TORCH_CHECK(alloc_st2 == cudaSuccess, "cudaMalloc(scale CUtensorMap) failed");
    }

    // Build a 1D tensor map over the scale bytes.
    // Base pointer: k_index_cache_ptr + kPackedFp8Bytes (start of scales in page 0).
    // globalDim: num_pages * kPageSize * 4 total scale bytes across all pages.
    // BUT scales are NOT contiguous across pages — there's a gap of kPackedFp8Bytes between pages.
    // So use a 2D map: dim0 = scale bytes per page (256), dim1 = num_pages.
    CUtensorMap k_scale_tmap{};
    {
        constexpr cuuint32_t srank = 1;
        const cuuint64_t sglobalDim[srank] = {
            static_cast<cuuint64_t>(num_pages) * kPageSize * kScaleBytesPerToken,
        };
        const cuuint64_t sglobalStrides[1] = {};  // unused for rank 1
        const cuuint32_t sboxDim[srank] = {
            static_cast<cuuint32_t>(kPageSize * kScaleBytesPerToken),
        };
        const cuuint32_t selementStrides[srank] = {1U};

        // Base pointer is the start of the scale region (offset kPackedFp8Bytes from page 0).
        // Since scales are at a fixed offset per page and pages are kPageBytes apart,
        // and kPageBytes = kPackedFp8Bytes + kPageSize*kScaleBytesPerToken = 8192 + 256 = 8448,
        // the scales ARE contiguous across pages in the flat layout:
        //   page0_fp8(8192) | page0_scale(256) | page1_fp8(8192) | page1_scale(256) | ...
        // So they are NOT contiguous. We need a 2D map.
        // Actually, let's use 2D: dim0 = 256 (scale bytes per page), dim1 = num_pages.
        (void)sglobalDim; (void)sglobalStrides; (void)sboxDim; (void)selementStrides;
    }
    {
        constexpr cuuint32_t srank = 2;
        const uint8_t* scale_base = reinterpret_cast<const uint8_t*>(k_index_cache_ptr) + kPackedFp8Bytes;
        const cuuint64_t sglobalDim[srank] = {
            static_cast<cuuint64_t>(kPageSize * kScaleBytesPerToken),  // 256
            static_cast<cuuint64_t>(num_pages),
        };
        const cuuint64_t sglobalStrides[srank - 1] = {
            static_cast<cuuint64_t>(kPageBytes),  // stride between pages = 8448
        };
        const cuuint32_t sboxDim[srank] = {
            static_cast<cuuint32_t>(kPageSize * kScaleBytesPerToken),  // 256
            1U,
        };
        const cuuint32_t selementStrides[srank] = {1U, 1U};

        const CUresult st2 = cuTensorMapEncodeTiled(
            &k_scale_tmap,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            srank,
            const_cast<uint8_t*>(scale_base),
            sglobalDim,
            sglobalStrides,
            sboxDim,
            selementStrides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(st2 == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for scales");
    }

    cudaError_t cp_st2 = cudaMemcpy(g_k_scale_tmap_dev, &k_scale_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    TORCH_CHECK(cp_st2 == cudaSuccess, "cudaMemcpy(scale CUtensorMap) failed");
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

    ensure_k_fp8_tmap_on_device(reinterpret_cast<const int8_t*>(k_index_cache_fp8.data_ptr()), num_pages);

    int device = -1;
    cudaGetDevice(&device);

    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    const size_t smem_need = required_dynamic_smem_bytes(topk);
    TORCH_CHECK(smem_need <= static_cast<size_t>(INT_MAX), "required smem does not fit int");

    int dynamic_smem_bytes = kDesiredDynamicSmemBytes;
    const int smem_need_i = static_cast<int>(smem_need);
    if (dynamic_smem_bytes < smem_need_i) dynamic_smem_bytes = smem_need_i;
    if (max_optin_smem > 0 && dynamic_smem_bytes > max_optin_smem) dynamic_smem_bytes = max_optin_smem;
    TORCH_CHECK(dynamic_smem_bytes >= smem_need_i,
                "insufficient dynamic shared memory: need ", smem_need_i, ", have ", dynamic_smem_bytes);

    cudaFuncSetAttribute(
        dsa_topk_indexer_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem_bytes
    );
    cudaFuncSetAttribute(
        dsa_topk_indexer_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        100
    );

    const int blocks = batch_size;
    dsa_topk_indexer_kernel<<<blocks, kThreadsPerBlock, dynamic_smem_bytes>>>(
        reinterpret_cast<const uint8_t*>(q_index_fp8.data_ptr()),
        reinterpret_cast<const int8_t*>(k_index_cache_fp8.data_ptr()),
        reinterpret_cast<const float*>(weights.data_ptr()),
        reinterpret_cast<const int*>(seq_lens.data_ptr()),
        reinterpret_cast<const int*>(block_table.data_ptr()),
        g_k_fp8_tmap_dev,
        g_k_scale_tmap_dev,
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
            name="dsa_topk_indexer_singlefile_ext",
            cpp_sources=cpp_decl_src,
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            verbose=True,
            extra_cuda_cflags=[
                "-O0",
                "-gencode=arch=compute_100a,code=sm_100a",
                "--threads=4",
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
