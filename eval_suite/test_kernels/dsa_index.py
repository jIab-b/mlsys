"""Single-file DSA index submission with inlined CUDA source."""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


_module = None
_spill_workspace = {}


cuda_src = """
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

// ---------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------
constexpr int kNumHeads = 64;
constexpr int kHeadDim = 128;
constexpr int kPageSize = 64;
constexpr int kTopKDefault = 2048;

constexpr int kPayloadBytesPerToken = 128;
constexpr int kScaleBytesPerToken = 4;
constexpr int kRowBytes = 132;
constexpr int kPageBytes = kPageSize * kRowBytes;      // 8448
constexpr int kPackedFp8Bytes = kPageSize * kHeadDim;  // 8192

constexpr int kStageTokens = 64;
constexpr int kNumStages = 10;
constexpr int kProducerWarp = 0;
constexpr int kConsumerWarp = 1;
constexpr int kThreadsPerBlock = 64;  // 2 warps exactly
constexpr int kDesiredDynamicSmemBytes = 228 * 1024;

// Safety toggles for bring-up:
// Keep async PTX paths off until descriptor/idesc/barrier protocol is fully validated.
constexpr bool kEnableAsyncTma = false;
constexpr bool kEnableTcgenMma = false;

// ---------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------
__device__ inline uint32_t lane_id() {
    uint32_t lane;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
    return lane;
}

__device__ inline uint32_t cvta_to_shared_u32(const void* p) {
    uint32_t out;
    asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(out) : "l"(p));
    return out;
}

__device__ inline bool elect_lane0() {
    return lane_id() == 0;
}

__device__ inline constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ inline uint64_t make_desc_kmajor_128b(int smem_addr) {
    // Shared-memory descriptor, K-major, 128B swizzle.
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
    if (!elect_lane0()) {
        return;
    }
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;"
        :: "r"(dst_smem_addr), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
        : "memory");
}

__device__ inline void tcgen05_mma_f16_ss(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t idesc,
    int accumulate
) {
    if (!elect_lane0()) {
        return;
    }
    uint32_t mask[4] = {0, 0, 0, 0};
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %4, 0;\\n\\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\\n\\t"
        "}"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate),
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

__device__ inline void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ inline float fp8_byte_to_float(uint8_t x) {
    __nv_fp8_e4m3 v;
    v.__x = x;
    return static_cast<float>(v);
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
// Kernel: warp-specialized (2 warps)
// ---------------------------------------------------------------------------------
__global__ void dsa_topk_indexer_kernel(
    const uint8_t* q_index_bytes,   // [B,64,128], elem_size in {1,2,4}
    const int8_t* k_index_cache,    // [num_pages,64,1,132], deep_gemm packed
    const float* weights,           // [B,64]
    const int* seq_lens,            // [B]
    const int* block_table,         // [B,max_num_pages]
    const void* k_fp8_tmap,         // CUtensorMap in global memory
    float* spill_scores,            // [B,spill_stride]
    int* topk_indices,              // [B,topk]
    int batch_size,
    int num_pages,
    int max_num_pages,
    int spill_stride,
    int topk,
    int q_elem_size,
    int dynamic_smem_bytes
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    if (b >= batch_size) {
        return;
    }

    // Exactly two warps by contract.
    if (warp_id > kConsumerWarp) {
        return;
    }

    const int* block_table_b = block_table + static_cast<int64_t>(b) * max_num_pages;
    const float* weights_b = weights + static_cast<int64_t>(b) * kNumHeads;
    int* out_b = topk_indices + static_cast<int64_t>(b) * topk;

    const int max_seq_by_pages = max_num_pages * kPageSize;
    const int max_seq_by_spill = spill_stride;
    int seq_len = seq_lens[b];
    if (seq_len < 0) seq_len = 0;
    if (seq_len > max_seq_by_pages) seq_len = max_seq_by_pages;
    if (seq_len > max_seq_by_spill) seq_len = max_seq_by_spill;

    for (int i = tid; i < topk; i += blockDim.x) {
        out_b[i] = -1;
    }
    __syncthreads();

    if (seq_len == 0 || topk <= 0) {
        return;
    }

    extern __shared__ unsigned char smem_raw[];
    unsigned char* smem_ptr = smem_raw;

    // Producer-owned staged K buffers.
    uint8_t* k_stage_payload = smem_alloc<uint8_t>(smem_ptr, kNumStages * kStageTokens * kPayloadBytesPerToken);
    float* k_stage_scale = smem_alloc<float>(smem_ptr, kNumStages * kStageTokens);

    // Shared Q/weights and metadata.
    uint8_t* q_stage = smem_alloc<uint8_t>(smem_ptr, kNumHeads * kHeadDim);
    float* w_stage = smem_alloc<float>(smem_ptr, kNumHeads);

    int* stage_page_idx = smem_alloc<int>(smem_ptr, kNumStages);
    int* stage_valid_tokens = smem_alloc<int>(smem_ptr, kNumStages);
    int* stage_phase = smem_alloc<int>(smem_ptr, kNumStages);
    uint64_t* stage_mbar = smem_alloc<uint64_t>(smem_ptr, kNumStages);

    float* topk_scores = smem_alloc<float>(smem_ptr, topk);
    int* topk_ids = smem_alloc<int>(smem_ptr, topk);

    const int used_bytes = static_cast<int>(smem_ptr - smem_raw);
    const int seq_cap_smem = (dynamic_smem_bytes > used_bytes)
                                 ? (dynamic_smem_bytes - used_bytes) / static_cast<int>(sizeof(float))
                                 : 0;
    float* final_scores_smem = reinterpret_cast<float*>(smem_ptr);
    const bool use_smem_scores = (seq_len <= seq_cap_smem);
    float* spill_b = spill_scores + static_cast<int64_t>(b) * spill_stride;

    // Initialize mbarriers + phase.
    if (kEnableAsyncTma && warp_id == kProducerWarp && lane < kNumStages) {
        const int mbar_addr = cvta_to_shared_u32(&stage_mbar[lane]);
        mbarrier_init(mbar_addr, 1);
        stage_phase[lane] = 0;
    }
    __syncthreads();

    // Stage Q and weights once.
    if (warp_id == kConsumerWarp) {
        for (int idx = lane; idx < kNumHeads; idx += 32) {
            w_stage[idx] = weights_b[idx];
        }
        if (q_elem_size == 1) {
            for (int idx = lane; idx < kNumHeads * kHeadDim; idx += 32) {
                const int64_t q_off = static_cast<int64_t>(b) * kNumHeads * kHeadDim + idx;
                q_stage[idx] = q_index_bytes[q_off];
            }
        }
    }
    __syncthreads();

    const uint8_t* k_bytes = reinterpret_cast<const uint8_t*>(k_index_cache);
    const int num_tiles = (seq_len + kStageTokens - 1) / kStageTokens;

    for (int tile_base = 0; tile_base < num_tiles; tile_base += kNumStages) {
        // ---------------- Producer warp: all loads ----------------
        if (warp_id == kProducerWarp) {
            for (int s = 0; s < kNumStages; ++s) {
                const int tile_id = tile_base + s;
                const int tile_seq_start = tile_id * kStageTokens;
                const int remain = seq_len - tile_seq_start;
                const int valid_tokens = (remain > 0) ? ((remain < kStageTokens) ? remain : kStageTokens) : 0;

                if (lane == 0) {
                    stage_valid_tokens[s] = valid_tokens;
                    int page_idx = -1;
                    if (tile_id >= 0 && tile_id < max_num_pages) {
                        page_idx = block_table_b[tile_id];
                    }
                    stage_page_idx[s] = page_idx;
                }
                __syncwarp();

                const int page_idx = stage_page_idx[s];
                uint8_t* stage_payload = k_stage_payload + s * kStageTokens * kPayloadBytesPerToken;
                float* stage_scale = k_stage_scale + s * kStageTokens;

                // 1) Issue TMA payload load (best-effort tensor-map path).
                if (kEnableAsyncTma && k_fp8_tmap != nullptr && page_idx >= 0 && page_idx < num_pages && lane == 0) {
                    const int dst = cvta_to_shared_u32(stage_payload);
                    const int mbar_addr = cvta_to_shared_u32(&stage_mbar[s]);
                    mbarrier_arrive_expect_tx(mbar_addr, kStageTokens * kPayloadBytesPerToken);
                    tma_3d_gmem2smem(dst, k_fp8_tmap, 0, 0, page_idx, mbar_addr, 0ULL);
                }

                // 2) Manual fallback copy for payload + scales (ground-truth data path).
                if (page_idx >= 0 && page_idx < num_pages && valid_tokens > 0) {
                    const uint8_t* page_ptr = k_bytes + static_cast<int64_t>(page_idx) * kPageBytes;

                    for (int i = lane; i < valid_tokens * kPayloadBytesPerToken; i += 32) {
                        const int tok = i / kPayloadBytesPerToken;
                        const int d = i % kPayloadBytesPerToken;
                        stage_payload[i] = page_ptr[tok * kPayloadBytesPerToken + d];
                    }

                    for (int tok = lane; tok < valid_tokens; tok += 32) {
                        const int scale_off = kPackedFp8Bytes + tok * kScaleBytesPerToken;
                        const uint32_t bits = read_u32_le(page_ptr + scale_off);
                        stage_scale[tok] = __uint_as_float(bits);
                    }
                }

                for (int i = lane + valid_tokens * kPayloadBytesPerToken;
                     i < kStageTokens * kPayloadBytesPerToken;
                     i += 32) {
                    stage_payload[i] = 0;
                }
                for (int tok = lane + valid_tokens; tok < kStageTokens; tok += 32) {
                    stage_scale[tok] = 0.0f;
                }

                __syncwarp();
            }
        }

        __syncthreads();

        // ---------------- Consumer warp: all compute ----------------
        if (warp_id == kConsumerWarp) {
            // tcgen path setup (placeholder descriptors / idesc; kept explicit by design).
            const uint32_t i_desc = (1U << 7U)   // placeholder atype field
                                  | (1U << 10U)  // placeholder btype field
                                  | ((uint32_t)kStageTokens >> 3U << 17U)
                                  | ((uint32_t)128 >> 7U << 27U);

            for (int s = 0; s < kNumStages; ++s) {
                const int tile_id = tile_base + s;
                if (tile_id >= num_tiles) {
                    continue;
                }

                const int valid_tokens = stage_valid_tokens[s];
                const int page_idx = stage_page_idx[s];
                if (valid_tokens <= 0 || page_idx < 0 || page_idx >= num_pages) {
                    continue;
                }

                // Wait for producer-issued TMA transaction for this stage.
                if (kEnableAsyncTma && k_fp8_tmap != nullptr && lane == 0) {
                    const int mbar_addr = cvta_to_shared_u32(&stage_mbar[s]);
                    mbarrier_wait_parity(mbar_addr, stage_phase[s]);
                    stage_phase[s] ^= 1;
                }
                __syncwarp();

                // Explicit tcgen issue point in consumer warp.
                if (kEnableTcgenMma && lane == 0) {
                    const int q_addr = cvta_to_shared_u32(q_stage);
                    const int k_addr = cvta_to_shared_u32(k_stage_payload + s * kStageTokens * kPayloadBytesPerToken);
                    const uint64_t desc_a = make_desc_kmajor_128b(q_addr);
                    const uint64_t desc_b = make_desc_kmajor_128b(k_addr);
                    tcgen05_mma_f16_ss(/*tmem_c=*/0, desc_a, desc_b, i_desc, /*accumulate=*/0);
                    tcgen05_commit(cvta_to_shared_u32(&stage_mbar[s]));
                    tcgen05_wait_ld();
                    tcgen05_fence_before_thread_sync();
                }
                __syncwarp();

                // Scalar reduce path (kept as correctness anchor while tcgen path is refined).
                const uint8_t* stage_payload = k_stage_payload + s * kStageTokens * kPayloadBytesPerToken;
                const float* stage_scale = k_stage_scale + s * kStageTokens;

                for (int t = lane; t < valid_tokens; t += 32) {
                    const int seq_token = tile_id * kStageTokens + t;
                    float reduced = 0.0f;
                    const uint8_t* row = stage_payload + t * kPayloadBytesPerToken;
                    const float scale = stage_scale[t];

                    for (int h = 0; h < kNumHeads; ++h) {
                        float dot = 0.0f;
                        for (int d = 0; d < kHeadDim; ++d) {
                            const int64_t q_off = (static_cast<int64_t>(b) * kNumHeads + h) * kHeadDim + d;
                            float qv;
                            if (q_elem_size == 1) {
                                qv = fp8_byte_to_float(q_stage[h * kHeadDim + d]);
                            } else if (q_elem_size == 2) {
                                const __half* q_half = reinterpret_cast<const __half*>(q_index_bytes);
                                qv = __half2float(q_half[q_off]);
                            } else {
                                const float* q_f32 = reinterpret_cast<const float*>(q_index_bytes);
                                qv = q_f32[q_off];
                            }
                            const float kv = fp8_byte_to_float(row[d]) * scale;
                            dot += qv * kv;
                        }
                        if (dot > 0.0f) {
                            reduced += dot * w_stage[h];
                        }
                    }

                    if (use_smem_scores) {
                        final_scores_smem[seq_token] = reduced;
                    } else {
                        spill_b[seq_token] = reduced;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Final exact top-k in consumer lane 0.
    if (warp_id == kConsumerWarp && lane == 0) {
        const int actual_topk = (topk < seq_len) ? topk : seq_len;
        for (int i = 0; i < actual_topk; ++i) {
            topk_scores[i] = -FLT_MAX;
            topk_ids[i] = -1;
        }

        int topk_count = 0;
        for (int seq_token = 0; seq_token < seq_len; ++seq_token) {
            const float score = use_smem_scores ? final_scores_smem[seq_token] : spill_b[seq_token];
            const int page_slot = seq_token / kPageSize;
            const int offset = seq_token % kPageSize;

            int page_idx = -1;
            if (page_slot >= 0 && page_slot < max_num_pages) {
                page_idx = block_table_b[page_slot];
            }
            if (page_idx < 0 || page_idx >= num_pages) {
                continue;
            }

            const int token_idx = page_idx * kPageSize + offset;
            insert_topk_desc(score, token_idx, topk_scores, topk_ids, topk_count, actual_topk);
        }

        for (int i = 0; i < actual_topk; ++i) {
            out_b[i] = (i < topk_count) ? topk_ids[i] : -1;
        }
    }
}

// ---------------------------------------------------------------------------------
// Host-side tensor-map encode helpers
// ---------------------------------------------------------------------------------
static void* g_k_fp8_tmap_dev = nullptr;

static void ensure_k_fp8_tmap_on_device(const int8_t* k_index_cache_ptr, int num_pages) {
    if (g_k_fp8_tmap_dev == nullptr) {
        cudaMalloc(&g_k_fp8_tmap_dev, sizeof(CUtensorMap));
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
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (st == CUDA_SUCCESS) {
        cudaMemcpy(g_k_fp8_tmap_dev, &k_fp8_tmap, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    } else {
        // Best-effort mode: keep pointer null to skip TMA path if encode fails.
        g_k_fp8_tmap_dev = nullptr;
    }
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
    torch::Tensor spill_scores,
    torch::Tensor topk_indices
) {
    const int batch_size = static_cast<int>(q_index_fp8.size(0));
    const int num_pages = static_cast<int>(k_index_cache_fp8.size(0));
    const int max_num_pages = static_cast<int>(block_table.size(1));
    const int spill_stride = static_cast<int>(spill_scores.size(1));
    const int topk = static_cast<int>(topk_indices.size(1));
    const int q_elem_size = static_cast<int>(q_index_fp8.element_size());

    if (batch_size == 0 || topk == 0) {
        return;
    }

    ensure_k_fp8_tmap_on_device(reinterpret_cast<const int8_t*>(k_index_cache_fp8.data_ptr()), num_pages);

    int device = -1;
    cudaGetDevice(&device);

    int max_optin_smem = 0;
    cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    int dynamic_smem_bytes = kDesiredDynamicSmemBytes;
    if (max_optin_smem > 0 && dynamic_smem_bytes > max_optin_smem) {
        dynamic_smem_bytes = max_optin_smem;
    }

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
        reinterpret_cast<float*>(spill_scores.data_ptr()),
        reinterpret_cast<int*>(topk_indices.data_ptr()),
        batch_size,
        num_pages,
        max_num_pages,
        spill_stride,
        topk,
        q_elem_size,
        dynamic_smem_bytes
    );
}
"""


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_indexer_singlefile_ext",
            cpp_sources="",
            cuda_sources=cuda_src,
            functions=["dsa_topk_indexer_launch"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _module


def compile_kernel():
    _get_module()


def _get_spill_workspace(device: torch.device, batch: int, spill_stride: int) -> torch.Tensor:
    dev_key = (device.type, -1 if device.index is None else device.index)
    ws = _spill_workspace.get(dev_key)

    if ws is None or ws.shape[0] < batch or ws.shape[1] < spill_stride:
        rows = batch if ws is None else max(batch, ws.shape[0])
        cols = spill_stride if ws is None else max(spill_stride, ws.shape[1])
        ws = torch.empty((rows, cols), dtype=torch.float32, device=device)
        _spill_workspace[dev_key] = ws

    return ws[:batch, :spill_stride]


def _dsa_topk_indexer(
    q_index_fp8: torch.Tensor,
    k_index_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    batch = int(q_index_fp8.shape[0])
    spill_stride = int(block_table.shape[1]) * 64
    spill_scores = _get_spill_workspace(q_index_fp8.device, batch, spill_stride)

    mod = _get_module()
    mod.dsa_topk_indexer_launch(
        q_index_fp8,
        k_index_cache_fp8,
        weights,
        seq_lens,
        block_table,
        spill_scores,
        topk_indices,
    )
    return topk_indices


def custom_kernel(data: input_t) -> output_t:
    q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table = data
    batch = int(q_index_fp8.shape[0])
    topk_indices = torch.empty((batch, 2048), dtype=torch.int32, device=q_index_fp8.device)
    out = _dsa_topk_indexer(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices)
    return (out,)
