// Split-K GEMM: K-dimension parallelism with reduction
#pragma once

#include "ptx_lib/ptx_common.cuh"
#include "ptx_lib/ptx_mbarrier.cuh"
#include "ptx_lib/ptx_tma.cuh"
#include "ptx_lib/ptx_tcgen05_mma.cuh"
#include "ptx_lib/ptx_tcgen05_ldst.cuh"
#include "ptx_lib/ptx_tcgen05_sync.cuh"

namespace split_k_gemm {

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;
constexpr int SPLIT_K = 4;
constexpr int NUM_STAGES = 4;

constexpr int SMEM_A_PER_STAGE = TILE_M * TILE_K * sizeof(__half);
constexpr int SMEM_B_PER_STAGE = TILE_K * TILE_N * sizeof(__half);
constexpr int SMEM_A_TOTAL = SMEM_A_PER_STAGE * NUM_STAGES;
constexpr int SMEM_B_TOTAL = SMEM_B_PER_STAGE * NUM_STAGES;
constexpr int SMEM_TOTAL = SMEM_A_TOTAL + SMEM_B_TOTAL;

constexpr int MBAR_SIZE = 8;
constexpr int MBAR_OFFSET = SMEM_TOTAL;

constexpr int TMA_A_BYTES = SMEM_A_PER_STAGE;
constexpr int TMA_B_BYTES = SMEM_B_PER_STAGE;
constexpr int TMEM_C_OFFSET = 0;

__device__ __forceinline__ int smem_A_offset(int stage) { return stage * SMEM_A_PER_STAGE; }
__device__ __forceinline__ int smem_B_offset(int stage) { return SMEM_A_TOTAL + stage * SMEM_B_PER_STAGE; }
__device__ __forceinline__ int mbar_offset(int stage) { return MBAR_OFFSET + stage * MBAR_SIZE; }

}

// Phase 1: Partial GEMM (grid.z = SPLIT_K)
template <int M, int N, int K>
__global__ void split_k_partial_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ workspace,
    int ldC
) {
    using namespace split_k_gemm;

    extern __shared__ char smem[];

    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int split_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int k_per_split = K / SPLIT_K;
    const int k_start = split_id * k_per_split;
    const int k_end = (split_id == SPLIT_K - 1) ? K : k_start + k_per_split;
    const int num_k_tiles = (k_end - k_start) / TILE_K;

    if (tid < NUM_STAGES) {
        mbarrier_init(mbar_offset(tid), 1);
    }
    __syncthreads();

    // Prologue
    int prologue = min(NUM_STAGES - 1, num_k_tiles);
    for (int k = 0; k < prologue; k++) {
        int stage = k % NUM_STAGES;
        int global_k = k_start + k * TILE_K;
        mbarrier_arrive_expect_tx(mbar_offset(stage), TMA_A_BYTES + TMA_B_BYTES);
        tma_2d_gmem2smem<1>(smem_A_offset(stage), tma_desc_A, global_k, tile_m * TILE_M, mbar_offset(stage), 0);
        tma_2d_gmem2smem<1>(smem_B_offset(stage), tma_desc_B, tile_n * TILE_N, global_k, mbar_offset(stage), 0);
    }

    // Main loop
    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;
        mbarrier_wait(mbar_offset(stage), (k / NUM_STAGES) & 1);

        uint64_t desc_A = 0, desc_B = 0;
        uint32_t idesc = 0;
        tcgen05_mma_f16_ss(TMEM_C_OFFSET, desc_A, desc_B, idesc, (k > 0) ? 1 : 0);

        int next_k = k + NUM_STAGES - 1;
        if (next_k < num_k_tiles) {
            int next_stage = next_k % NUM_STAGES;
            int global_k = k_start + next_k * TILE_K;
            mbarrier_arrive_expect_tx(mbar_offset(next_stage), TMA_A_BYTES + TMA_B_BYTES);
            tma_2d_gmem2smem<1>(smem_A_offset(next_stage), tma_desc_A, global_k, tile_m * TILE_M, mbar_offset(next_stage), 0);
            tma_2d_gmem2smem<1>(smem_B_offset(next_stage), tma_desc_B, tile_n * TILE_N, global_k, mbar_offset(next_stage), 0);
        }
    }

    // Store to workspace[split_id, :, :]
    tcgen05_wait_ld();
    float c_frag[8];
    tcgen05_ld_16x256b(c_frag, warp_id, lane_id, 8);

    float* partial_C = workspace + split_id * M * N;
    int global_row = tile_m * TILE_M + (tid / (TILE_N / 8)) * 8;
    int global_col = tile_n * TILE_N + (tid % (TILE_N / 8)) * 8;

    if (global_row < M && global_col < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (global_col + i < N) {
                partial_C[global_row * N + global_col + i] = c_frag[i];
            }
        }
    }
}

// Phase 2: Reduce partials
template <int M, int N>
__global__ void split_k_reduce_kernel(
    const float* __restrict__ workspace,
    float* __restrict__ C,
    int ldC
) {
    using namespace split_k_gemm;

    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int tid = threadIdx.x;

    const int elements_per_tile = TILE_M * TILE_N;
    const int elements_per_thread = elements_per_tile / blockDim.x;

    for (int e = 0; e < elements_per_thread; e++) {
        int local_idx = tid + e * blockDim.x;
        int local_row = local_idx / TILE_N;
        int local_col = local_idx % TILE_N;

        int global_row = tile_m * TILE_M + local_row;
        int global_col = tile_n * TILE_N + local_col;

        if (global_row < M && global_col < N) {
            float sum = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLIT_K; s++) {
                sum += workspace[s * M * N + global_row * N + global_col];
            }
            C[global_row * ldC + global_col] = sum;
        }
    }
}

// Alternative: atomic reduction (no workspace)
template <int M, int N, int K>
__global__ void split_k_atomic_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ C,
    int ldC
) {
    using namespace split_k_gemm;

    extern __shared__ char smem[];

    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int split_id = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int k_per_split = K / SPLIT_K;
    const int k_start = split_id * k_per_split;
    const int k_end = (split_id == SPLIT_K - 1) ? K : k_start + k_per_split;
    const int num_k_tiles = (k_end - k_start) / TILE_K;

    if (tid < NUM_STAGES) {
        mbarrier_init(mbar_offset(tid), 1);
    }
    __syncthreads();

    int prologue = min(NUM_STAGES - 1, num_k_tiles);
    for (int k = 0; k < prologue; k++) {
        int stage = k % NUM_STAGES;
        int global_k = k_start + k * TILE_K;
        mbarrier_arrive_expect_tx(mbar_offset(stage), TMA_A_BYTES + TMA_B_BYTES);
        tma_2d_gmem2smem<1>(smem_A_offset(stage), tma_desc_A, global_k, tile_m * TILE_M, mbar_offset(stage), 0);
        tma_2d_gmem2smem<1>(smem_B_offset(stage), tma_desc_B, tile_n * TILE_N, global_k, mbar_offset(stage), 0);
    }

    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;
        mbarrier_wait(mbar_offset(stage), (k / NUM_STAGES) & 1);

        uint64_t desc_A = 0, desc_B = 0;
        uint32_t idesc = 0;
        tcgen05_mma_f16_ss(TMEM_C_OFFSET, desc_A, desc_B, idesc, (k > 0) ? 1 : 0);

        int next_k = k + NUM_STAGES - 1;
        if (next_k < num_k_tiles) {
            int next_stage = next_k % NUM_STAGES;
            int global_k = k_start + next_k * TILE_K;
            mbarrier_arrive_expect_tx(mbar_offset(next_stage), TMA_A_BYTES + TMA_B_BYTES);
            tma_2d_gmem2smem<1>(smem_A_offset(next_stage), tma_desc_A, global_k, tile_m * TILE_M, mbar_offset(next_stage), 0);
            tma_2d_gmem2smem<1>(smem_B_offset(next_stage), tma_desc_B, tile_n * TILE_N, global_k, mbar_offset(next_stage), 0);
        }
    }

    tcgen05_wait_ld();
    float c_frag[8];
    tcgen05_ld_16x256b(c_frag, warp_id, lane_id, 8);

    int global_row = tile_m * TILE_M + (tid / (TILE_N / 8)) * 8;
    int global_col = tile_n * TILE_N + (tid % (TILE_N / 8)) * 8;

    if (global_row < M && global_col < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (global_col + i < N) {
                atomicAdd(&C[global_row * ldC + global_col + i], c_frag[i]);
            }
        }
    }
}
