// Warp-Specialized GEMM: Producer/Consumer pattern
#pragma once

#include "ptx_lib/ptx_common.cuh"
#include "ptx_lib/ptx_mbarrier.cuh"
#include "ptx_lib/ptx_tma.cuh"
#include "ptx_lib/ptx_tcgen05_cp.cuh"
#include "ptx_lib/ptx_tcgen05_mma.cuh"
#include "ptx_lib/ptx_tcgen05_ldst.cuh"
#include "ptx_lib/ptx_tcgen05_sync.cuh"

namespace warp_specialized_gemm {

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;

constexpr int NUM_PRODUCER_WARPS = 1;
constexpr int NUM_CONSUMER_WARPS = 3;
constexpr int TOTAL_WARPS = NUM_PRODUCER_WARPS + NUM_CONSUMER_WARPS;
constexpr int THREADS_PER_BLOCK = TOTAL_WARPS * 32;

constexpr int NUM_STAGES = 4;

constexpr int SMEM_A_PER_STAGE = TILE_M * TILE_K * sizeof(__half);
constexpr int SMEM_B_PER_STAGE = TILE_K * TILE_N * sizeof(__half);
constexpr int SMEM_A_TOTAL = SMEM_A_PER_STAGE * NUM_STAGES;
constexpr int SMEM_B_TOTAL = SMEM_B_PER_STAGE * NUM_STAGES;
constexpr int SMEM_TOTAL = SMEM_A_TOTAL + SMEM_B_TOTAL;

constexpr int MBAR_SIZE = 8;
constexpr int MBAR_FULL_OFFSET = SMEM_TOTAL;
constexpr int MBAR_EMPTY_OFFSET = MBAR_FULL_OFFSET + NUM_STAGES * MBAR_SIZE;

constexpr int TMA_A_BYTES = SMEM_A_PER_STAGE;
constexpr int TMA_B_BYTES = SMEM_B_PER_STAGE;

constexpr int TMEM_C_OFFSET = 0;
constexpr int TMEM_A_OFFSET = 4096;

__device__ __forceinline__ int smem_A_offset(int stage) { return stage * SMEM_A_PER_STAGE; }
__device__ __forceinline__ int smem_B_offset(int stage) { return SMEM_A_TOTAL + stage * SMEM_B_PER_STAGE; }
__device__ __forceinline__ int mbar_full(int stage) { return MBAR_FULL_OFFSET + stage * MBAR_SIZE; }
__device__ __forceinline__ int mbar_empty(int stage) { return MBAR_EMPTY_OFFSET + stage * MBAR_SIZE; }

}

template <int M, int N, int K>
__device__ void producer_warp(
    const void* tma_desc_A,
    const void* tma_desc_B,
    int block_m,
    int block_n
) {
    using namespace warp_specialized_gemm;
    const int num_k_tiles = K / TILE_K;

    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;

        if (k >= NUM_STAGES) {
            mbarrier_wait(mbar_empty(stage), ((k - NUM_STAGES) / NUM_STAGES) & 1);
        }

        mbarrier_arrive_expect_tx(mbar_full(stage), TMA_A_BYTES + TMA_B_BYTES);
        tma_2d_gmem2smem<1>(smem_A_offset(stage), tma_desc_A, k * TILE_K, block_m * TILE_M, mbar_full(stage), 0);
        tma_2d_gmem2smem<1>(smem_B_offset(stage), tma_desc_B, block_n * TILE_N, k * TILE_K, mbar_full(stage), 0);
    }
}

template <int M, int N, int K>
__device__ void consumer_warp(int consumer_id) {
    using namespace warp_specialized_gemm;
    const int num_k_tiles = K / TILE_K;

    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;

        mbarrier_wait(mbar_full(stage), (k / NUM_STAGES) & 1);

        uint64_t desc_A = 0;  // TODO
        tcgen05_cp_128x256b<1>(TMEM_A_OFFSET, desc_A);

        uint64_t desc_B = 0;
        uint32_t idesc = 0;

        tcgen05_mma_ws_f16_ts(TMEM_C_OFFSET, TMEM_A_OFFSET, desc_B, idesc, (k > 0) ? 1 : 0);

        if (consumer_id == 0) {
            tcgen05_commit<1>(mbar_empty(stage));
        }
    }
}

template <int M, int N, int K>
__global__ void warp_specialized_gemm_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ C,
    int ldC
) {
    using namespace warp_specialized_gemm;

    extern __shared__ char smem[];

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Init barriers
    if (tid < NUM_STAGES) {
        mbarrier_init(mbar_full(tid), 1);
    }
    if (tid >= NUM_STAGES && tid < 2 * NUM_STAGES) {
        mbarrier_init(mbar_empty(tid - NUM_STAGES), 1);
    }
    __syncthreads();

    // Warp divergence
    if (warp_id < NUM_PRODUCER_WARPS) {
        producer_warp<M, N, K>(tma_desc_A, tma_desc_B, block_m, block_n);
    } else {
        consumer_warp<M, N, K>(warp_id - NUM_PRODUCER_WARPS);
    }

    __syncthreads();
    tcgen05_wait_ld();

    // Store
    if (warp_id >= NUM_PRODUCER_WARPS) {
        float c_frag[8];
        tcgen05_ld_16x256b(c_frag, warp_id - NUM_PRODUCER_WARPS, lane_id, 8);

        int global_row = block_m * TILE_M + (warp_id - NUM_PRODUCER_WARPS) * 32 + lane_id;
        int global_col = block_n * TILE_N;

        if (global_row < M) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (global_col + i < N) {
                    C[global_row * ldC + global_col + i] = c_frag[i];
                }
            }
        }
    }
}
