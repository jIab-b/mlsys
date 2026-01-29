// Pipelined GEMM: Multi-stage software pipeline
#pragma once

#include "ptx_lib/ptx_common.cuh"
#include "ptx_lib/ptx_mbarrier.cuh"
#include "ptx_lib/ptx_tma.cuh"
#include "ptx_lib/ptx_tcgen05_mma.cuh"
#include "ptx_lib/ptx_tcgen05_ldst.cuh"
#include "ptx_lib/ptx_tcgen05_sync.cuh"

namespace pipelined_gemm {

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;
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

template <int M, int N, int K>
__global__ void pipelined_gemm_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ C,
    int ldC
) {
    using namespace pipelined_gemm;

    extern __shared__ char smem[];

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_k_tiles = K / TILE_K;

    if (tid < NUM_STAGES) {
        mbarrier_init(mbar_offset(tid), 1);
    }
    __syncthreads();

    // Prologue: fill pipeline
    int prologue_tiles = min(NUM_STAGES - 1, num_k_tiles);
    for (int k = 0; k < prologue_tiles; k++) {
        int stage = k % NUM_STAGES;
        int mbar = mbar_offset(stage);

        mbarrier_arrive_expect_tx(mbar, TMA_A_BYTES + TMA_B_BYTES);
        tma_2d_gmem2smem<1>(smem_A_offset(stage), tma_desc_A, k * TILE_K, block_m * TILE_M, mbar, 0);
        tma_2d_gmem2smem<1>(smem_B_offset(stage), tma_desc_B, block_n * TILE_N, k * TILE_K, mbar, 0);
    }

    // Main loop
    for (int k = 0; k < num_k_tiles; k++) {
        int stage = k % NUM_STAGES;
        int mbar = mbar_offset(stage);

        mbarrier_wait(mbar, (k / NUM_STAGES) & 1);

        uint64_t desc_A = 0;  // TODO
        uint64_t desc_B = 0;
        uint32_t idesc = 0;

        tcgen05_mma_f16_ss(TMEM_C_OFFSET, desc_A, desc_B, idesc, (k > 0) ? 1 : 0);

        int next_k = k + NUM_STAGES - 1;
        if (next_k < num_k_tiles) {
            int next_stage = next_k % NUM_STAGES;
            int next_mbar = mbar_offset(next_stage);

            mbarrier_arrive_expect_tx(next_mbar, TMA_A_BYTES + TMA_B_BYTES);
            tma_2d_gmem2smem<1>(smem_A_offset(next_stage), tma_desc_A, next_k * TILE_K, block_m * TILE_M, next_mbar, 0);
            tma_2d_gmem2smem<1>(smem_B_offset(next_stage), tma_desc_B, block_n * TILE_N, next_k * TILE_K, next_mbar, 0);
        }
    }

    // Epilogue
    tcgen05_wait_ld();

    float c_frag[8];
    tcgen05_ld_16x256b(c_frag, warp_id, lane_id, 8);

    int global_row = block_m * TILE_M + (tid / (TILE_N / 8)) * 8;
    int global_col = block_n * TILE_N + (tid % (TILE_N / 8)) * 8;

    if (global_row < M && global_col < N) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (global_col + i < N) {
                C[global_row * ldC + global_col + i] = c_frag[i];
            }
        }
    }
}
