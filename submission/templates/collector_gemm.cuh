// Collector GEMM: A-matrix reuse via tcgen05 collector
#pragma once

#include "ptx_lib/ptx_common.cuh"
#include "ptx_lib/ptx_mbarrier.cuh"
#include "ptx_lib/ptx_tma.cuh"
#include "ptx_lib/ptx_tcgen05_cp.cuh"
#include "ptx_lib/ptx_tcgen05_mma.cuh"
#include "ptx_lib/ptx_tcgen05_ldst.cuh"
#include "ptx_lib/ptx_tcgen05_sync.cuh"

namespace collector_gemm {

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;
constexpr int N_TILES_PER_A = 4;  // reuse A across this many N tiles
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
constexpr int TMEM_SCALE_A_OFFSET = 4096;
constexpr int TMEM_SCALE_B_OFFSET = 4096 + 512;

__device__ __forceinline__ int smem_A_offset(int stage) { return stage * SMEM_A_PER_STAGE; }
__device__ __forceinline__ int smem_B_offset(int stage) { return SMEM_A_TOTAL + stage * SMEM_B_PER_STAGE; }
__device__ __forceinline__ int mbar_offset(int stage) { return MBAR_OFFSET + stage * MBAR_SIZE; }

}

// Block-scaled MMA with collector pattern: fill -> use -> ... -> lastuse
template <int M, int N, int K>
__global__ void collector_gemm_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ C,
    int ldC
) {
    using namespace collector_gemm;

    extern __shared__ char smem[];

    const int tile_m = blockIdx.x;
    const int tile_n_base = blockIdx.y * N_TILES_PER_A;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_k_tiles = K / TILE_K;
    const int tiles_n = (N + TILE_N - 1) / TILE_N;

    if (tid < NUM_STAGES) {
        mbarrier_init(mbar_offset(tid), 1);
    }
    __syncthreads();

    for (int n_offset = 0; n_offset < N_TILES_PER_A; n_offset++) {
        int tile_n = tile_n_base + n_offset;
        if (tile_n >= tiles_n) break;

        // Prologue
        int prologue = min(NUM_STAGES - 1, num_k_tiles);
        for (int k = 0; k < prologue; k++) {
            int stage = k % NUM_STAGES;
            mbarrier_arrive_expect_tx(mbar_offset(stage), TMA_A_BYTES + TMA_B_BYTES);

            if (n_offset == 0) {
                tma_2d_gmem2smem<1>(smem_A_offset(stage), tma_desc_A, k * TILE_K, tile_m * TILE_M, mbar_offset(stage), 0);
            }
            tma_2d_gmem2smem<1>(smem_B_offset(stage), tma_desc_B, tile_n * TILE_N, k * TILE_K, mbar_offset(stage), 0);
        }

        // K-loop with collector
        for (int k = 0; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;
            mbarrier_wait(mbar_offset(stage), (k / NUM_STAGES) & 1);

            uint64_t desc_A = 0, desc_B = 0;
            uint32_t idesc = 0;
            int accumulate = (k > 0) ? 1 : 0;

            // Collector usage based on position in N-tile loop
            if (n_offset == 0) {
                tcgen05_mma_mxf4nvf4_block16<1, COLLECTOR_USAGE::A_FILL>(
                    TMEM_C_OFFSET, desc_A, desc_B, idesc,
                    TMEM_SCALE_A_OFFSET, TMEM_SCALE_B_OFFSET, accumulate);
            } else if (n_offset == N_TILES_PER_A - 1) {
                tcgen05_mma_mxf4nvf4_block16<1, COLLECTOR_USAGE::A_LASTUSE>(
                    TMEM_C_OFFSET, desc_A, desc_B, idesc,
                    TMEM_SCALE_A_OFFSET, TMEM_SCALE_B_OFFSET, accumulate);
            } else {
                tcgen05_mma_mxf4nvf4_block16<1, COLLECTOR_USAGE::A_USE>(
                    TMEM_C_OFFSET, desc_A, desc_B, idesc,
                    TMEM_SCALE_A_OFFSET, TMEM_SCALE_B_OFFSET, accumulate);
            }

            int next_k = k + NUM_STAGES - 1;
            if (next_k < num_k_tiles) {
                int next_stage = next_k % NUM_STAGES;
                mbarrier_arrive_expect_tx(mbar_offset(next_stage), TMA_A_BYTES + TMA_B_BYTES);
                if (n_offset == 0) {
                    tma_2d_gmem2smem<1>(smem_A_offset(next_stage), tma_desc_A, next_k * TILE_K, tile_m * TILE_M, mbar_offset(next_stage), 0);
                }
                tma_2d_gmem2smem<1>(smem_B_offset(next_stage), tma_desc_B, tile_n * TILE_N, next_k * TILE_K, mbar_offset(next_stage), 0);
            }
        }

        // Store
        tcgen05_wait_ld();
        float c_frag[8];
        tcgen05_ld_16x256b(c_frag, warp_id, lane_id, 8);

        int global_row = tile_m * TILE_M + (tid / (TILE_N / 8)) * 8;
        int global_col = tile_n * TILE_N + (tid % (TILE_N / 8)) * 8;

        if (global_row < M && global_col < N) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (global_col + i < N) {
                    C[global_row * ldC + global_col + i] = c_frag[i];
                }
            }
        }
        __syncthreads();
    }
}

// F16 variant: A in tmem, reused across N tiles
template <int M, int N, int K>
__global__ void collector_f16_gemm_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ C,
    int ldC
) {
    using namespace collector_gemm;

    extern __shared__ char smem[];

    const int tile_m = blockIdx.x;
    const int tile_n_base = blockIdx.y * N_TILES_PER_A;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int num_k_tiles = K / TILE_K;
    const int tiles_n = (N + TILE_N - 1) / TILE_N;

    constexpr int TMEM_A_OFFSET = 4096;

    if (tid < NUM_STAGES) {
        mbarrier_init(mbar_offset(tid), 1);
    }
    __syncthreads();

    for (int n_offset = 0; n_offset < N_TILES_PER_A; n_offset++) {
        int tile_n = tile_n_base + n_offset;
        if (tile_n >= tiles_n) break;

        for (int k = 0; k < num_k_tiles; k++) {
            int stage = k % NUM_STAGES;

            mbarrier_arrive_expect_tx(mbar_offset(stage), TMA_B_BYTES);
            tma_2d_gmem2smem<1>(smem_B_offset(stage), tma_desc_B, tile_n * TILE_N, k * TILE_K, mbar_offset(stage), 0);

            mbarrier_wait(mbar_offset(stage), (k / NUM_STAGES) & 1);

            uint64_t desc_B = 0;
            uint32_t idesc = 0;

            tcgen05_mma_f16_ts(TMEM_C_OFFSET, TMEM_A_OFFSET + k * 256, desc_B, idesc, (k > 0) ? 1 : 0);
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
                    C[global_row * ldC + global_col + i] = c_frag[i];
                }
            }
        }
        __syncthreads();
    }
}
