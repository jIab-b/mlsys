// Simple GEMM: Single-stage TMA + MMA
#pragma once

#include "ptx_lib/ptx_common.cuh"
#include "ptx_lib/ptx_mbarrier.cuh"
#include "ptx_lib/ptx_tma.cuh"
#include "ptx_lib/ptx_tcgen05_cp.cuh"
#include "ptx_lib/ptx_tcgen05_mma.cuh"
#include "ptx_lib/ptx_tcgen05_ldst.cuh"
#include "ptx_lib/ptx_tcgen05_sync.cuh"

namespace simple_gemm {

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;

constexpr int SMEM_A_SIZE = TILE_M * TILE_K * sizeof(__half);
constexpr int SMEM_B_SIZE = TILE_K * TILE_N * sizeof(__half);
constexpr int SMEM_TOTAL = SMEM_A_SIZE + SMEM_B_SIZE;
constexpr int MBAR_OFFSET = SMEM_TOTAL;

constexpr int TMA_A_BYTES = SMEM_A_SIZE;
constexpr int TMA_B_BYTES = SMEM_B_SIZE;
constexpr int TMEM_C_OFFSET = 0;

}

template <int M, int N, int K>
__global__ void simple_gemm_kernel(
    const void* __restrict__ tma_desc_A,
    const void* __restrict__ tma_desc_B,
    float* __restrict__ C,
    int ldC
) {
    using namespace simple_gemm;

    extern __shared__ char smem[];
    int mbar_addr = MBAR_OFFSET;

    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    if (tid == 0) {
        mbarrier_init(mbar_addr, 1);
    }
    __syncthreads();

    const int num_k_tiles = K / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int tma_A_x = k_tile * TILE_K;
        int tma_A_y = block_m * TILE_M;
        int tma_B_x = block_n * TILE_N;
        int tma_B_y = k_tile * TILE_K;

        mbarrier_arrive_expect_tx(mbar_addr, TMA_A_BYTES + TMA_B_BYTES);

        tma_2d_gmem2smem<1>(0, tma_desc_A, tma_A_x, tma_A_y, mbar_addr, 0);
        tma_2d_gmem2smem<1>(SMEM_A_SIZE, tma_desc_B, tma_B_x, tma_B_y, mbar_addr, 0);

        mbarrier_wait(mbar_addr, k_tile & 1);

        uint64_t desc_A = 0;  // TODO: build smem descriptor
        uint64_t desc_B = 0;
        uint32_t idesc = 0;

        tcgen05_mma_f16_ss(TMEM_C_OFFSET, desc_A, desc_B, idesc, (k_tile > 0) ? 1 : 0);

        __syncthreads();
    }

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
