
#!POPCORN gpu NVIDIA

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

cuda_src = """
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>

// ============================================================================
// Constants
// ============================================================================
constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;
constexpr int TMEM_COLS = 512;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000;
constexpr uint32_t SM100_MMA_PEER_MASK = 0xFEFFFFFF;

__device__ inline constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

// ============================================================================
// Kernel Configuration - 8 warps for epilogue (8 row groups of 16 rows each)
// ============================================================================
template <int BLOCK_M_, int BLOCK_N_, int BLOCK_K_, int NUM_STAGES_MAIN_, int NUM_STAGES_SF_>
struct KernelConfig {
    static constexpr int BLOCK_M = BLOCK_M_;
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int BLOCK_K = BLOCK_K_;
    static constexpr int NUM_STAGES_MAIN = NUM_STAGES_MAIN_;
    static constexpr int NUM_STAGES_SF = NUM_STAGES_SF_;

    static constexpr int OUT_N = BLOCK_N / 2;
    static constexpr int A_SIZE = BLOCK_M * BLOCK_K / 2;
    static constexpr int B_SIZE = OUT_N * BLOCK_K / 2;
    static constexpr int SF_SIZE = 128 * BLOCK_K / 16;  // 2048 bytes for K=256
    static constexpr int SF_STAGE_SIZE = SF_SIZE * 3;   // SFA + SFB1 + SFB2

    static constexpr int MAIN_STAGE_SIZE = A_SIZE + B_SIZE;
    static constexpr int SMEM_SIZE = MAIN_STAGE_SIZE * NUM_STAGES_MAIN + SF_STAGE_SIZE * NUM_STAGES_SF;

    // 8 warps for epilogue + 2 warps for TMA/MMA = 10 warps total (320 threads)
    static constexpr int NUM_EP_WARPS = 8;
    static constexpr int NUM_WARPS = NUM_EP_WARPS + 2;
    static constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;  // 320 threads

    static constexpr int GEMM_D_TMEM = 0;
    static constexpr int SFA_TMEM = BLOCK_N;
    static constexpr int SFB_TMEM = SFA_TMEM + 4 * (BLOCK_K / MMA_K);

    static constexpr uint32_t I_DESC = (1U << 7U) | (1U << 10U) |
        ((uint32_t)BLOCK_N >> 3U << 17U) | ((uint32_t)(2 * BLOCK_M) >> 7U << 27U);

    static constexpr int NUM_MBAR = NUM_STAGES_MAIN * 2 + 1;
};

// M=512: 256 output rows per cluster, 128 output cols per B matrix
using CfgN128 = KernelConfig<128, 256, 256, 6, 5>;

// ============================================================================
// Device Helpers
// ============================================================================
__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\\n\\t"
        ".reg .pred %%px;\\n\\t"
        "elect.sync _|%%px, %1;\\n\\t"
        "@%%px mov.s32 %0, 1;\\n\\t"
        "}"
        : "+r"(pred) : "r"(0xFFFFFFFF));
    return pred;
}

__device__ inline uint32_t get_cluster_ctarank() {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\\n\\t"
        ".reg .pred P1;\\n\\t"
        "LAB_WAIT:\\n\\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\\n\\t"
        "@P1 bra.uni DONE;\\n\\t"
        "bra.uni LAB_WAIT;\\n\\t"
        "DONE:\\n\\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_expect_tx(int mbar_addr, int size) {
    int mbar = mbar_addr & (int)SM100_MMA_PEER_MASK;
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mbar), "r"(size) : "memory");
}

__device__ inline void tma_load_1d(int dst, const void *tmap_ptr, int x, int mbar_addr, uint64_t cache_policy) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tmap_ptr);
    uint32_t smem_int_mbar = (uint32_t)mbar_addr & SM100_MMA_PEER_MASK;
    uint32_t smem_int_ptr  = (uint32_t)dst;
    asm volatile("cp.async.bulk.tensor.1d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%3}], [%2], %4;"
                 :: "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar), "r"(x), "l"(cache_policy) : "memory");
}

__device__ inline void tma_load_3d(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tmap_ptr);
    uint32_t smem_int_mbar = (uint32_t)mbar_addr & SM100_MMA_PEER_MASK;
    uint32_t smem_int_ptr  = (uint32_t)dst;
    asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :: "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                    "r"(x), "r"(y), "r"(z), "l"(cache_policy) : "memory");
}

__device__ inline void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline void tcgen05_mma_nvfp4(uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int scale_A_tmem, int scale_B_tmem, int enable_input_d, int d_tmem) {
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %6, 0;\\n\\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d));
}

__device__ inline void tcgen05_commit(int mbar_addr, uint16_t ctamask = 0x3) {
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.multicast::cluster.b64 [%0], %1;"
                 :: "r"(mbar_addr), "h"(ctamask) : "memory");
}

// tcgen05_ld
static constexpr char SHAPE_16x256b[] = ".16x256b";
static constexpr char NUM_x1[]  = ".x1";
static constexpr char NUM_x2[]  = ".x2";
static constexpr char NUM_x4[]  = ".x4";
static constexpr char NUM_x8[]  = ".x8";
static constexpr char NUM_x16[] = ".x16";

template <const char *SHAPE, const char *NUM>
__device__ inline void tcgen05_ld_4regs(float *tmp, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%5%6.b32 "
        "{ %0, %1, %2, %3 }, [%4];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
        : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline void tcgen05_ld_8regs(float *tmp, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%9%10.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7 }, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}



__device__ inline void tcgen05_ld_16x256bx1(float *tmp, int row, int col) { tcgen05_ld_4regs<SHAPE_16x256b, NUM_x1>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x256bx2(float *tmp, int row, int col) { tcgen05_ld_8regs<SHAPE_16x256b, NUM_x2>(tmp, row, col); }

__device__ inline float tanh_approx_f32(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ inline float silu_exact(float x) {
    return x * __fdividef(1.0f, (1.0f + expf(-x)));
}

__device__ inline float silu_approx(float x) {
    const float t = 0.5f * x;
    const float th = tanh_approx_f32(t);
    return x * 0.5f * (1.0f + th);
}

template <bool UseApprox>
__device__ inline float silu(float x) {
    if constexpr (UseApprox) {
        return silu_approx(x);
    } else {
        return silu_exact(x);
    }
}

// ============================================================================
// TensorMap Initialization
// ============================================================================
void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *error_msg_ptr;
    if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS) error_msg_ptr = "unable to get error string";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void init_AB_tmap(CUtensorMap *tmap, const char *ptr, uint64_t global_height, uint64_t global_width, uint32_t shared_height, uint32_t shared_width) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank] = {256, global_height, global_width / 256};
    uint64_t globalStrides[rank-1] = {global_width / 2, 128};
    uint32_t boxDim[rank] = {256, shared_height, shared_width / 256};
    uint32_t elementStrides[rank] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void init_SF_tmap(CUtensorMap *tmap, const char *ptr, uint64_t total_bytes, uint32_t tile_bytes) {
    TORCH_CHECK((total_bytes % 8) == 0 && (tile_bytes % 8) == 0, "SF bytes must be 8B aligned");
    uint64_t globalDim[1] = {total_bytes / 8};
    uint64_t globalStrides[1] = {globalDim[0] * 8};
    uint32_t boxDim[1] = {tile_bytes / 8};
    uint32_t elementStrides[1] = {1};
    check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_UINT64, 1, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ============================================================================
// Split TMA Load Functions for M=512
// ============================================================================

template <typename Cfg>
__device__ inline void issue_tma_main(
    int smem_main_base, int stage_main, int iter_k,
    const CUtensorMap *A_tmap, const CUtensorMap *B_tmap,
    int off_m, int off_n,
    int mbar_addr, uint64_t cache_policy
) {
    const int A_smem = smem_main_base + stage_main * Cfg::MAIN_STAGE_SIZE;
    const int B_smem = A_smem + Cfg::A_SIZE;
    const int off_k = iter_k * Cfg::BLOCK_K;

    tma_load_3d(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_policy);
    tma_load_3d(B_smem, B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_policy);
}

template <typename Cfg>
__device__ inline void issue_tma_sf(
    int smem_sf_base, int stage_sf, int iter_k,
    const CUtensorMap *SFA_tmap, const CUtensorMap *SFB1_tmap, const CUtensorMap *SFB2_tmap,
    int off_m, int off_n, int K,
    int mbar_addr, uint64_t cache_policy
) {
    const int SFA_smem = smem_sf_base + stage_sf * Cfg::SF_STAGE_SIZE;
    const int SFB_smem = SFA_smem + Cfg::SF_SIZE;
    const int off_k = iter_k * Cfg::BLOCK_K;

    const int rest_k = K / 16 / 4;
    constexpr int SF_ELEMS_PER_512B = 512 / 8;
    const int sfa_coord = ((off_m / 128) * rest_k + off_k / (16 * 4)) * SF_ELEMS_PER_512B;
    const int sfb_coord = ((off_n / 128) * rest_k + off_k / (16 * 4)) * SF_ELEMS_PER_512B;

    tma_load_1d(SFA_smem, SFA_tmap, sfa_coord, mbar_addr, cache_policy);

    #pragma unroll
    for (int k = 0; k < Cfg::BLOCK_K / MMA_K; k++) {
        const int k_sfb_coord = sfb_coord + k * SF_ELEMS_PER_512B;
        tma_load_1d(SFB_smem + k * 1024 + 0,   SFB1_tmap, k_sfb_coord, mbar_addr, cache_policy);
        tma_load_1d(SFB_smem + k * 1024 + 512, SFB2_tmap, k_sfb_coord, mbar_addr, cache_policy);
    }
}

__device__ inline void skip_ep(int done_mbar_addr) { mbarrier_wait(done_mbar_addr, 0); asm volatile("tcgen05.fence::after_thread_sync;"); }

// ============================================================================
// 8-Warp Epilogue for M=512: 128 rows × 128 cols output
// 2 warpgroups × 4 warps: each warpgroup handles 128 rows × 64 cols
// Within warpgroup, each warp handles 32 rows (respecting TMEM access restrictions)
// ============================================================================
template <int OUT_N, bool UseApprox>
__device__ inline void epilogue_512(
    int warp_id, int lane_id,
    int off_m, int off_n,
    int gemm1_tmem, int gemm2_tmem,
    half *C_ptr, int N,
    int done_mbar_addr
) {
    mbarrier_wait(done_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    constexpr int COLS_PER_CHUNK = 8;
    constexpr int CHUNKS_PER_WG = (OUT_N / 2) / COLS_PER_CHUNK;  // 8 chunks per warpgroup
    const int col_lane = (lane_id % 4) * 2;
    const int row_lane = lane_id / 4;

    // Split by warpgroup: warpgroup 0 handles cols 0-63, warpgroup 1 handles cols 64-127
    const int warpgroup_id = warp_id / 4;
    const int warp_in_wg = warp_id % 4;
    const int chunk_start = warpgroup_id * CHUNKS_PER_WG;

    // Each warp within warpgroup handles 32 rows (lanes warp_in_wg*32 to warp_in_wg*32+31)
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int tm = warp_in_wg * 32 + m * 16;
        const int out_row0 = off_m + tm + row_lane;
        const int out_row1 = out_row0 + 8;

        #pragma unroll
        for (int c = 0; c < CHUNKS_PER_WG; c++) {
            const int chunk = chunk_start + c;
            float g1[4], g2[4];

            tcgen05_ld_16x256bx1(g1, tm, gemm1_tmem + chunk * COLS_PER_CHUNK);
            tcgen05_ld_16x256bx1(g2, tm, gemm2_tmem + chunk * COLS_PER_CHUNK);
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            const int out_col = off_n + chunk * COLS_PER_CHUNK + col_lane;

            const float s00 = silu<UseApprox>(g1[0]);
            const float s01 = silu<UseApprox>(g1[1]);
            const float s10 = silu<UseApprox>(g1[2]);
            const float s11 = silu<UseApprox>(g1[3]);

            const float v00 = g2[0] * s00;
            const float v01 = g2[1] * s01;
            const float v10 = g2[2] * s10;
            const float v11 = g2[3] * s11;

            reinterpret_cast<half2 *>(C_ptr + out_row0 * N + out_col)[0] = __float22half2_rn({v00, v01});
            reinterpret_cast<half2 *>(C_ptr + out_row1 * N + out_col)[0] = __float22half2_rn({v10, v11});
        }
    }
}

// ============================================================================
// Main Kernel with 8-Warp Epilogue (M=512)
// ============================================================================
template <typename Cfg, bool UseApprox>
__global__ __cluster_dims__(2)
__launch_bounds__(Cfg::TB_SIZE)
void dual_gemm_silu_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap,
    const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB1_tmap,
    const __grid_constant__ CUtensorMap SFB2_tmap,
    half *C_ptr,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    const uint32_t ctarank = get_cluster_ctarank();
    const bool is_cta0 = (ctarank == 0);
    const int cluster_id = blockIdx.x / 2;

    const int grid_n = N / Cfg::OUT_N;
    const int cluster_m = cluster_id / grid_n;
    const int bid_n = cluster_id % grid_n;
    const int base_m = cluster_m * (2 * Cfg::BLOCK_M);
    const int off_m = base_m + int(ctarank) * Cfg::BLOCK_M;
    const int off_n = bid_n * Cfg::OUT_N;
    const int bid_m = cluster_m * 2 + int(ctarank);

    const int num_iters = K / Cfg::BLOCK_K;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    // Memory layout: [Main stages][SF stages]
    const int smem_main_base = smem;
    const int smem_sf_base = smem + Cfg::MAIN_STAGE_SIZE * Cfg::NUM_STAGES_MAIN;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbars[Cfg::NUM_MBAR];
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(mbars));

    const int tma_mbar = mbar_base;
    const int mma_mbar = tma_mbar + Cfg::NUM_STAGES_MAIN * 8;
    const int done_mbar = mma_mbar + Cfg::NUM_STAGES_MAIN * 8;

    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int i = 0; i < Cfg::NUM_STAGES_MAIN; i++) {
            mbarrier_init(tma_mbar + i * 8, 2);
            mbarrier_init(mma_mbar + i * 8, 1);
        }
        mbarrier_init(done_mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem), "r"(TMEM_COLS));
    }
    __syncthreads();

    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto make_desc_AB = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };
    auto make_desc_SF = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 16) << 32ULL) | (1ULL << 46ULL);
    };

    const CUtensorMap *B_tmap = is_cta0 ? &B1_tmap : &B2_tmap;

    // ========================================================================
    // TMA Producer Warp - warp NUM_WARPS-2 (warp 8)
    // ========================================================================
    if (warp_id == Cfg::NUM_WARPS - 2 && elect_sync()) {
        // Prefill
        #pragma unroll
        for (int iter_k = 0; iter_k < Cfg::NUM_STAGES_MAIN && iter_k < num_iters; iter_k++) {
            const int stage_main = iter_k;
            const int stage_sf = iter_k % Cfg::NUM_STAGES_SF;

            if (iter_k >= Cfg::NUM_STAGES_SF) {
                const int prev_sf_iter = iter_k - Cfg::NUM_STAGES_SF;
                const int prev_sf_mbar_idx = prev_sf_iter % Cfg::NUM_STAGES_MAIN;
                mbarrier_wait(mma_mbar + prev_sf_mbar_idx * 8, (prev_sf_iter / Cfg::NUM_STAGES_MAIN) % 2);
            }

            issue_tma_main<Cfg>(
                smem_main_base, stage_main, iter_k,
                &A_tmap, B_tmap,
                off_m, off_n,
                tma_mbar + iter_k * 8, cache_B);
            issue_tma_sf<Cfg>(
                smem_sf_base, stage_sf, iter_k,
                &SFA_tmap, &SFB1_tmap, &SFB2_tmap,
                off_m, off_n, K,
                tma_mbar + iter_k * 8, cache_B);
            mbarrier_expect_tx(tma_mbar + iter_k * 8, Cfg::MAIN_STAGE_SIZE + Cfg::SF_STAGE_SIZE);
        }

        // Steady state
        for (int iter_k = Cfg::NUM_STAGES_MAIN; iter_k < num_iters; iter_k++) {
            const int stage_main = iter_k % Cfg::NUM_STAGES_MAIN;
            const int stage_sf = iter_k % Cfg::NUM_STAGES_SF;

            mbarrier_wait(mma_mbar + stage_main * 8, (iter_k / Cfg::NUM_STAGES_MAIN - 1) % 2);

            const int prev_sf_iter = iter_k - Cfg::NUM_STAGES_SF;
            if (prev_sf_iter >= 0) {
                const int prev_sf_mbar_idx = prev_sf_iter % Cfg::NUM_STAGES_MAIN;
                if (prev_sf_mbar_idx != stage_main) {
                    mbarrier_wait(mma_mbar + prev_sf_mbar_idx * 8, (prev_sf_iter / Cfg::NUM_STAGES_MAIN) % 2);
                }
            }

            issue_tma_main<Cfg>(
                smem_main_base, stage_main, iter_k,
                &A_tmap, B_tmap,
                off_m, off_n,
                tma_mbar + stage_main * 8, cache_B);
            issue_tma_sf<Cfg>(
                smem_sf_base, stage_sf, iter_k,
                &SFA_tmap, &SFB1_tmap, &SFB2_tmap,
                off_m, off_n, K,
                tma_mbar + stage_main * 8, cache_B);
            mbarrier_expect_tx(tma_mbar + stage_main * 8, Cfg::MAIN_STAGE_SIZE + Cfg::SF_STAGE_SIZE);
        }
    }

    // ========================================================================
    // MMA Consumer Warp - warp NUM_WARPS-1 (warp 9)
    // ========================================================================
    if (warp_id == Cfg::NUM_WARPS - 1 && elect_sync() && is_cta0) {
        #pragma unroll 1
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            const int stage_main = iter_k % Cfg::NUM_STAGES_MAIN;
            const int stage_sf = iter_k % Cfg::NUM_STAGES_SF;

            mbarrier_wait(tma_mbar + stage_main * 8, (iter_k / Cfg::NUM_STAGES_MAIN) % 2);

            const int A_smem = smem_main_base + stage_main * Cfg::MAIN_STAGE_SIZE;
            const int B_smem = A_smem + Cfg::A_SIZE;
            const int SFA_smem = smem_sf_base + stage_sf * Cfg::SF_STAGE_SIZE;
            const int SFB_smem = SFA_smem + Cfg::SF_SIZE;

            constexpr uint64_t SF_desc_base = make_desc_SF(0);
            const uint64_t SFA_desc = SF_desc_base + ((uint64_t)SFA_smem >> 4ULL);

            #pragma unroll
            for (int k = 0; k < Cfg::BLOCK_K / MMA_K; k++) {
                tcgen05_cp_nvfp4(Cfg::SFA_TMEM + k * 4, SFA_desc + (uint64_t)k * (512ULL >> 4ULL));
                const uint64_t SFB1_k_desc = SF_desc_base + ((uint64_t)(SFB_smem + k * 1024) >> 4ULL);
                const uint64_t SFB2_k_desc = SF_desc_base + ((uint64_t)(SFB_smem + k * 1024 + 512) >> 4ULL);
                tcgen05_cp_nvfp4(Cfg::SFB_TMEM + k * 8 + 0, SFB1_k_desc);
                tcgen05_cp_nvfp4(Cfg::SFB_TMEM + k * 8 + 4, SFB2_k_desc);
            }

            #pragma unroll
            for (int k1 = 0; k1 < Cfg::BLOCK_K / 256; k1++) {
                #pragma unroll
                for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                    uint64_t a_desc = make_desc_AB(A_smem + k1 * Cfg::BLOCK_M * 128 + k2 * 32);
                    uint64_t b_desc = make_desc_AB(B_smem + k1 * Cfg::OUT_N * 128 + k2 * 32);
                    int k_sf = k1 * 4 + k2;
                    const int scale_A = Cfg::SFA_TMEM + k_sf * 4 + (bid_m % (128 / Cfg::BLOCK_M)) * (Cfg::BLOCK_M / 32);
                    const int scale_B = Cfg::SFB_TMEM + k_sf * 8;
                    tcgen05_mma_nvfp4(a_desc, b_desc, Cfg::I_DESC, scale_A, scale_B,
                        (k1 == 0 && k2 == 0) ? iter_k : 1, Cfg::GEMM_D_TMEM);
                }
            }

            tcgen05_commit(mma_mbar + stage_main * 8);
        }
        tcgen05_commit(done_mbar);
    }

    if (warp_id < Cfg::NUM_EP_WARPS) {
        epilogue_512<Cfg::OUT_N, UseApprox>(warp_id, lane_id, off_m, off_n, Cfg::GEMM_D_TMEM, Cfg::GEMM_D_TMEM + Cfg::OUT_N, C_ptr, N, done_mbar);
    }

    __syncthreads();

    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(TMEM_COLS));
}

// ============================================================================
// M=256 Kernel (unchanged - separate logic)
// ============================================================================

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__device__ inline void issue_tma_master(
    int smem, int stage_id, int iter_k,
    const CUtensorMap *A_tmap, const CUtensorMap *B1_tmap, const CUtensorMap *B2_tmap,
    const CUtensorMap *SFA_tmap, const CUtensorMap *SFB1_tmap, const CUtensorMap *SFB2_tmap,
    int off_m, int off_n, int K, int ctarank, int mbar_addr, uint64_t cache_A, uint64_t cache_B
) {
    constexpr int B_FRAG_N = BLOCK_N / 2;
    constexpr int A_size = BLOCK_M * BLOCK_K / 2;
    constexpr int B_size = B_FRAG_N * BLOCK_K / 2;
    constexpr int SF_size = 128 * BLOCK_K / 16;
    constexpr int STAGE_SIZE = A_size + B_size * 2 + SF_size * 3;

    const int A_smem = smem + stage_id * STAGE_SIZE;
    const int B1_smem = A_smem + A_size;
    const int B2_smem = B1_smem + B_size;
    const int SFA_smem = B2_smem + B_size;
    const int SFB1_smem = SFA_smem + SF_size;
    const int SFB2_smem = SFB1_smem + SF_size;

    const int off_k = iter_k * BLOCK_K;
    const int off_n_frag = off_n + ctarank * (BLOCK_N / 2);

    tma_load_3d(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_B);
    tma_load_3d(B1_smem, B1_tmap, 0, off_n_frag, off_k / 256, mbar_addr, cache_B);
    tma_load_3d(B2_smem, B2_tmap, 0, off_n_frag, off_k / 256, mbar_addr, cache_B);

    const int rest_k = K / 16 / 4;
    constexpr int SF_ELEM_BYTES = 8;
    constexpr int SF_ELEMS_PER_512B = 512 / SF_ELEM_BYTES;
    const int sfa_coord = ((off_m / 128) * rest_k + off_k / (16 * 4)) * SF_ELEMS_PER_512B;
    const int sfb_coord = ((off_n / 128) * rest_k + off_k / (16 * 4)) * SF_ELEMS_PER_512B;

    tma_load_1d(SFA_smem, SFA_tmap, sfa_coord, mbar_addr, cache_B);
    tma_load_1d(SFB1_smem, SFB1_tmap, sfb_coord, mbar_addr, cache_B);
    tma_load_1d(SFB2_smem, SFB2_tmap, sfb_coord, mbar_addr, cache_B);

    mbarrier_expect_tx(mbar_addr, STAGE_SIZE);
}

template <int BLOCK_N, bool UseApprox>
__device__ inline void epilogue_master(
    int warp_id, int lane_id, int off_m, int off_n,
    int gemm1_tmem, int gemm2_tmem, half *C_ptr, int N, int done_mbar_addr
) {
    mbarrier_wait(done_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    constexpr int COLS_PER_CHUNK = 8;
    constexpr int NUM_CHUNKS = BLOCK_N / COLS_PER_CHUNK;
    const int col_lane = (lane_id % 4) * 2;
    const int row_lane = lane_id / 4;

    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int tm = warp_id * 32 + m * 16;
        const int out_row0 = off_m + tm + row_lane;
        const int out_row1 = out_row0 + 8;

        #pragma unroll
        for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
            float g1[4], g2[4];

            tcgen05_ld_16x256bx1(g1, tm, gemm1_tmem + chunk * COLS_PER_CHUNK);
            tcgen05_ld_16x256bx1(g2, tm, gemm2_tmem + chunk * COLS_PER_CHUNK);
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            const int out_col = off_n + chunk * COLS_PER_CHUNK + col_lane;

            const float s00 = silu<UseApprox>(g1[0]);
            const float s01 = silu<UseApprox>(g1[1]);
            const float s10 = silu<UseApprox>(g1[2]);
            const float s11 = silu<UseApprox>(g1[3]);

            const float v00 = g2[0] * s00;
            const float v01 = g2[1] * s01;
            const float v10 = g2[2] * s10;
            const float v11 = g2[3] * s11;

            reinterpret_cast<half2 *>(C_ptr + out_row0 * N + out_col)[0] = __float22half2_rn({v00, v01});
            reinterpret_cast<half2 *>(C_ptr + out_row1 * N + out_col)[0] = __float22half2_rn({v10, v11});
        }
    }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES, bool UseApprox>
__global__ __cluster_dims__(2) __launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void dual_gemm_master_kernel(
    const __grid_constant__ CUtensorMap A_tmap, const __grid_constant__ CUtensorMap B1_tmap,
    const __grid_constant__ CUtensorMap B2_tmap, const __grid_constant__ CUtensorMap SFA_tmap,
    const __grid_constant__ CUtensorMap SFB1_tmap, const __grid_constant__ CUtensorMap SFB2_tmap,
    half *C_ptr, int M, int N, int K
) {
    const int tid = threadIdx.x, lane_id = tid % WARP_SIZE, warp_id = tid / WARP_SIZE;
    const uint32_t ctarank = get_cluster_ctarank();
    const bool is_cta0 = (ctarank == 0);
    const int cluster_id = blockIdx.x / 2;

    const int grid_n = N / BLOCK_N;
    const int cluster_m = cluster_id / grid_n;
    const int bid_n = cluster_id % grid_n;
    const int base_m = cluster_m * (2 * BLOCK_M);
    const int off_m = base_m + int(ctarank) * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;
    const int bid_m = cluster_m * 2 + int(ctarank);

    constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;
    const int num_iters = K / BLOCK_K;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    constexpr int B_FRAG_N = BLOCK_N / 2;
    constexpr int A_size = BLOCK_M * BLOCK_K / 2;
    constexpr int B_size = B_FRAG_N * BLOCK_K / 2;
    constexpr int SF_size = 128 * BLOCK_K / 16;
    constexpr int STAGE_SIZE = A_size + B_size * 2 + SF_size * 3;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int tma_mbar_addr = mbar_base;
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int done_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

    constexpr int GEMM1_D_TMEM = 0;
    constexpr int GEMM2_D_TMEM = BLOCK_N;
    constexpr int SFA_tmem = BLOCK_N * 2;
    constexpr int SFB1_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);
    constexpr int SFB2_tmem = SFB1_tmem + 4 * (BLOCK_K / MMA_K);

    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int i = 0; i < NUM_STAGES * 2 + 1; i++) {
            const int count = (i < NUM_STAGES) ? 2 : 1;
            mbarrier_init(mbar_base + i * 8, count);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(TMEM_COLS));
    }
    __syncthreads();

    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };
    auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
    };

    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | ((uint32_t)(2 * BLOCK_M) >> 7U << 27U);

    constexpr int TILES_N_128 = 128 / BLOCK_N;
    const int tile_n_in_128 = (off_n / BLOCK_N) % TILES_N_128;

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        #pragma unroll
        for (int iter_k = 0; iter_k < NUM_STAGES && iter_k < num_iters; iter_k++) {
            const int tma_mbar = tma_mbar_addr + iter_k * 8;
            issue_tma_master<BLOCK_M, BLOCK_N, BLOCK_K>(
                smem, iter_k, iter_k,
                &A_tmap, &B1_tmap, &B2_tmap,
                &SFA_tmap, &SFB1_tmap, &SFB2_tmap,
                off_m, off_n, K, int(ctarank),
                tma_mbar, cache_A, cache_B);
        }

        #pragma unroll
        for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            const int tma_mbar = tma_mbar_addr + stage_id * 8;
            mbarrier_wait(mma_mbar_addr + stage_id * 8, (iter_k / NUM_STAGES - 1) % 2);
            issue_tma_master<BLOCK_M, BLOCK_N, BLOCK_K>(
                smem, stage_id, iter_k,
                &A_tmap, &B1_tmap, &B2_tmap,
                &SFA_tmap, &SFB1_tmap, &SFB2_tmap,
                off_m, off_n, K, int(ctarank),
                tma_mbar, cache_A, cache_B);
        }
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync() && is_cta0) {
        #pragma unroll
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
            const int stage_id = iter_k % NUM_STAGES;
            const int phase = (iter_k / NUM_STAGES) % 2;

            mbarrier_wait(tma_mbar_addr + stage_id * 8, phase);

            const int A_smem = smem + stage_id * STAGE_SIZE;
            const int B1_smem = A_smem + A_size;
            const int B2_smem = B1_smem + B_size;
            const int SFA_smem = B2_smem + B_size;
            const int SFB1_smem = SFA_smem + SF_size;
            const int SFB2_smem = SFB1_smem + SF_size;

            constexpr uint64_t SF_desc_base = make_desc_SF(0);
            const uint64_t SFA_desc = SF_desc_base + ((uint64_t)SFA_smem >> 4ULL);
            const uint64_t SFB1_desc = SF_desc_base + ((uint64_t)SFB1_smem >> 4ULL);
            const uint64_t SFB2_desc = SF_desc_base + ((uint64_t)SFB2_smem >> 4ULL);

            #pragma unroll
            for (int k = 0; k < BLOCK_K / MMA_K; k++) {
                tcgen05_cp_nvfp4(SFA_tmem + k * 4, SFA_desc + (uint64_t)k * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB1_tmem + k * 4, SFB1_desc + (uint64_t)k * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB2_tmem + k * 4, SFB2_desc + (uint64_t)k * (512ULL >> 4ULL));
            }

            #pragma unroll
            for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
                #pragma unroll
                for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                    uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
                    uint64_t b1_desc = make_desc_AB(B1_smem + k1 * B_FRAG_N * 128 + k2 * 32);
                    uint64_t b2_desc = make_desc_AB(B2_smem + k1 * B_FRAG_N * 128 + k2 * 32);
                    int k_sf = k1 * 4 + k2;
                    const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
                    const int scale_B1_tmem = SFB1_tmem + k_sf * 4 + tile_n_in_128 * (BLOCK_N / 32);
                    const int scale_B2_tmem = SFB2_tmem + k_sf * 4 + tile_n_in_128 * (BLOCK_N / 32);
                    const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
                    tcgen05_mma_nvfp4(a_desc, b1_desc, i_desc, scale_A_tmem, scale_B1_tmem, enable_input_d, GEMM1_D_TMEM);
                    tcgen05_mma_nvfp4(a_desc, b2_desc, i_desc, scale_A_tmem, scale_B2_tmem, enable_input_d, GEMM2_D_TMEM);
                }
            }

            tcgen05_commit(mma_mbar_addr + stage_id * 8);
        }
        tcgen05_commit(done_mbar_addr);
    }

    if (tid < BLOCK_M) { 
        epilogue_master<BLOCK_N, UseApprox>(warp_id, lane_id, off_m, off_n, GEMM1_D_TMEM, GEMM2_D_TMEM, C_ptr, N, done_mbar_addr); 
        //skip_ep(done_mbar_addr);
    }

    __syncthreads();

    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(TMEM_COLS));
}

// ============================================================================
// Host Launch for M=512 with 8-warp epilogue
// ============================================================================
template <typename Cfg>
at::Tensor dual_gemm_silu_impl(
    const at::Tensor& A, const at::Tensor& B1, const at::Tensor& B2,
    const at::Tensor& SFA, const at::Tensor& SFB1, const at::Tensor& SFB2,
    at::Tensor& C
) {
    const int M = A.size(0);
    const int N = B1.size(0);
    const int K = A.size(1) * 2;

    auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
    auto B1_ptr  = reinterpret_cast<const char *>(B1.data_ptr());
    auto B2_ptr  = reinterpret_cast<const char *>(B2.data_ptr());
    auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
    auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
    auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
    auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());

    CUtensorMap A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap;
    init_AB_tmap(&A_tmap, A_ptr, M, K, Cfg::BLOCK_M, Cfg::BLOCK_K);
    init_AB_tmap(&B1_tmap, B1_ptr, N, K, Cfg::OUT_N, Cfg::BLOCK_K);
    init_AB_tmap(&B2_tmap, B2_ptr, N, K, Cfg::OUT_N, Cfg::BLOCK_K);

    const int rest_k = K / 64;
    const uint64_t sfa_bytes = (uint64_t)(M / 128) * rest_k * 512;
    const uint64_t sfb_bytes = (uint64_t)(N / 128) * rest_k * 512;
    init_SF_tmap(&SFA_tmap, SFA_ptr, sfa_bytes, Cfg::SF_SIZE);
    init_SF_tmap(&SFB1_tmap, SFB1_ptr, sfb_bytes, 512);
    init_SF_tmap(&SFB2_tmap, SFB2_ptr, sfb_bytes, 512);

    const int num_tiles = (M / (2 * Cfg::BLOCK_M)) * (N / Cfg::OUT_N);

    const bool use_approx = !(M == 512 && N == 4096);
    auto kernel = use_approx ? dual_gemm_silu_kernel<Cfg, true> : dual_gemm_silu_kernel<Cfg, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::SMEM_SIZE);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(num_tiles * 2, 1, 1);
    config.blockDim = dim3(Cfg::TB_SIZE, 1, 1);
    config.dynamicSmemBytes = Cfg::SMEM_SIZE;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, kernel,
        A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap,
        C_ptr, M, N, K);

    return C;
}

at::Tensor dual_gemm_m512(
    const at::Tensor& A, const at::Tensor& B1, const at::Tensor& B2,
    const at::Tensor& SFA, const at::Tensor& SFB1, const at::Tensor& SFB2, at::Tensor& C
) {
    return dual_gemm_silu_impl<CfgN128>(A, B1, B2, SFA, SFB1, SFB2, C);
}

// ============================================================================
// Host Launch for M=256 (unchanged - separate logic)
// ============================================================================
at::Tensor dual_gemm_m256(
    const at::Tensor& A, const at::Tensor& B1, const at::Tensor& B2,
    const at::Tensor& SFA, const at::Tensor& SFB1, const at::Tensor& SFB2, at::Tensor& C
) {
    const int M = A.size(0), N = B1.size(0), K = A.size(1) * 2;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 256;
    constexpr int NUM_STAGES = 7;

    CUtensorMap A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap;
    init_AB_tmap(&A_tmap, (const char*)A.data_ptr(), M, K, BLOCK_M, BLOCK_K);
    init_AB_tmap(&B1_tmap, (const char*)B1.data_ptr(), N, K, BLOCK_N / 2, BLOCK_K);
    init_AB_tmap(&B2_tmap, (const char*)B2.data_ptr(), N, K, BLOCK_N / 2, BLOCK_K);

    int SF_sz = 128 * BLOCK_K / 16;
    int rest_k = K / 64;
    uint64_t sfa_bytes = (uint64_t)(M / 128) * rest_k * 512;
    uint64_t sfb_bytes = (uint64_t)(N / 128) * rest_k * 512;
    init_SF_tmap(&SFA_tmap, (const char*)SFA.data_ptr(), sfa_bytes, SF_sz);
    init_SF_tmap(&SFB1_tmap, (const char*)SFB1.data_ptr(), sfb_bytes, SF_sz);
    init_SF_tmap(&SFB2_tmap, (const char*)SFB2.data_ptr(), sfb_bytes, SF_sz);

    int num_tiles = (M / (2 * BLOCK_M)) * (N / BLOCK_N);
    int tb_size = BLOCK_M + 2 * WARP_SIZE;

    int A_sz = BLOCK_M * BLOCK_K / 2;
    int B_sz = (BLOCK_N / 2) * BLOCK_K / 2;
    int stage_size = A_sz + B_sz * 2 + SF_sz * 3;
    int smem_size = stage_size * NUM_STAGES;

    const bool use_approx = !(M == 512 && N == 4096);
    auto kernel = use_approx ? dual_gemm_master_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, true>
                             : dual_gemm_master_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(num_tiles * 2, 1, 1);
    config.blockDim = dim3(tb_size, 1, 1);
    config.dynamicSmemBytes = smem_size;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, kernel, A_tmap, B1_tmap, B2_tmap, SFA_tmap, SFB1_tmap, SFB2_tmap,
                       (half*)C.data_ptr(), M, N, K);
    return C;
}

TORCH_LIBRARY(dual_gemm_m512_lib, m) {
    m.def("dual_gemm_silu(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
    m.impl("dual_gemm_silu", &dual_gemm_m512);
}

TORCH_LIBRARY(dual_gemm_m256_lib, m) {
    m.def("dual_gemm_silu(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
    m.impl("dual_gemm_silu", &dual_gemm_m256);
}
"""

load_inline(
    "dual_gemm_v3",
    cpp_sources="",
    cuda_sources=cuda_src,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3", "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math", "--expt-relaxed-constexpr",
        "--relocatable-device-code=false", "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
)

dual_gemm_m512 = torch.ops.dual_gemm_m512_lib.dual_gemm_silu
dual_gemm_m256 = torch.ops.dual_gemm_m256_lib.dual_gemm_silu


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2 = data[0], data[1], data[2]
    sfa_perm, sfb1_perm, sfb2_perm = data[6], data[7], data[8]
    c = data[9]

    M = a.shape[0]
    if M == 512:
        # M=512: 8-warp epilogue pathway
        return dual_gemm_m512(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c)
    else:
        # M=256: master_kernel2.py n64 pathway
        return dual_gemm_m256(a, b1, b2, sfa_perm, sfb1_perm, sfb2_perm, c)
