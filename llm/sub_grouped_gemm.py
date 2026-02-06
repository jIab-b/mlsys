
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
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 256;
constexpr int A_SIZE  = BLOCK_M * BLOCK_K / 2;   // 16384
constexpr int B_SIZE  = BLOCK_N * BLOCK_K / 2;   // 16384
constexpr int SFA_SIZE = 128 * BLOCK_K / 16;     // 2048
constexpr int SFB_SIZE = 128 * BLOCK_K / 16;     // 2048
constexpr int MAIN_STAGE = A_SIZE + B_SIZE;       // 32768
constexpr int SF_STAGE   = SFA_SIZE + SFB_SIZE;   // 4096
constexpr int TMAP_SMEM  = 4 * 128;              // 512

constexpr int NUM_STAGES = 5;
constexpr int SMEM_SIZE  = TMAP_SMEM + MAIN_STAGE * NUM_STAGES + SF_STAGE * NUM_STAGES;
constexpr int NUM_MBAR = NUM_STAGES * 2 + 1;      // 11

constexpr int NUM_EP_WARPS = 4;
constexpr int NUM_WARPS = NUM_EP_WARPS + 2;       // 6
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;     // 192
constexpr int TMEM_COLS = 512;

constexpr int D_TMEM   = 0;
constexpr int SFA_TMEM = BLOCK_N;                 // 128
constexpr int SFB_TMEM = SFA_TMEM + 4 * (BLOCK_K / MMA_K); // 144

constexpr uint32_t I_DESC = (1U << 7U) | (1U << 10U) |
    ((uint32_t)BLOCK_N >> 3U << 17U) | ((uint32_t)BLOCK_M >> 7U << 27U);

constexpr int MAX_GROUPS = 8;

// ============================================================================
// Device structures
// ============================================================================
struct GroupInfo {
    half* c_ptr;
    int M, N, K;
    int tile_offset;
    int m_tiles, n_tiles;
};

struct KernelParams {
    GroupInfo groups[MAX_GROUPS];
    int num_groups;
};

// ============================================================================
// Inline PTX helpers
// ============================================================================
__device__ inline constexpr uint64_t desc_encode(uint64_t x) {
    return (x & 0x3'FFFFULL) >> 4ULL;
}

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

__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\\n\\t"
        ".reg .pred P1;\\n\\t"
        "LAB_WAIT:\\n\\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\\n\\t"
        "@P1 bra.uni DONE;\\n\\t"
        "bra.uni LAB_WAIT;\\n\\t"
        "DONE:\\n\\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline void tma_load_1d(int dst, const void *tmap_ptr, int x, int mbar_addr, uint64_t cache_policy) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tmap_ptr);
    asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                 "[%0], [%1, {%3}], [%2], %4;"
                 :: "r"(dst), "l"(gmem_int_desc), "r"(mbar_addr), "r"(x), "l"(cache_policy) : "memory");
}

__device__ inline void tma_load_3d(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tmap_ptr);
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                 "[%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :: "r"(dst), "l"(gmem_int_desc), "r"(mbar_addr),
                    "r"(x), "r"(y), "r"(z), "l"(cache_policy) : "memory");
}

__device__ inline void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline void tcgen05_mma_nvfp4(uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int scale_A_tmem, int scale_B_tmem, int enable_input_d, int d_tmem) {
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %6, 0;\\n\\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d));
}

__device__ inline void tcgen05_commit(int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_addr) : "memory");
}

static constexpr char SHAPE_16x256b[] = ".16x256b";
static constexpr char NUM_x1[] = ".x1";

template <const char *SHAPE, const char *NUM>
__device__ inline void tcgen05_ld_4regs(float *tmp, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned%5%6.b32 "
        "{ %0, %1, %2, %3 }, [%4];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
        : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

__device__ inline void tcgen05_ld_16x256bx1(float *tmp, int row, int col) {
    tcgen05_ld_4regs<SHAPE_16x256b, NUM_x1>(tmp, row, col);
}

__device__ inline void fence_proxy_tensormap(const void *smem_ptr) {
    uint64_t addr = reinterpret_cast<uint64_t>(smem_ptr);
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(addr));
}

__device__ inline void do_epilogue(int warp_id, int lane_id, int done_mbar,
    int M, int N, int off_m, int off_n, const GroupInfo& gi) {
    mbarrier_wait(done_mbar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int col_lane = (lane_id % 4) * 2;
    const int row_lane = lane_id / 4;
    const int residue_m = M - off_m;

    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const int tm = warp_id * 32 + m * 16;
        const int out_row0 = off_m + tm + row_lane;
        const int out_row1 = out_row0 + 8;

        #pragma unroll
        for (int chunk = 0; chunk < BLOCK_N / 8; chunk++) {
            float vals[4];
            tcgen05_ld_16x256bx1(vals, tm, D_TMEM + chunk * 8);
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            const int out_col = off_n + chunk * 8 + col_lane;

            if (tm + row_lane < residue_m) {
                reinterpret_cast<half2 *>(gi.c_ptr + out_row0 * N + out_col)[0] =
                    __float22half2_rn({vals[0], vals[1]});
            }
            if (tm + row_lane + 8 < residue_m) {
                reinterpret_cast<half2 *>(gi.c_ptr + out_row1 * N + out_col)[0] =
                    __float22half2_rn({vals[2], vals[3]});
            }
        }
    }
}

// ============================================================================
// TensorMap Initialization
// ============================================================================
void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *msg;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unknown";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", msg);
}

void init_AB_tmap(CUtensorMap *tmap, const char *ptr, uint64_t height, uint64_t width,
                  uint32_t box_h, uint32_t box_w) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank] = {256, height, width / 256};
    uint64_t globalStrides[rank - 1] = {width / 2, 128};
    uint32_t boxDim[rank] = {256, box_h, box_w / 256};
    uint32_t elementStrides[rank] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// SF reordered tensors have logical shape [32, 4, rest_m, 4, rest_k, L] but are
// a permuted view of a contiguous [L, rest_m, rest_k, 32, 4, 4] allocation.
// Physical memory is thus [rest_m][rest_k][512 bytes], i.e. each 512-byte SF tile
// (covering 128 M-rows x 1 MMA_K=64 step) is already contiguous.
// We encode this as a 3D TMA: dim0 = 256 uint16 (=512B block), dim1 = mn_blocks, dim2 = k_blocks.
void init_SF_tmap(CUtensorMap *tmap, const char *ptr, uint64_t mn, uint64_t K) {
    constexpr uint32_t rank = 3;
    const uint64_t k_blocks = K / 64;
    const uint64_t mn_blocks = (mn + 127) / 128;
    const uint32_t tile_k_blocks = BLOCK_K / 64;          // 4
    constexpr uint64_t SF_BLOCK_BYTES = 512;
    constexpr uint64_t X_ELEMS = SF_BLOCK_BYTES / sizeof(uint16_t);  // 256
    uint64_t globalDim[rank]       = {X_ELEMS, mn_blocks, k_blocks};
    uint64_t globalStrides[rank-1] = {k_blocks * SF_BLOCK_BYTES, SF_BLOCK_BYTES};
    uint32_t boxDim[rank]          = {(uint32_t)X_ELEMS, 1, tile_k_blocks};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_UINT16, rank, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ============================================================================
// Kernel
// ============================================================================
__global__ __launch_bounds__(TB_SIZE)
void grouped_gemm_kernel(
    const __grid_constant__ KernelParams params,
    const CUtensorMap* __restrict__ d_tmaps
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // --- Derive group and tile from blockIdx.x ---
    const int bid = blockIdx.x;
    int gidx = 0;
    #pragma unroll
    for (int g = 1; g < MAX_GROUPS; g++) {
        if (g < params.num_groups && bid >= params.groups[g].tile_offset)
            gidx = g;
    }
    const GroupInfo& gi = params.groups[gidx];
    const int local_tile = bid - gi.tile_offset;
    const int coord_x = local_tile % gi.m_tiles;
    const int coord_y = local_tile / gi.m_tiles;
    const int M = gi.M, N = gi.N, K = gi.K;
    const int num_k = K / BLOCK_K;
    const int off_m = coord_x * BLOCK_M;
    const int off_n = coord_y * BLOCK_N;

    // --- SMEM setup ---
    extern __shared__ __align__(1024) char smem_raw[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    const int smem_main = smem + TMAP_SMEM;
    const int smem_sf   = smem_main + MAIN_STAGE * NUM_STAGES;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbars[NUM_MBAR];
    __shared__ int32_t tmem_alloc_buf;
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int tma_mbar  = mbar_base;
    const int mma_mbar  = tma_mbar + NUM_STAGES * 8;
    const int done_mbar = mma_mbar + NUM_STAGES * 8;

    // --- INIT: mbarriers + TMEM alloc + tmap copy ---
    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int i = 0; i < NUM_MBAR; i++)
            mbarrier_init(mbar_base + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 1) {
        int alloc_addr = static_cast<int>(__cvta_generic_to_shared(&tmem_alloc_buf));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(alloc_addr), "r"(TMEM_COLS));
    }

    __syncthreads();

    // Tmap pointers in global memory (tensor maps must reside in .param/.const/.global)
    const CUtensorMap *g_tmaps = d_tmaps + gidx * 4;
    const void *A_tmap   = static_cast<const void *>(g_tmaps + 0);
    const void *B_tmap   = static_cast<const void *>(g_tmaps + 1);
    const void *SFA_tmap = static_cast<const void *>(g_tmaps + 2);
    const void *SFB_tmap = static_cast<const void *>(g_tmaps + 3);

    // --- Descriptor helpers ---
    auto make_desc_AB = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };
    auto make_desc_SF = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 16) << 32ULL) | (1ULL << 46ULL);
    };

    constexpr int SF_K_PER_BLOCK = BLOCK_K / 64;  // 4

    // ========================================================================
    // TMA Producer Warp (warp 4)
    // ========================================================================
    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        // Prefill
        #pragma unroll
        for (int ik = 0; ik < NUM_STAGES && ik < num_k; ik++) {
            int s = ik;
            int A_s = smem_main + s * MAIN_STAGE;
            int B_s = A_s + A_SIZE;
            int SFA_s = smem_sf + s * SF_STAGE;
            int SFB_s = SFA_s + SFA_SIZE;

            tma_load_3d(A_s, A_tmap, 0, off_m, ik, tma_mbar + s * 8, 0);
            tma_load_3d(B_s, B_tmap, 0, off_n, ik, tma_mbar + s * 8, 0);

            int z_sf = ik * SF_K_PER_BLOCK;
            tma_load_3d(SFA_s, SFA_tmap, 0, coord_x, z_sf, tma_mbar + s * 8, 0);
            tma_load_3d(SFB_s, SFB_tmap, 0, coord_y, z_sf, tma_mbar + s * 8, 0);

            mbarrier_arrive_expect_tx(tma_mbar + s * 8, MAIN_STAGE + SF_STAGE);
        }

        // Steady state
        for (int ik = NUM_STAGES; ik < num_k; ik++) {
            int s = ik % NUM_STAGES;
            mbarrier_wait(mma_mbar + s * 8, (ik / NUM_STAGES - 1) % 2);

            int A_s = smem_main + s * MAIN_STAGE;
            int B_s = A_s + A_SIZE;
            int SFA_s = smem_sf + s * SF_STAGE;
            int SFB_s = SFA_s + SFA_SIZE;

            tma_load_3d(A_s, A_tmap, 0, off_m, ik, tma_mbar + s * 8, 0);
            tma_load_3d(B_s, B_tmap, 0, off_n, ik, tma_mbar + s * 8, 0);

            int z_sf = ik * SF_K_PER_BLOCK;
            tma_load_3d(SFA_s, SFA_tmap, 0, coord_x, z_sf, tma_mbar + s * 8, 0);
            tma_load_3d(SFB_s, SFB_tmap, 0, coord_y, z_sf, tma_mbar + s * 8, 0);

            mbarrier_arrive_expect_tx(tma_mbar + s * 8, MAIN_STAGE + SF_STAGE);
        }
    }

    // ========================================================================
    // MMA Consumer Warp (warp 5)
    // ========================================================================
    if (warp_id == NUM_WARPS - 1 && elect_sync()) {
        #pragma unroll 1
        for (int ik = 0; ik < num_k; ik++) {
            int s = ik % NUM_STAGES;
            mbarrier_wait(tma_mbar + s * 8, (ik / NUM_STAGES) % 2);

            int A_s   = smem_main + s * MAIN_STAGE;
            int B_s   = A_s + A_SIZE;
            int SFA_s = smem_sf + s * SF_STAGE;
            int SFB_s = SFA_s + SFA_SIZE;

            // Copy scale factors smem -> tmem
            constexpr uint64_t sf_base = desc_encode(0) | (desc_encode(8 * 16) << 32ULL) | (1ULL << 46ULL);
            uint64_t sfa_desc = sf_base + ((uint64_t)SFA_s >> 4ULL);
            uint64_t sfb_desc = sf_base + ((uint64_t)SFB_s >> 4ULL);

            #pragma unroll
            for (int k = 0; k < BLOCK_K / MMA_K; k++) {
                tcgen05_cp_nvfp4(SFA_TMEM + k * 4, sfa_desc + (uint64_t)k * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB_TMEM + k * 4, sfb_desc + (uint64_t)k * (512ULL >> 4ULL));
            }

            // MMA: BLOCK_K=256 = 1 × 256, so k1=0 only, k2=0..3
            #pragma unroll
            for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                uint64_t a_desc = make_desc_AB(A_s + k2 * 32);
                uint64_t b_desc = make_desc_AB(B_s + k2 * 32);
                int enable_d = (ik == 0 && k2 == 0) ? 0 : 1;
                tcgen05_mma_nvfp4(a_desc, b_desc, I_DESC,
                    SFA_TMEM + k2 * 4, SFB_TMEM + k2 * 4, enable_d, D_TMEM);
            }

            tcgen05_commit(mma_mbar + s * 8);
        }
        tcgen05_commit(done_mbar);
    }

    // ========================================================================
    // Epilogue: warps 0-3, f32 -> f16, predicated store
    // ========================================================================
    if (warp_id < NUM_EP_WARPS) {
        do_epilogue(warp_id, lane_id, done_mbar, M, N, off_m, off_n, gi);
        mbarrier_wait(done_mbar, 0);
    }

    __syncthreads();

    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(TMEM_COLS));
}

// ============================================================================
// Host launch
// ============================================================================
static CUtensorMap* s_d_tmaps = nullptr;

void grouped_gemm_impl(
    at::TensorList A_list,
    at::TensorList B_list,
    at::TensorList C_list,
    at::TensorList SFA_list,
    at::TensorList SFB_list
) {
    int G = A_list.size();

    CUtensorMap h_tmaps[MAX_GROUPS * 4];
    KernelParams params = {};
    params.num_groups = G;

    int total_tiles = 0;
    for (int g = 0; g < G; g++) {
        int Mi = A_list[g].size(0);
        int Ki = A_list[g].size(1) * 2;
        int Ni = B_list[g].size(0);

        init_AB_tmap(&h_tmaps[g * 4 + 0], (const char *)A_list[g].data_ptr(), Mi, Ki, BLOCK_M, BLOCK_K);
        init_AB_tmap(&h_tmaps[g * 4 + 1], (const char *)B_list[g].data_ptr(), Ni, Ki, BLOCK_N, BLOCK_K);

        init_SF_tmap(&h_tmaps[g * 4 + 2], (const char *)SFA_list[g].data_ptr(), Mi, Ki);
        init_SF_tmap(&h_tmaps[g * 4 + 3], (const char *)SFB_list[g].data_ptr(), Ni, Ki);

        int mt = (Mi + BLOCK_M - 1) / BLOCK_M;
        int nt = (Ni + BLOCK_N - 1) / BLOCK_N;
        params.groups[g] = {(half *)C_list[g].data_ptr(), Mi, Ni, Ki, total_tiles, mt, nt};
        total_tiles += mt * nt;
    }

    if (!s_d_tmaps) cudaMalloc(&s_d_tmaps, MAX_GROUPS * 4 * sizeof(CUtensorMap));
    cudaMemcpy(s_d_tmaps, h_tmaps, G * 4 * sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    auto kernel = grouped_gemm_kernel;
    static int smem_size = 0;
    if (!smem_size) {
        int dev; cudaGetDevice(&dev);
        int smem_max;
        cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        smem_size = smem_max - 1024;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    }
    kernel<<<total_tiles, TB_SIZE, smem_size>>>(params, s_d_tmaps);
}

TORCH_LIBRARY(gg, m) {
    m.def("run(Tensor[] A, Tensor[] B, Tensor[] C, Tensor[] SFA, Tensor[] SFB) -> ()");
    m.impl("run", &grouped_gemm_impl);
}
"""

load_inline(
    "grouped_gemm_v1",
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

_run = torch.ops.gg.run

def custom_kernel(data: input_t) -> output_t:
    # data = (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
    # sfasfb_reordered has logical shape [32, 4, rest_m, 4, rest_k, L] but is a permuted
    # view of contiguous [L, rest_m, rest_k, 32, 4, 4]. Physical memory is already
    # [rest_m][rest_k][512B tiles] — no host-side permute/contiguous needed.
    abc, _, sf_reordered, _ = data
    a, b, c = zip(*abc)
    sfa, sfb = zip(*sf_reordered)
    _run(list(a), list(b), list(c), list(sfa), list(sfb))
    return list(c)
