#!POPCORN gpu NVIDIA

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# CTA-copy no-ep variant aligned to cta_full control flow.
cuda_src = """
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <cstdint>

// ============================================================================
// Constants
// ============================================================================
constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;

constexpr uint64_t EVICT_NORMAL = 0x1000000000000000ULL;
constexpr uint64_t EVICT_FIRST  = 0x12F0000000000000ULL;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 256;
constexpr int A_SIZE  = BLOCK_M * BLOCK_K / 2;   // 16384
constexpr int B_SIZE  = BLOCK_N * BLOCK_K / 2;   // 16384
constexpr int B_HALF  = BLOCK_N * BLOCK_K / 4;   // 8192 (per CTA in cta_group::2)
constexpr int SFA_SIZE = 128 * BLOCK_K / 16;     // 2048
constexpr int SFB_SIZE = 128 * BLOCK_K / 16;     // 2048
constexpr int SF_STAGE   = SFA_SIZE + SFB_SIZE;   // 4096
constexpr int EP_STRIDE = 136;                    // padded stride for chunk32 epilogue
constexpr int EP_SMEM_BYTES = 32 * EP_STRIDE * (int)sizeof(half); // 8704B scratch

constexpr int NUM_EP_WARPS = 4;
constexpr int NUM_WARPS = NUM_EP_WARPS + 2;       // 6
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;     // 192
constexpr int TMEM_COLS = 512;
constexpr int MAX_LAUNCH_CTAS = 148;
constexpr int MAX_CLUSTERS = MAX_LAUNCH_CTAS / 2;
constexpr int MAX_TILE_LUT = 1024;

constexpr int D_TMEM0  = 0;
constexpr int D_TMEM1  = BLOCK_N;                 // 128
constexpr int SFA_TMEM = 2 * BLOCK_N;             // 256
constexpr int SFB_TMEM = SFA_TMEM + 4 * (BLOCK_K / MMA_K); // 272

constexpr int MAX_GROUPS = 8;
constexpr int MAX_NS = 10;  // maximum pipeline stages

// ============================================================================
// Device structures
// ============================================================================
struct GroupInfo {
    half* c_ptr;
    uint16_t M, N, K, _pad;
};

// Packed LUT: each tile is encoded in two words.
// lut_meta0 bits:
//   [2:0]    gidx
//   [12:3]   coord_m
//   [22:13]  coord_n_pair (256-wide N-pair tile index)
//   [30:23]  tile_num_k
//   [31]     rank1_aliased
// lut_meta1 bits:
//   [3:0]    mma_n_q16
//   [4]      a_tmap_r0
//   [5]      a_tmap_r1
//   [7:6]    b_tmap_r0
//   [9:8]    b_tmap_r1
//   [10]     off_m_b_r1_is_base
//   [18:11]  expect_r0_q128
//   [26:19]  expect_r1_q128
struct KernelParams {
    GroupInfo groups[MAX_GROUPS];
    int num_groups;
    int total_tiles;
    int launch_ctas;
    int smem_size_bytes;
    int ns;          // pipeline stages (smem-driven; independent of num_k divisibility)
    int main_stg;    // max stage size for smem budgeting / NS selection

    uint16_t lut_worker_start[MAX_CLUSTERS];
    uint16_t lut_worker_count[MAX_CLUSTERS];

    uint32_t lut_meta0[MAX_TILE_LUT];
    uint32_t lut_meta1[MAX_TILE_LUT];
};

enum : int {
    SCHED_BASE = 0,
    SCHED_REV = 1,
    SCHED_F2_G2 = 2,
    SCHED_F1_G1 = 3,
};

enum : int {
    PROFILE_GENERIC_G8 = 0,
    PROFILE_GENERIC_G2 = 1,
    PROFILE_BENCH1 = 2,
    PROFILE_BENCH2 = 3,
    PROFILE_BENCH3 = 4,
    PROFILE_BENCH4 = 5,
};

constexpr int V_FORCE_SCHEDULE = -1;
constexpr int V_BENCH1_CLUSTERS = -1;
constexpr int V_BENCH2_CLUSTERS = -1;
constexpr int V_GENERIC_CLUSTERS = -1;

struct TmapParamPackG8 {
    CUtensorMap A_full[MAX_GROUPS];  // orig B data (MMA A operand), box_h=128
    CUtensorMap A_tail[MAX_GROUPS];  // orig B data, N-tail
    CUtensorMap B_full[MAX_GROUPS];  // orig A data (MMA B operand), box_h=64 per CTA
    CUtensorMap B_tail0[MAX_GROUPS]; // orig A data, M-tail for CTA0
    CUtensorMap B_tail1[MAX_GROUPS]; // orig A data, M-tail for CTA1
    CUtensorMap SFA[MAX_GROUPS];     // orig SFB (scale for MMA A)
    CUtensorMap SFB[MAX_GROUPS];     // orig SFA (scale for MMA B)
};

// ============================================================================
// Inline PTX helpers
// ============================================================================
__host__ __device__ inline uint32_t pack_tile_meta0(
    uint32_t gidx, uint32_t coord_m, uint32_t coord_n_pair, uint32_t tile_num_k, uint32_t rank1_aliased
) {
    return (gidx & 0x7U)
         | ((coord_m & 0x3FFU) << 3U)
         | ((coord_n_pair & 0x3FFU) << 13U)
         | ((tile_num_k & 0xFFU) << 23U)
         | ((rank1_aliased & 0x1U) << 31U);
}

__host__ __device__ inline uint32_t pack_tile_meta1(
    uint32_t mma_n_q16, uint32_t a_tmap_r0, uint32_t a_tmap_r1,
    uint32_t b_tmap_r0, uint32_t b_tmap_r1, uint32_t off_m_b_r1_is_base,
    uint32_t expect_r0_q128, uint32_t expect_r1_q128
) {
    return (mma_n_q16 & 0xFU)
         | ((a_tmap_r0 & 0x1U) << 4U)
         | ((a_tmap_r1 & 0x1U) << 5U)
         | ((b_tmap_r0 & 0x3U) << 6U)
         | ((b_tmap_r1 & 0x3U) << 8U)
         | ((off_m_b_r1_is_base & 0x1U) << 10U)
         | ((expect_r0_q128 & 0xFFU) << 11U)
         | ((expect_r1_q128 & 0xFFU) << 19U);
}

__device__ inline void unpack_tile_meta0(
    uint32_t meta0, int &gidx, int &coord_m, int &coord_n_pair, int &tile_num_k, int &rank1_aliased
) {
    gidx = static_cast<int>(meta0 & 0x7U);
    coord_m = static_cast<int>((meta0 >> 3U) & 0x3FFU);
    coord_n_pair = static_cast<int>((meta0 >> 13U) & 0x3FFU);
    tile_num_k = static_cast<int>((meta0 >> 23U) & 0xFFU);
    rank1_aliased = static_cast<int>((meta0 >> 31U) & 0x1U);
}

__device__ inline void unpack_tile_meta1(
    uint32_t meta1, int &mma_n, int &a_tmap_r0, int &a_tmap_r1, int &b_tmap_r0, int &b_tmap_r1,
    int &off_m_b_r1_is_base, int &expect_r0, int &expect_r1
) {
    mma_n = static_cast<int>((meta1 & 0xFU) << 4U);
    a_tmap_r0 = static_cast<int>((meta1 >> 4U) & 0x1U);
    a_tmap_r1 = static_cast<int>((meta1 >> 5U) & 0x1U);
    b_tmap_r0 = static_cast<int>((meta1 >> 6U) & 0x3U);
    b_tmap_r1 = static_cast<int>((meta1 >> 8U) & 0x3U);
    off_m_b_r1_is_base = static_cast<int>((meta1 >> 10U) & 0x1U);
    expect_r0 = static_cast<int>(((meta1 >> 11U) & 0xFFU) << 7U);
    expect_r1 = static_cast<int>(((meta1 >> 19U) & 0xFFU) << 7U);
}

struct TileRuntimeInfo {
    int gidx;
    int coord_m;
    int off_n_r0, off_n_r1;
    int coord_n_r0, coord_n_r1;
    int off_m_b_r0, off_m_b_r1;
    int mma_n;
    int tile_num_k;
    int a_tmap_r0, a_tmap_r1;
    int b_tmap_r0, b_tmap_r1;
    int expect_r0, expect_r1;
};

__device__ inline void decode_tile_runtime(
    const KernelParams& params, int lut_idx, TileRuntimeInfo& t
) {
    uint32_t meta0 = params.lut_meta0[lut_idx];
    uint32_t meta1 = params.lut_meta1[lut_idx];
    int coord_n_pair = 0, rank1_aliased = 0, off_m_b_r1_is_base = 0;
    unpack_tile_meta0(meta0, t.gidx, t.coord_m, coord_n_pair, t.tile_num_k, rank1_aliased);
    unpack_tile_meta1(
        meta1, t.mma_n, t.a_tmap_r0, t.a_tmap_r1, t.b_tmap_r0, t.b_tmap_r1,
        off_m_b_r1_is_base, t.expect_r0, t.expect_r1);

    t.off_n_r0 = coord_n_pair * 256;
    t.off_n_r1 = t.off_n_r0 + ((1 - rank1_aliased) * BLOCK_M);
    t.coord_n_r0 = coord_n_pair * 2;
    t.coord_n_r1 = t.coord_n_r0 + (1 - rank1_aliased);

    const int b_half_rows = t.mma_n / 2;
    const int off_m_base = t.coord_m * BLOCK_N;
    t.off_m_b_r0 = off_m_base;
    t.off_m_b_r1 = off_m_base + ((1 - off_m_b_r1_is_base) * b_half_rows);
}

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

__device__ inline uint32_t get_cluster_ctarank() {
    uint32_t rank = 0;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x10000;
    asm volatile(
        "{\\n\\t"
        ".reg .pred P1;\\n\\t"
        "LAB_WAIT:\\n\\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\\n\\t"
        "@!P1 bra.uni LAB_WAIT;\\n\\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(mbar_addr), "r"(size) : "memory");
}

template <int CTA_GROUP = 2>
__device__ inline void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::%7.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy), "n"(CTA_GROUP)
                 : "memory");
}

template <int CTA_GROUP = 2>
__device__ inline void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::%2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 2>
__device__ inline void tcgen05_mma_nvfp4(int d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int scale_A_tmem, int scale_B_tmem, int enable_input_d) {
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %6, 0;\\n\\t"
        "tcgen05.mma.cta_group::%7.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 2>
__device__ inline void tcgen05_commit(int mbar_addr) {
    asm volatile("tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(mbar_addr), "n"(CTA_GROUP) : "memory");
}

template <int CTA_GROUP = 2>
__device__ inline void tcgen05_commit_mcast(int mbar_addr, uint16_t cta_mask) {
    asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
                 :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
}

__device__ inline void tcgen05_ld_16x256bx2(float *tmp, int row, int col) {
    asm volatile("tcgen05.ld.sync.aligned.16x256b.x2.b32 "
        "{ %0, %1, %2, %3, %4, %5, %6, %7 }, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"((row << 16) | col));
}

// Chunk32 transposed epilogue (copied from cg2_full style).
__device__ inline void do_epilogue_transposed_chunk32(
    int warp_id, int lane_id, int cta_rank,
    int done_mbar, int done_phase, int d_tmem_base,
    half* __restrict__ smem_ep, half* __restrict__ c_ptr,
    int M, int N, int off_m, int off_n, int mma_n
) {
    mbarrier_wait(done_mbar, done_phase);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int col_lane = (lane_id % 4) * 2;
    const int row_lane = lane_id / 4;
    const int tid_ep = warp_id * WARP_SIZE + lane_id;
    const int residue_n = N - off_n;
    const int num_chunks = (mma_n + 31) / 32;

    #pragma unroll 1
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        #pragma unroll
        for (int mc = 0; mc < 2; mc++) {
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                const int tm = cta_rank * BLOCK_M + warp_id * 32 + m * 16;
                float vals[8];
                tcgen05_ld_16x256bx2(vals, tm, d_tmem_base + chunk * 32 + mc * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                const int n0 = warp_id * 32 + m * 16 + row_lane;
                const int n1 = n0 + 8;
                const int m_off = mc * 16;

                smem_ep[(col_lane + m_off)       * EP_STRIDE + n0] = __float2half_rn(vals[0]);
                smem_ep[(col_lane + 1 + m_off)   * EP_STRIDE + n0] = __float2half_rn(vals[1]);
                smem_ep[(col_lane + m_off)       * EP_STRIDE + n1] = __float2half_rn(vals[2]);
                smem_ep[(col_lane + 1 + m_off)   * EP_STRIDE + n1] = __float2half_rn(vals[3]);
                smem_ep[(col_lane + 8 + m_off)   * EP_STRIDE + n0] = __float2half_rn(vals[4]);
                smem_ep[(col_lane + 9 + m_off)   * EP_STRIDE + n0] = __float2half_rn(vals[5]);
                smem_ep[(col_lane + 8 + m_off)   * EP_STRIDE + n1] = __float2half_rn(vals[6]);
                smem_ep[(col_lane + 9 + m_off)   * EP_STRIDE + n1] = __float2half_rn(vals[7]);
            }
        }

        asm volatile("bar.sync 15, %0;" :: "r"(NUM_EP_WARPS * WARP_SIZE));

        const int n_group = tid_ep % 8;
        const int n_start = n_group * 16;
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {
            const int m_row = pass * 16 + tid_ep / 8;
            const int m_local = chunk * 32 + m_row;
            const int m_global = off_m + m_local;

            if (m_local < mma_n && m_global < M && n_start < residue_n) {
                half* dst_base = &c_ptr[m_global * N + off_n + n_start];
                const half* src_base = &smem_ep[m_row * EP_STRIDE + n_start];

                if (n_start + 16 <= residue_n) {
                    *reinterpret_cast<int4*>(dst_base) = *reinterpret_cast<const int4*>(src_base);
                    *reinterpret_cast<int4*>(dst_base + 8) = *reinterpret_cast<const int4*>(src_base + 8);
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        if (n_start + i < residue_n) dst_base[i] = src_base[i];
                    }
                }
            }
        }

        asm volatile("bar.sync 15, %0;" :: "r"(NUM_EP_WARPS * WARP_SIZE));
    }
}

__device__ inline void do_epilogue_transposed(
    int warp_id, int lane_id, int cta_rank,
    int done_mbar, int done_phase, int d_tmem_base,
    half* __restrict__ smem_ep, half* __restrict__ c_ptr,
    int M, int N, int off_m, int off_n, int mma_n
) {
    do_epilogue_transposed_chunk32(
        warp_id, lane_id, cta_rank, done_mbar, done_phase, d_tmem_base,
        smem_ep, c_ptr, M, N, off_m, off_n, mma_n);
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
                  uint32_t box_h, uint32_t box_w, CUtensorMapL2promotion l2_promotion) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank] = {256, height, width / 256};
    uint64_t globalStrides[rank - 1] = {width / 2, 128};
    uint32_t boxDim[rank] = {256, box_h, box_w / 256};
    uint32_t elementStrides[rank] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        l2_promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void init_SF_tmap(CUtensorMap *tmap, const char *ptr, uint64_t mn, uint64_t K,
                  CUtensorMapL2promotion l2_promotion) {
    constexpr uint32_t rank = 3;
    const uint64_t k_blocks = K / 64;
    const uint64_t mn_blocks = (mn + 127) / 128;
    const uint32_t tile_k_blocks = BLOCK_K / 64;
    constexpr uint64_t SF_BLOCK_BYTES = 512;
    constexpr uint64_t X_ELEMS = SF_BLOCK_BYTES / sizeof(uint16_t);
    uint64_t globalDim[rank]       = {X_ELEMS, mn_blocks, k_blocks};
    uint64_t globalStrides[rank-1] = {k_blocks * SF_BLOCK_BYTES, SF_BLOCK_BYTES};
    uint32_t boxDim[rank]          = {(uint32_t)X_ELEMS, 1, tile_k_blocks};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(tmap, CU_TENSOR_MAP_DATA_TYPE_UINT16, rank, (void *)ptr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2_promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ============================================================================
// Host-side fat LUT builder
// ============================================================================
struct TileLutTmp {
    uint32_t meta0;
    uint32_t meta1;
};

inline void build_tile_lut(KernelParams& params, int schedule) {
    TileLutTmp tmp[MAX_TILE_LUT];
    int total = 0;

    for (int g = 0; g < params.num_groups; g++) {
        const GroupInfo& gi = params.groups[g];
        const int M = gi.M, N = gi.N, K = gi.K;
        const int n_tiles = (N + 255) / 256;
        const int m_tiles = (M + BLOCK_N - 1) / BLOCK_N;
        const int m_rem = M % BLOCK_N;
        const int mma_n_tail = (m_rem == 0) ? BLOCK_N : ((m_rem + 15) & ~15);
        const int tile_num_k = K / BLOCK_K;
        TORCH_CHECK(tile_num_k >= 1 && tile_num_k <= 255, "tile_num_k out of packable range");

        for (int cn = 0; cn < n_tiles; cn++) {
            for (int cm = 0; cm < m_tiles; cm++) {
                TORCH_CHECK(total < MAX_TILE_LUT, "tile LUT overflow");
                TORCH_CHECK(g <= 7, "gidx overflow in packed tile meta");
                TORCH_CHECK(cm <= 1023, "coord_m overflow in packed tile meta");
                TORCH_CHECK(cn <= 1023, "coord_n_pair overflow in packed tile meta");

                const bool is_m_tail = (cm == m_tiles - 1) && (mma_n_tail != BLOCK_N);
                const int mma_n = is_m_tail ? mma_n_tail : BLOCK_N;
                TORCH_CHECK((mma_n & 15) == 0, "mma_n must be multiple of 16");
                TORCH_CHECK(mma_n >= 16 && mma_n <= 128, "mma_n out of supported range");

                const int off_m_base = cm * BLOCK_N;
                int off_n_r0 = cn * 256;
                int off_n_r1 = off_n_r0 + BLOCK_M;
                int rank1_aliased = 0;
                if (off_n_r1 >= N) {
                    off_n_r1 = off_n_r0;
                    rank1_aliased = 1;
                }

                const int n_residue_r0 = N - off_n_r0;
                const int n_residue_r1 = N - off_n_r1;
                const int a_tmap_r0 = (n_residue_r0 > 0 && n_residue_r0 < BLOCK_M) ? 1 : 0;
                const int a_tmap_r1 = (n_residue_r1 > 0 && n_residue_r1 < BLOCK_M) ? 1 : 0;
                const int a_bytes_r0 = a_tmap_r0 ? (n_residue_r0 * BLOCK_K / 2) : A_SIZE;
                const int a_bytes_r1 = a_tmap_r1 ? (n_residue_r1 * BLOCK_K / 2) : A_SIZE;

                const int b_half_rows = mma_n / 2;
                int b_rows_r0 = b_half_rows;
                int b_rows_r1 = b_half_rows;
                int b_tmap_r0 = 0;
                int b_tmap_r1 = 0;
                int off_m_b_r1_is_base = 0;
                if (is_m_tail) {
                    const int m_residue = M - off_m_base;
                    b_rows_r0 = (m_residue < b_half_rows) ? m_residue : b_half_rows;
                    b_rows_r1 = m_residue - b_half_rows;
                    if (b_rows_r1 < 0) b_rows_r1 = 0;
                    if (b_rows_r1 > b_half_rows) b_rows_r1 = b_half_rows;
                    b_tmap_r0 = 1;
                    b_tmap_r1 = 2;
                }
                if (b_rows_r0 < 1) b_rows_r0 = 1;
                if (b_rows_r1 < 1) {
                    b_rows_r1 = 1;
                    off_m_b_r1_is_base = 1;
                }

                const int b_bytes_r0 = b_rows_r0 * BLOCK_K / 2;
                const int b_bytes_r1 = b_rows_r1 * BLOCK_K / 2;
                const int expect_r0 = a_bytes_r0 + b_bytes_r0 + SFA_SIZE + SFB_SIZE;
                const int expect_r1 = a_bytes_r1 + b_bytes_r1 + SFA_SIZE + SFB_SIZE;
                TORCH_CHECK((expect_r0 & 127) == 0, "expect_r0 must be 128-byte aligned");
                TORCH_CHECK((expect_r1 & 127) == 0, "expect_r1 must be 128-byte aligned");
                const int expect_r0_q128 = expect_r0 >> 7;
                const int expect_r1_q128 = expect_r1 >> 7;
                TORCH_CHECK(expect_r0_q128 <= 255, "expect_r0_q128 overflow");
                TORCH_CHECK(expect_r1_q128 <= 255, "expect_r1_q128 overflow");

                tmp[total++] = {
                    pack_tile_meta0(
                        static_cast<uint32_t>(g), static_cast<uint32_t>(cm), static_cast<uint32_t>(cn),
                        static_cast<uint32_t>(tile_num_k), static_cast<uint32_t>(rank1_aliased)),
                    pack_tile_meta1(
                        static_cast<uint32_t>(mma_n >> 4),
                        static_cast<uint32_t>(a_tmap_r0), static_cast<uint32_t>(a_tmap_r1),
                        static_cast<uint32_t>(b_tmap_r0), static_cast<uint32_t>(b_tmap_r1),
                        static_cast<uint32_t>(off_m_b_r1_is_base),
                        static_cast<uint32_t>(expect_r0_q128), static_cast<uint32_t>(expect_r1_q128)),
                };
            }
        }
    }

    params.total_tiles = total;
    TORCH_CHECK(params.total_tiles <= MAX_TILE_LUT, "total_tiles exceeds LUT capacity");

    const int num_clusters = params.launch_ctas / 2;
    TORCH_CHECK(num_clusters > 0 && num_clusters <= MAX_CLUSTERS, "num_clusters out of range");

    int cursor = 0;
    for (int cid = 0; cid < num_clusters; cid++) {
        int logical_cid = cid;
        if (schedule == SCHED_F2_G2) logical_cid = (cid * 17) % num_clusters;
        else if (schedule == SCHED_F1_G1) logical_cid = num_clusters - 1 - cid;

        const int my_count = (params.total_tiles - logical_cid + num_clusters - 1) / num_clusters;
        TORCH_CHECK(cursor <= 65535, "lut_worker_start overflow");
        TORCH_CHECK(my_count <= 65535, "lut_worker_count overflow");
        params.lut_worker_start[cid] = static_cast<uint16_t>(cursor);
        params.lut_worker_count[cid] = static_cast<uint16_t>(my_count);

        #pragma unroll
        for (int tile_iter = 0; tile_iter < my_count; tile_iter++) {
            int k = tile_iter;
            if (schedule == SCHED_REV || schedule == SCHED_F1_G1) {
                k = my_count - 1 - tile_iter;
            } else if (schedule == SCHED_F2_G2) {
                const int h = (my_count + 1) >> 1;
                k = (tile_iter < h) ? (tile_iter << 1) : (((tile_iter - h) << 1) + 1);
            }
            const int tile_id = logical_cid + k * num_clusters;
            TORCH_CHECK(tile_id >= 0 && tile_id < params.total_tiles, "tile_id out of range");

            const TileLutTmp& t = tmp[tile_id];
            params.lut_meta0[cursor] = t.meta0;
            params.lut_meta1[cursor] = t.meta1;
            cursor++;
        }
    }

    for (int cid = num_clusters; cid < MAX_CLUSTERS; cid++) {
        params.lut_worker_start[cid] = 0;
        params.lut_worker_count[cid] = 0;
    }

    TORCH_CHECK(cursor == params.total_tiles, "tile LUT size mismatch");
}

// ============================================================================
// Kernel — persistent mbars, fat LUT, NS as template param
// ============================================================================
template <int SCHEDULE_ID, int PROFILE_ID>
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(TB_SIZE)
void grouped_gemm_kernel(
    const __grid_constant__ KernelParams params,
    const __grid_constant__ TmapParamPackG8 tmap_pack_g8
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bid = blockIdx.x;
    if (bid >= params.launch_ctas) return;

    const int cta_rank = static_cast<int>(get_cluster_ctarank());
    const int cluster_id = bid / 2;
    const int my_count = params.lut_worker_count[cluster_id];
    const int worker_start = params.lut_worker_start[cluster_id];

    extern __shared__ __align__(1024) char smem_raw[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_raw));
    const int smem_main = smem;
    half* smem_ep = reinterpret_cast<half*>(smem_raw + params.smem_size_bytes - EP_SMEM_BYTES);

    const int NS = params.ns;
    const int main_stg = params.main_stg;
    const int smem_sf = smem_main + NS * main_stg;

    // Mbarrier layout:
    // - NS tma + NS mma
    // - 2 done mbars (TMEM slot ready for EP)
    // - 2 ep mbars (EP drained slot; TMEM backpressure)
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbars[2 * MAX_NS + 4];
    __shared__ int32_t tmem_alloc_buf;
    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int tma_mbar  = mbar_base;
    const int mma_mbar  = tma_mbar + NS * 8;
    const int done_mbar0 = mma_mbar + NS * 8;
    const int done_mbar1 = done_mbar0 + 8;
    const int ep_mbar0 = done_mbar1 + 8;
    const int ep_mbar1 = ep_mbar0 + 8;

    if (my_count <= 0) return;

    // === ONE-TIME SETUP ===

    // MMA warp: allocate TMEM (both CTAs must issue)
    if (warp_id == NUM_WARPS - 1) {
        int alloc_addr = static_cast<int>(__cvta_generic_to_shared(&tmem_alloc_buf));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(alloc_addr), "r"(TMEM_COLS));
    }

    // Warp 0: init all mbarriers ONCE for the entire kernel lifetime
    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NS; i++) {
            mbarrier_init(tma_mbar + i * 8, 2);   // 2 CTAs arrive
            mbarrier_init(mma_mbar + i * 8, 1);   // 1 MMA warp (CTA0) arrives
        }
        mbarrier_init(done_mbar0, 1);
        mbarrier_init(done_mbar1, 1);
        mbarrier_init(ep_mbar0, 2);
        mbarrier_init(ep_mbar1, 2);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    // Single cluster barrier for the entire kernel
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    constexpr uint64_t cache_A =
        (PROFILE_ID == PROFILE_BENCH3 || PROFILE_ID == PROFILE_BENCH4) ? EVICT_FIRST :
        ((PROFILE_ID == PROFILE_GENERIC_G2) ? EVICT_NORMAL : 0ULL);
    constexpr uint64_t cache_B =
        (PROFILE_ID == PROFILE_BENCH3 || PROFILE_ID == PROFILE_GENERIC_G2) ? EVICT_FIRST :
        ((PROFILE_ID == PROFILE_BENCH4) ? EVICT_NORMAL : 0ULL);
    constexpr uint64_t cache_SF = cache_B;
    constexpr uint16_t cta_mask = 0x3;
    constexpr int SF_K_PER_BLOCK_L = BLOCK_K / 64;
    constexpr uint32_t MMA_M_2CTA = BLOCK_M * 2;

    auto make_desc_AB = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    // Helper to select A tmap based on index
    auto get_a_tmap = [&](int gidx, int a_idx) -> const void* {
        if (a_idx == 0) return static_cast<const void*>(&tmap_pack_g8.A_full[gidx]);
        return static_cast<const void*>(&tmap_pack_g8.A_tail[gidx]);
    };

    // Helper to select B tmap based on index
    auto get_b_tmap = [&](int gidx, int b_idx) -> const void* {
        if (b_idx == 0) return static_cast<const void*>(&tmap_pack_g8.B_full[gidx]);
        if (b_idx == 1) return static_cast<const void*>(&tmap_pack_g8.B_tail0[gidx]);
        return static_cast<const void*>(&tmap_pack_g8.B_tail1[gidx]);
    };

    // === TMA WARP ===
    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        int tma_stage = 0;
        int mma_wait_phase = 1;
        int total_produced = 0;
        #pragma unroll
        for (int tile = 0; tile < my_count; tile++) {
            const int lut_idx = worker_start + tile;
            TileRuntimeInfo tile_rt;
            decode_tile_runtime(params, lut_idx, tile_rt);
            const int gidx = tile_rt.gidx;
            const int tile_num_k = tile_rt.tile_num_k;
            const int off_n_cta = (cta_rank == 0) ? tile_rt.off_n_r0 : tile_rt.off_n_r1;
            const int coord_n_cta = (cta_rank == 0) ? tile_rt.coord_n_r0 : tile_rt.coord_n_r1;
            const int off_m_b = (cta_rank == 0) ? tile_rt.off_m_b_r0 : tile_rt.off_m_b_r1;
            const int coord_m = tile_rt.coord_m;
            const int tma_expect_bytes = (cta_rank == 0) ? tile_rt.expect_r0 : tile_rt.expect_r1;

            const int a_tmap_idx = (cta_rank == 0) ? tile_rt.a_tmap_r0 : tile_rt.a_tmap_r1;
            const int b_tmap_idx = (cta_rank == 0) ? tile_rt.b_tmap_r0 : tile_rt.b_tmap_r1;

            const void* A_tmap = get_a_tmap(gidx, a_tmap_idx);
            const void* B_tmap = get_b_tmap(gidx, b_tmap_idx);
            const void* SFA_tmap = static_cast<const void*>(&tmap_pack_g8.SFA[gidx]);
            const void* SFB_tmap = static_cast<const void*>(&tmap_pack_g8.SFB[gidx]);

            #pragma unroll 1
            for (int ik = 0; ik < tile_num_k; ik++) {
                if (total_produced >= NS) {
                    mbarrier_wait(mma_mbar + tma_stage * 8, mma_wait_phase);
                }

                const int mbar_addr = (tma_mbar + tma_stage * 8) & 0xFEFFFFFF;
                int A_s = smem_main + tma_stage * main_stg;
                int SFA_s = smem_sf + tma_stage * SF_STAGE;

                tma_3d_gmem2smem(A_s + A_SIZE, B_tmap, 0, off_m_b, ik, mbar_addr, cache_B);
                tma_3d_gmem2smem(A_s, A_tmap, 0, off_n_cta, ik, mbar_addr, cache_A);
                const int z_sf = ik * SF_K_PER_BLOCK_L;
                tma_3d_gmem2smem(SFA_s, SFA_tmap, 0, coord_n_cta, z_sf, mbar_addr, cache_SF);
                tma_3d_gmem2smem(SFA_s + SFA_SIZE, SFB_tmap, 0, coord_m, z_sf, mbar_addr, cache_SF);
                mbarrier_arrive_expect_tx(mbar_addr, tma_expect_bytes);

                total_produced++;
                tma_stage++;
                if (tma_stage == NS) {
                    tma_stage = 0;
                    mma_wait_phase ^= 1;
                }
            }
        }
    }

    // === MMA WARP (CTA0 only) ===
    if (cta_rank == 0 && warp_id == NUM_WARPS - 1 && elect_sync()) {
        int mma_stage = 0;
        int tma_wait_phase = 0;
        #pragma unroll
        for (int tile = 0; tile < my_count; tile++) {
            const int slot = tile & 1;
            const int d_tmem_base = (slot == 0) ? D_TMEM0 : D_TMEM1;
            const int done_mbar = (slot == 0) ? done_mbar0 : done_mbar1;

            // Do not reuse a TMEM slot before EP drains it on both CTAs.
            if (tile >= 2) {
                const int ep_wait_phase = (((tile >> 1) - 1) & 1);
                mbarrier_wait((slot == 0) ? ep_mbar0 : ep_mbar1, ep_wait_phase);
            }

            const int lut_idx = worker_start + tile;
            TileRuntimeInfo tile_rt;
            decode_tile_runtime(params, lut_idx, tile_rt);
            const int mma_n = tile_rt.mma_n;
            const int tile_num_k = tile_rt.tile_num_k;
            const uint32_t i_desc = (1U << 7U) | (1U << 10U) |
                (((uint32_t)mma_n >> 3U) << 17U) | (((uint32_t)MMA_M_2CTA >> 7U) << 27U);

            #pragma unroll 
            for (int ik = 0; ik < tile_num_k; ik++) {
                mbarrier_wait(tma_mbar + mma_stage * 8, tma_wait_phase);

                int A_s   = smem_main + mma_stage * main_stg;
                int B_s   = A_s + A_SIZE;
                int SFA_s = smem_sf + mma_stage * SF_STAGE;
                int SFB_s = SFA_s + SFA_SIZE;

                constexpr uint64_t sf_base = desc_encode(0) | (desc_encode(8*16) << 32ULL) | (1ULL << 46ULL);
                uint64_t sfa_desc = sf_base + ((uint64_t)SFA_s >> 4ULL);
                uint64_t sfb_desc = sf_base + ((uint64_t)SFB_s >> 4ULL);

                #pragma unroll
                for (int kk = 0; kk < BLOCK_K / MMA_K; kk++) {
                    tcgen05_cp_nvfp4(SFA_TMEM + kk * 4, sfa_desc + (uint64_t)kk * (512ULL >> 4ULL));
                    tcgen05_cp_nvfp4(SFB_TMEM + kk * 4, sfb_desc + (uint64_t)kk * (512ULL >> 4ULL));
                }

                #pragma unroll
                for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                    uint64_t a_desc = make_desc_AB(A_s + k2 * 32);
                    uint64_t b_desc = make_desc_AB(B_s + k2 * 32);
                    // Reset accumulator at start of each tile (enable_d=0 clears accum)
                    int enable_d = (ik == 0 && k2 == 0) ? 0 : 1;
                    tcgen05_mma_nvfp4(d_tmem_base, a_desc, b_desc, i_desc,
                        SFA_TMEM + k2 * 4, SFB_TMEM + k2 * 4, enable_d);
                }

                tcgen05_commit_mcast(mma_mbar + mma_stage * 8, cta_mask);

                mma_stage++;
                if (mma_stage == NS) {
                    mma_stage = 0;
                    tma_wait_phase ^= 1;
                }
            }
            tcgen05_commit_mcast(done_mbar, cta_mask);
        }
    }

    // === EP WARPS (both CTAs) ===
    if (warp_id < NUM_EP_WARPS) {
        for (int tile = 0; tile < my_count; tile++) {
            const int slot = tile & 1;
            const int done_mbar = (slot == 0) ? done_mbar0 : done_mbar1;
            const int done_phase = (tile >> 1) & 1;
            const int d_tmem_base = (slot == 0) ? D_TMEM0 : D_TMEM1;

            const int lut_idx = worker_start + tile;
            TileRuntimeInfo tile_rt;
            decode_tile_runtime(params, lut_idx, tile_rt);
            const int gidx = tile_rt.gidx;
            const GroupInfo& gi = params.groups[gidx];
            const int M = static_cast<int>(gi.M);
            const int N = static_cast<int>(gi.N);
            const int off_m = tile_rt.coord_m * BLOCK_N;
            const int mma_n = tile_rt.mma_n;
            const int off_n_r0 = tile_rt.off_n_r0;
            const int off_n_r1 = tile_rt.off_n_r1;
            const bool rank1_aliased = (off_n_r1 == off_n_r0);
            const int off_n = (cta_rank == 0) ? off_n_r0 : off_n_r1;

            if (!(cta_rank == 1 && rank1_aliased)) {
                do_epilogue_transposed(
                    warp_id, lane_id, cta_rank,
                    done_mbar, done_phase, d_tmem_base,
                    smem_ep, gi.c_ptr, M, N, off_m, off_n, mma_n);
            } else {
                mbarrier_wait(done_mbar, done_phase);
            }

            if (warp_id == 0 && elect_sync()) {
                tcgen05_commit_mcast((slot == 0) ? ep_mbar0 : ep_mbar1, cta_mask);
            }
        }
    }

    // === CLEANUP ===
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(TMEM_COLS));
}

template <int SCHEDULE_ID, int PROFILE_ID>
inline void launch_grouped_kernel(const KernelParams& params, const TmapParamPackG8& tmap_pack_g8, int smem_size) {
    auto kernel = grouped_gemm_kernel<SCHEDULE_ID, PROFILE_ID>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = params.launch_ctas;
    launch_config.blockDim = TB_SIZE;
    launch_config.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute cluster_attr = {};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = 2;
    cluster_attr.val.clusterDim.y = 1;
    cluster_attr.val.clusterDim.z = 1;
    launch_config.attrs = &cluster_attr;
    launch_config.numAttrs = 1;

    cudaLaunchKernelEx(&launch_config, kernel, params, tmap_pack_g8);
}

// ============================================================================
// Host launch
// ============================================================================
void grouped_gemm_impl(
    at::TensorList A_list,
    at::TensorList B_list,
    at::TensorList C_list,
    at::TensorList SFA_list,
    at::TensorList SFB_list
) {
    int G = A_list.size();
    TORCH_CHECK(G <= MAX_GROUPS, "num groups exceeds MAX_GROUPS");
    if (G == 0) return;
    KernelParams params = {};
    params.num_groups = G;

    static int smem_size = 0;
    static int smem_avail = 0;
    if (!smem_size) {
        int dev; cudaGetDevice(&dev);
        int smem_max;
        cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        smem_size = smem_max - 1024;
        smem_avail = smem_size - EP_SMEM_BYTES;
        TORCH_CHECK(smem_avail > 0, "Insufficient shared memory for EP scratch");
    }
    params.smem_size_bytes = smem_size;

    // Choose NS: largest value <= MAX_NS that fits in smem (independent of num_k divisibility).
    // For stage sizing, use the worst case (full tiles): main_stg = A_SIZE + B_HALF
    // First pass: find max B payload across all tiles to determine stage size
    int max_b_half = B_HALF;  // full tile
    // Actually all tiles fit B_HALF at most, tail tiles use less.
    // Use full B_HALF for stage size (wastes some smem on tail tiles but keeps layout uniform).
    int main_stg = A_SIZE + max_b_half;
    params.main_stg = main_stg;

    // Find best NS: largest that fits in smem.
    int best_ns = 1;
    for (int ns = MAX_NS; ns >= 1; ns--) {
        int total_smem = ns * main_stg + ns * SF_STAGE;
        if (total_smem <= smem_avail) {
            best_ns = ns;
            break;
        }
    }
    params.ns = best_ns;

    int raw_total_tiles = 0;
    for (int g = 0; g < G; g++) {
        int Mi = A_list[g].size(0);
        int Ki = A_list[g].size(1) * 2;
        int Ni = B_list[g].size(0);
        TORCH_CHECK(Mi >= 0 && Mi <= 65535, "M out of uint16 range");
        TORCH_CHECK(Ni >= 0 && Ni <= 65535, "N out of uint16 range");
        TORCH_CHECK(Ki >= 0 && Ki <= 65535, "K out of uint16 range");

        int nt = (Ni + 255) / 256;
        int mt = (Mi + BLOCK_N - 1) / BLOCK_N;

        params.groups[g] = {
            (half *)C_list[g].data_ptr(),
            static_cast<uint16_t>(Mi),
            static_cast<uint16_t>(Ni),
            static_cast<uint16_t>(Ki),
            0
        };
        raw_total_tiles += mt * nt;
    }

    const int cap_clusters = MAX_CLUSTERS;
    int num_clusters = raw_total_tiles < cap_clusters ? raw_total_tiles : cap_clusters;

    const bool is_bench1 = (params.num_groups == 8 && raw_total_tiles == 176);
    const bool is_bench2 = (params.num_groups == 8 && raw_total_tiles == 364);
    if (is_bench1 && V_BENCH1_CLUSTERS > 0) num_clusters = V_BENCH1_CLUSTERS;
    if (is_bench2 && V_BENCH2_CLUSTERS > 0) num_clusters = V_BENCH2_CLUSTERS;
    if (!is_bench1 && !is_bench2 && V_GENERIC_CLUSTERS > 0) num_clusters = V_GENERIC_CLUSTERS;
    if (num_clusters > MAX_CLUSTERS) num_clusters = MAX_CLUSTERS;
    if (num_clusters > raw_total_tiles) num_clusters = raw_total_tiles;
    if (num_clusters < 1) num_clusters = 1;

    params.launch_ctas = num_clusters * 2;

    int bench_profile = PROFILE_GENERIC_G8;
    if (params.num_groups == 8 && raw_total_tiles == 176) {
        bench_profile = PROFILE_BENCH1;
    } else if (params.num_groups == 8 && raw_total_tiles == 364) {
        bench_profile = PROFILE_BENCH2;
    } else if (params.num_groups == 2 && raw_total_tiles == 60) {
        bench_profile = PROFILE_BENCH3;
    } else if (params.num_groups == 2 && raw_total_tiles == 64) {
        bench_profile = PROFILE_BENCH4;
    } else if (params.num_groups == 2) {
        bench_profile = PROFILE_GENERIC_G2;
    }

    CUtensorMapL2promotion ab_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    CUtensorMapL2promotion sf_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    switch (bench_profile) {
        case PROFILE_BENCH1:
            ab_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            sf_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            break;
        case PROFILE_BENCH2:
            ab_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            sf_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            break;
        case PROFILE_BENCH3:
        case PROFILE_BENCH4:
        case PROFILE_GENERIC_G2:
            ab_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            sf_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            break;
        case PROFILE_GENERIC_G8:
        default:
            ab_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            sf_l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
            break;
    }

    TmapParamPackG8 tmap_pack_g8 = {};

    bool uniform_nk = true;
    for (int g = 1; g < G; g++) {
        if (B_list[g].size(0) != B_list[0].size(0) || A_list[g].size(1) != A_list[0].size(1)) {
            uniform_nk = false;
            break;
        }
    }

    int sfb_src[MAX_GROUPS];
    for (int g = 0; g < MAX_GROUPS; g++) sfb_src[g] = -1;
    if (uniform_nk) {
        int mn_first[4] = {-1, -1, -1, -1};
        for (int g = 0; g < G; g++) {
            int mnb = (A_list[g].size(0) + 127) / 128;
            sfb_src[g] = (mnb < 4) ? mn_first[mnb] : -1;
            if (mnb < 4 && mn_first[mnb] < 0) mn_first[mnb] = g;
        }
    }

    for (int g = 0; g < G; g++) {
        int Mi = A_list[g].size(0);
        int Ki = A_list[g].size(1) * 2;
        int Ni = B_list[g].size(0);

        int n_tail = Ni % BLOCK_M;
        if (g > 0 && uniform_nk) {
            tmap_pack_g8.A_full[g] = tmap_pack_g8.A_full[0];
            check_cu(cuTensorMapReplaceAddress(&tmap_pack_g8.A_full[g], (void *)B_list[g].data_ptr()));
            if (n_tail == 0) {
                tmap_pack_g8.A_tail[g] = tmap_pack_g8.A_full[g];
            } else {
                tmap_pack_g8.A_tail[g] = tmap_pack_g8.A_tail[0];
                check_cu(cuTensorMapReplaceAddress(&tmap_pack_g8.A_tail[g], (void *)B_list[g].data_ptr()));
            }
        } else {
            init_AB_tmap(&tmap_pack_g8.A_full[g], (const char *)B_list[g].data_ptr(), Ni, Ki, BLOCK_M, BLOCK_K, ab_l2_promotion);
            if (n_tail == 0) {
                tmap_pack_g8.A_tail[g] = tmap_pack_g8.A_full[g];
            } else {
                init_AB_tmap(&tmap_pack_g8.A_tail[g], (const char *)B_list[g].data_ptr(), Ni, Ki, n_tail, BLOCK_K, ab_l2_promotion);
            }
        }

        int m_tail = Mi % BLOCK_N;
        int mma_n_tail = (m_tail == 0) ? BLOCK_N : ((m_tail + 15) & ~15);
        int b_full_box_h = BLOCK_N / 2;
        int b_tail_half = mma_n_tail / 2;
        int b_tail0_box_h = m_tail < b_tail_half ? m_tail : b_tail_half;
        int b_tail1_box_h = m_tail - b_tail_half;
        if (b_tail1_box_h < 0) b_tail1_box_h = 0;
        if (b_tail1_box_h > b_tail_half) b_tail1_box_h = b_tail_half;
        init_AB_tmap(&tmap_pack_g8.B_full[g], (const char *)A_list[g].data_ptr(), Mi, Ki, b_full_box_h, BLOCK_K, ab_l2_promotion);
        if (m_tail == 0) {
            tmap_pack_g8.B_tail0[g] = tmap_pack_g8.B_full[g];
            tmap_pack_g8.B_tail1[g] = tmap_pack_g8.B_full[g];
        } else {
            init_AB_tmap(&tmap_pack_g8.B_tail0[g], (const char *)A_list[g].data_ptr(), Mi, Ki, b_tail0_box_h, BLOCK_K, ab_l2_promotion);
            const int b_tail1_box_safe = b_tail1_box_h > 0 ? b_tail1_box_h : 1;
            init_AB_tmap(&tmap_pack_g8.B_tail1[g], (const char *)A_list[g].data_ptr(), Mi, Ki, b_tail1_box_safe, BLOCK_K, ab_l2_promotion);
        }

        if (g > 0 && uniform_nk) {
            tmap_pack_g8.SFA[g] = tmap_pack_g8.SFA[0];
            check_cu(cuTensorMapReplaceAddress(&tmap_pack_g8.SFA[g], (void *)SFB_list[g].data_ptr()));
        } else {
            init_SF_tmap(&tmap_pack_g8.SFA[g], (const char *)SFB_list[g].data_ptr(), Ni, Ki, sf_l2_promotion);
        }

        if (uniform_nk && sfb_src[g] >= 0) {
            tmap_pack_g8.SFB[g] = tmap_pack_g8.SFB[sfb_src[g]];
            check_cu(cuTensorMapReplaceAddress(&tmap_pack_g8.SFB[g], (void *)SFA_list[g].data_ptr()));
        } else {
            init_SF_tmap(&tmap_pack_g8.SFB[g], (const char *)SFA_list[g].data_ptr(), Mi, Ki, sf_l2_promotion);
        }
    }

    int schedule = SCHED_REV;
    if (params.num_groups == 8 && raw_total_tiles == 176) {
        schedule = SCHED_F2_G2;
    } else if (params.num_groups == 8 && raw_total_tiles == 364) {
        schedule = SCHED_F1_G1;
    } else if (params.num_groups == 2 && raw_total_tiles == 64) {
        schedule = SCHED_REV;
    } else if (params.num_groups == 2) {
        schedule = SCHED_BASE;
    }

    if (V_FORCE_SCHEDULE >= 0) schedule = V_FORCE_SCHEDULE;
    build_tile_lut(params, schedule);

    #define LAUNCH_FOR_PROFILE(PROFILE_ID)                                                  \
        switch (schedule) {                                                                  \
            case SCHED_BASE:                                                                 \
                launch_grouped_kernel<SCHED_BASE, PROFILE_ID>(params, tmap_pack_g8, smem_size); \
                break;                                                                       \
            case SCHED_F2_G2:                                                                \
                launch_grouped_kernel<SCHED_F2_G2, PROFILE_ID>(params, tmap_pack_g8, smem_size); \
                break;                                                                       \
            case SCHED_F1_G1:                                                                \
                launch_grouped_kernel<SCHED_F1_G1, PROFILE_ID>(params, tmap_pack_g8, smem_size); \
                break;                                                                       \
            case SCHED_REV:                                                                  \
            default:                                                                         \
                launch_grouped_kernel<SCHED_REV, PROFILE_ID>(params, tmap_pack_g8, smem_size);   \
                break;                                                                       \
        }

    switch (bench_profile) {
        case PROFILE_BENCH1:
            LAUNCH_FOR_PROFILE(PROFILE_BENCH1);
            break;
        case PROFILE_BENCH2:
            LAUNCH_FOR_PROFILE(PROFILE_BENCH2);
            break;
        case PROFILE_BENCH3:
            LAUNCH_FOR_PROFILE(PROFILE_BENCH3);
            break;
        case PROFILE_BENCH4:
            LAUNCH_FOR_PROFILE(PROFILE_BENCH4);
            break;
        case PROFILE_GENERIC_G2:
            LAUNCH_FOR_PROFILE(PROFILE_GENERIC_G2);
            break;
        case PROFILE_GENERIC_G8:
        default:
            LAUNCH_FOR_PROFILE(PROFILE_GENERIC_G8);
            break;
    }

    #undef LAUNCH_FOR_PROFILE
}

TORCH_LIBRARY(gg_cg2_cta_noep_nosmem, m) {
    m.def("run(Tensor[] A, Tensor[] B, Tensor[] C, Tensor[] SFA, Tensor[] SFB) -> ()");
    m.impl("run", &grouped_gemm_impl);
}
"""

load_inline(
    "grouped_gemm_cg2_cta_noep_nosmem",
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


_run = torch.ops.gg_cg2_cta_noep_nosmem.run

def custom_kernel(data: input_t) -> output_t:
    abc, _, sf_reordered, _ = data
    a, b, c = zip(*abc)
    sfa, sfb = zip(*sf_reordered)
    _run(list(a), list(b), list(c), list(sfa), list(sfb))
    return list(c)
