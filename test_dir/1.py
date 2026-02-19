
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# Fuse along M (variable M tiles) with transposed TMEM epilogue (sub_dyn-style)
# and no collector usage.
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

// NOTE: In the transposed formulation (sub_dyn-style), we swap original A/B roles
// in the MMA. A_tmap loads original B tiles (N dimension), and B_tmap loads
// original A tiles (M dimension).
constexpr int A_SIZE  = BLOCK_M * BLOCK_K / 2;   // 16384 (shared original-B tile)
constexpr int B_SIZE  = BLOCK_N * BLOCK_K / 2;   // 16384 (per-M tile original-A)

constexpr int SFA_SIZE = 128 * BLOCK_K / 16;     // 2048 (scales for A operand)
constexpr int SFB_SIZE = 128 * BLOCK_K / 16;     // 2048 (scales for B operand)

constexpr int MAIN_STAGE = A_SIZE + B_SIZE;
constexpr int SF_STAGE   = SFA_SIZE + SFB_SIZE;
constexpr int SF_STAGE_FUSE_M = SFA_SIZE + 2 * SFB_SIZE;

constexpr int TMAP_SMEM  = 4 * 128;
constexpr int EP_BUF_SIZE = 16 * BLOCK_M * 2;  // 4096 bytes: 16 M-rows x 128 N-cols x sizeof(half)

constexpr int FULL_STAGES = 6;
constexpr int MAX_STAGES = 9;

// tma + mma + done0/done1
constexpr int NUM_MBAR = MAX_STAGES * 2 + 2;

constexpr int NUM_EP_WARPS = 4;
constexpr int NUM_WARPS = NUM_EP_WARPS + 2;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int TMEM_COLS = 512;

constexpr int MAX_GROUPS = 8;
constexpr int MAX_TILE_LUT = 1024;
constexpr int MAX_LAUNCH_CTAS = 148;

// TMEM layout (transposed epilogue expects cols = M, rows = N)
constexpr int D_TMEM0  = 0;
constexpr int D_TMEM1  = BLOCK_N;                 // 128
constexpr int SFA_TMEM = 2 * BLOCK_N;             // 256
constexpr int SFB0_TMEM = SFA_TMEM + 4 * (BLOCK_K / MMA_K); // 272
constexpr int SFB1_TMEM = SFB0_TMEM + 4 * (BLOCK_K / MMA_K); // 288

// ============================================================================
// Device structures
// ============================================================================
struct GroupInfo {
    half* c_ptr;
    int M, N, K;
    int m_tiles, n_tiles;
    int mma_n_tail;        // padded (multiple of 8) tail height for M-dimension tiles
    int num_stages_tail;   // pipeline stages for M-tail tiles
    int main_stage_tail;   // A_SIZE + b_tail_bytes
};

struct KernelParams {
    GroupInfo groups[MAX_GROUPS];
    int num_groups;

    int total_tiles;    // number of work units in LUT (after fusion)
    int launch_ctas;

    // Host-built worker mapping: per CTA contiguous range in packed LUT arrays.
    int32_t lut_worker_start[MAX_LAUNCH_CTAS];
    int32_t lut_worker_count[MAX_LAUNCH_CTAS];

    // LUT entries
    uint16_t lut_gidx[MAX_TILE_LUT];
    uint16_t lut_coord_n[MAX_TILE_LUT];
    uint16_t lut_coord_m_raw[MAX_TILE_LUT]; // high bit = is_fused_m
    uint16_t lut_off_m[MAX_TILE_LUT];
    uint16_t lut_off_n[MAX_TILE_LUT];
    uint16_t lut_b0_bytes[MAX_TILE_LUT];
    uint16_t lut_b1_bytes[MAX_TILE_LUT];
    uint16_t lut_num_stages[MAX_TILE_LUT];
    uint16_t lut_main_stg[MAX_TILE_LUT];
};

enum : int {
    PROFILE_GENERIC_G8 = 0,
    PROFILE_GENERIC_G2 = 1,
    PROFILE_BENCH1 = 2,
    PROFILE_BENCH2 = 3,
    PROFILE_BENCH3 = 4,
    PROFILE_BENCH4 = 5,
};

enum : int {
    SCHED_BASE = 0,
    SCHED_REV = 1,
    SCHED_F2_G2 = 2,
    SCHED_F1_G1 = 3,
};

constexpr int V_FORCE_SCHEDULE = -1;            // -1 keeps existing auto schedule selection
constexpr int V_BENCH1_LAUNCH_CTAS = 148;       // >0 overrides launch_ctas on bench1-like shapes
constexpr int V_BENCH2_LAUNCH_CTAS = 148;       // >0 overrides launch_ctas on bench2-like shapes
constexpr int V_GENERIC_LAUNCH_CTAS = -1;       // >0 overrides launch_ctas on other shapes
constexpr int V_USE_EXPLICIT_BENCH_PACK = 1;    // 1: use hand-packed bench LUT, 0: generic schedule packing

struct TmapParamPackG8 {
    CUtensorMap A_full[MAX_GROUPS];
    CUtensorMap A_tail[MAX_GROUPS];
    CUtensorMap B_full[MAX_GROUPS];
    CUtensorMap B_tail[MAX_GROUPS];
    CUtensorMap SFA[MAX_GROUPS];
    CUtensorMap SFB[MAX_GROUPS];
};

// ============================================================================
// Inline PTX helpers
// ============================================================================
__device__ inline constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

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

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
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
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline void tma_load_3d(int dst, const void *tmap_ptr, int x, int y, int z,
                                  int mbar_addr, uint64_t cache_policy) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%3, %4, %5}], [%2], %6;"
        :: "r"(dst), "l"(reinterpret_cast<uint64_t>(tmap_ptr)), "r"(mbar_addr),
           "r"(x), "r"(y), "r"(z), "l"(cache_policy)
        : "memory");
}

__device__ inline void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline void tcgen05_mma_nvfp4(uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
                                        int scale_A_tmem, int scale_B_tmem,
                                        int enable_input_d, int d_tmem) {
    asm volatile(
        "{\\n\\t"
        ".reg .pred p;\\n\\t"
        "setp.ne.b32 p, %6, 0;\\n\\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
        "}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
           "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d));
}

__device__ inline void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

__device__ inline void tcgen05_ld_16x256bx2(float *tmp, int row, int col) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x256b.x2.b32 { %0, %1, %2, %3, %4, %5, %6, %7 }, [%8];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
          "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
        : "r"((row << 16) | col));
}

// ============================================================================
// Epilogue (transposed, smem-based coalesced stores)
// ============================================================================
__device__ inline void do_epilogue_transposed_smem(int warp_id, int lane_id, int done_mbar, int d_tmem_base,
    half* __restrict__ smem_ep, half* __restrict__ c_ptr, int M, int N, int off_m, int off_n, int mma_n) {
    mbarrier_wait(done_mbar, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int col_lane = (lane_id % 4) * 2;
    const int row_lane = lane_id / 4;
    const int tid_ep = warp_id * WARP_SIZE + lane_id;
    const int residue_n = N - off_n;
    const int num_chunks = (mma_n + 15) / 16;

    // XOR swizzle masks for bank conflict reduction
    const int sw_lo = ((col_lane >> 1) & 7) << 4;
    const int sw_hi = (((col_lane + 8) >> 1) & 7) << 4;

    #pragma unroll 1
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        // Phase 1: TMEM -> smem (scatter with XOR swizzle)
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            const int tm = warp_id * 32 + m * 16;
            float vals[8];
            tcgen05_ld_16x256bx2(vals, tm, d_tmem_base + chunk * 16);
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            const int n0 = tm + row_lane;
            const int n1 = n0 + 8;

            smem_ep[col_lane       * 128 + (n0 ^ sw_lo)] = __float2half_rn(vals[0]);
            smem_ep[(col_lane + 1) * 128 + (n0 ^ sw_lo)] = __float2half_rn(vals[1]);
            smem_ep[col_lane       * 128 + (n1 ^ sw_lo)] = __float2half_rn(vals[2]);
            smem_ep[(col_lane + 1) * 128 + (n1 ^ sw_lo)] = __float2half_rn(vals[3]);
            smem_ep[(col_lane + 8) * 128 + (n0 ^ sw_hi)] = __float2half_rn(vals[4]);
            smem_ep[(col_lane + 9) * 128 + (n0 ^ sw_hi)] = __float2half_rn(vals[5]);
            smem_ep[(col_lane + 8) * 128 + (n1 ^ sw_hi)] = __float2half_rn(vals[6]);
            smem_ep[(col_lane + 9) * 128 + (n1 ^ sw_hi)] = __float2half_rn(vals[7]);
        }

        // Sync EP warps only (named barrier 15, 128 threads)
        asm volatile("bar.sync 15, %0;" :: "r"(NUM_EP_WARPS * WARP_SIZE));

        // Phase 2: smem -> global (coalesced int4 stores with XOR unswizzle)
        const int m_row = tid_ep / 8;       // 0..15
        const int n_group = tid_ep % 8;     // 0..7
        const int n_start = n_group * 16;   // 0,16,32,...,112
        const int m_local = chunk * 16 + m_row;
        const int m_global = off_m + m_local;

        if (m_local < mma_n && m_global < M && n_start < residue_n) {
            half* dst_base = &c_ptr[m_global * N + off_n + n_start];
            const int sw_rd = ((m_row >> 1) & 7) << 4;
            const half* src_base = &smem_ep[m_row * 128 + (n_start ^ sw_rd)];

            if (n_start + 16 <= residue_n) {
                *reinterpret_cast<int4*>(dst_base)     = *reinterpret_cast<const int4*>(src_base);
                *reinterpret_cast<int4*>(dst_base + 8) = *reinterpret_cast<const int4*>(src_base + 8);
            } else {
                for (int i = 0; i < 16 && n_start + i < residue_n; i++) {
                    dst_base[i] = src_base[i];
                }
            }
        }

        // Sync before next chunk (smem reuse)
        asm volatile("bar.sync 15, %0;" :: "r"(NUM_EP_WARPS * WARP_SIZE));
    }
}

// ============================================================================
// Tile compute: single tile (sub_dyn-style)
// ============================================================================
template <int NS>
__device__ __forceinline__ void do_tile_compute(
    int warp_id, int smem_main, int main_stg,
    const void *A_tmap, const void *B_tmap, const void *SFA_tmap, const void *SFB_tmap,
    int off_n, int off_m, int coord_n, int coord_m,
    int num_k, int tma_mbar, int mma_mbar, int done_mbar,
    int tma_expect_bytes, uint32_t i_desc, int d_tmem_base,
    uint64_t cache_A, uint64_t cache_B, uint64_t cache_SF
) {
    const int smem_sf = smem_main + NS * main_stg;
    constexpr int SF_K = BLOCK_K / 64;

    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            mbarrier_init(tma_mbar + i * 8, 1);
            mbarrier_init(mma_mbar + i * 8, 1);
        }
        mbarrier_init(done_mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    auto mk = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        #pragma unroll
        for (int ik = 0; ik < NS && ik < num_k; ik++) {
            int s = ik;
            int A_s = smem_main + s * main_stg;
            int SFA_s = smem_sf + s * SF_STAGE;
            tma_load_3d(A_s, A_tmap, 0, off_n, ik, tma_mbar + s * 8, cache_A);
            tma_load_3d(A_s + A_SIZE, B_tmap, 0, off_m, ik, tma_mbar + s * 8, cache_B);
            int z = ik * SF_K;
            tma_load_3d(SFA_s, SFA_tmap, 0, coord_n, z, tma_mbar + s * 8, cache_SF);
            tma_load_3d(SFA_s + SFA_SIZE, SFB_tmap, 0, coord_m, z, tma_mbar + s * 8, cache_SF);
            mbarrier_arrive_expect_tx(tma_mbar + s * 8, tma_expect_bytes);
        }
        for (int ik = NS; ik < num_k; ik++) {
            int s = ik % NS;
            mbarrier_wait(mma_mbar + s * 8, (ik / NS - 1) % 2);
            int A_s = smem_main + s * main_stg;
            int SFA_s = smem_sf + s * SF_STAGE;
            tma_load_3d(A_s, A_tmap, 0, off_n, ik, tma_mbar + s * 8, cache_A);
            tma_load_3d(A_s + A_SIZE, B_tmap, 0, off_m, ik, tma_mbar + s * 8, cache_B);
            int z = ik * SF_K;
            tma_load_3d(SFA_s, SFA_tmap, 0, coord_n, z, tma_mbar + s * 8, cache_SF);
            tma_load_3d(SFA_s + SFA_SIZE, SFB_tmap, 0, coord_m, z, tma_mbar + s * 8, cache_SF);
            mbarrier_arrive_expect_tx(tma_mbar + s * 8, tma_expect_bytes);
        }
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync()) {
        #pragma unroll 1
        for (int ik = 0; ik < num_k; ik++) {
            int s = ik % NS;
            mbarrier_wait(tma_mbar + s * 8, (ik / NS) % 2);
            int A_s = smem_main + s * main_stg;
            int B_s = A_s + A_SIZE;
            int SFA_s = smem_sf + s * SF_STAGE;
            int SFB_s = SFA_s + SFA_SIZE;

            constexpr uint64_t sf_base = desc_encode(0) | (desc_encode(8 * 16) << 32ULL) | (1ULL << 46ULL);
            uint64_t sfa_d = sf_base + ((uint64_t)SFA_s >> 4ULL);
            uint64_t sfb_d = sf_base + ((uint64_t)SFB_s >> 4ULL);

            #pragma unroll
            for (int kk = 0; kk < BLOCK_K / MMA_K; kk++) {
                tcgen05_cp_nvfp4(SFA_TMEM + kk * 4, sfa_d + (uint64_t)kk * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB0_TMEM + kk * 4, sfb_d + (uint64_t)kk * (512ULL >> 4ULL));
            }

            const uint64_t a_base_desc = mk(A_s);
            const uint64_t b_base_desc = mk(B_s);
            const uint64_t a_delta = mk(A_s + 32) - a_base_desc;
            const uint64_t b_delta = mk(B_s + 32) - b_base_desc;
            #pragma unroll
            for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                uint64_t ad = a_base_desc + (uint64_t)k2 * a_delta;
                uint64_t bd = b_base_desc + (uint64_t)k2 * b_delta;
                tcgen05_mma_nvfp4(ad, bd, i_desc,
                    SFA_TMEM + k2 * 4, SFB0_TMEM + k2 * 4,
                    (ik == 0 && k2 == 0) ? 0 : 1, d_tmem_base);
            }
            tcgen05_commit(mma_mbar + s * 8);
        }
        tcgen05_commit(done_mbar);
    }
}

// ============================================================================
// Tile compute: fuse along M (two adjacent M-tiles share original-B tile)
// Layout per stage: [A_shared | B0 | B1] + [SFA_shared | SFB0 | SFB1]
// No collector usage in fused path: issue two regular MMAs (D0 then D1).
// ============================================================================
template <int NS>
__device__ __forceinline__ void do_tile_compute_fuse_m_nocollector(
    int warp_id, int smem_main, int main_stg,
    int b0_bytes,
    const void *A_tmap, const void *B0_tmap, const void *B1_tmap,
    const void *SFA_tmap, const void *SFB_tmap,
    int off_n, int off_m0, int off_m1,
    int coord_n, int coord_m0, int coord_m1,
    int num_k, int tma_mbar, int mma_mbar,
    int done_mbar0, int done_mbar1,
    int tma_expect_bytes,
    uint32_t i_desc0, uint32_t i_desc1,
    int d_tmem0, int d_tmem1,
    uint64_t cache_A, uint64_t cache_B, uint64_t cache_SF
) {
    const int smem_sf = smem_main + NS * main_stg;
    constexpr int SF_K = BLOCK_K / 64;

    if (warp_id == 0 && elect_sync()) {
        #pragma unroll
        for (int i = 0; i < NS; i++) {
            mbarrier_init(tma_mbar + i * 8, 1);
            mbarrier_init(mma_mbar + i * 8, 1);
        }
        mbarrier_init(done_mbar0, 1);
        mbarrier_init(done_mbar1, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    auto mk = [](int addr) -> uint64_t {
        return desc_encode(addr) | (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
    };

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
        #pragma unroll
        for (int ik = 0; ik < NS && ik < num_k; ik++) {
            int s = ik;
            int base_s = smem_main + s * main_stg;
            int sf_s = smem_sf + s * SF_STAGE_FUSE_M;
            int A_s = base_s;
            int B0_s = A_s + A_SIZE;
            int B1_s = B0_s + b0_bytes;

            tma_load_3d(A_s, A_tmap, 0, off_n, ik, tma_mbar + s * 8, cache_A);
            tma_load_3d(B0_s, B0_tmap, 0, off_m0, ik, tma_mbar + s * 8, cache_B);
            tma_load_3d(B1_s, B1_tmap, 0, off_m1, ik, tma_mbar + s * 8, cache_B);

            int z = ik * SF_K;
            tma_load_3d(sf_s, SFA_tmap, 0, coord_n, z, tma_mbar + s * 8, cache_SF);
            tma_load_3d(sf_s + SFA_SIZE, SFB_tmap, 0, coord_m0, z, tma_mbar + s * 8, cache_SF);
            tma_load_3d(sf_s + SFA_SIZE + SFB_SIZE, SFB_tmap, 0, coord_m1, z, tma_mbar + s * 8, cache_SF);
            mbarrier_arrive_expect_tx(tma_mbar + s * 8, tma_expect_bytes);
        }
        for (int ik = NS; ik < num_k; ik++) {
            int s = ik % NS;
            mbarrier_wait(mma_mbar + s * 8, (ik / NS - 1) % 2);

            int base_s = smem_main + s * main_stg;
            int sf_s = smem_sf + s * SF_STAGE_FUSE_M;
            int A_s = base_s;
            int B0_s = A_s + A_SIZE;
            int B1_s = B0_s + b0_bytes;

            tma_load_3d(A_s, A_tmap, 0, off_n, ik, tma_mbar + s * 8, cache_A);
            tma_load_3d(B0_s, B0_tmap, 0, off_m0, ik, tma_mbar + s * 8, cache_B);
            tma_load_3d(B1_s, B1_tmap, 0, off_m1, ik, tma_mbar + s * 8, cache_B);

            int z = ik * SF_K;
            tma_load_3d(sf_s, SFA_tmap, 0, coord_n, z, tma_mbar + s * 8, cache_SF);
            tma_load_3d(sf_s + SFA_SIZE, SFB_tmap, 0, coord_m0, z, tma_mbar + s * 8, cache_SF);
            tma_load_3d(sf_s + SFA_SIZE + SFB_SIZE, SFB_tmap, 0, coord_m1, z, tma_mbar + s * 8, cache_SF);

            mbarrier_arrive_expect_tx(tma_mbar + s * 8, tma_expect_bytes);
        }
    }

    if (warp_id == NUM_WARPS - 1 && elect_sync()) {
        #pragma unroll 1
        for (int ik = 0; ik < num_k; ik++) {
            int s = ik % NS;
            mbarrier_wait(tma_mbar + s * 8, (ik / NS) % 2);

            int base_s = smem_main + s * main_stg;
            int A_s = base_s;
            int B0_s = A_s + A_SIZE;
            int B1_s = B0_s + b0_bytes;

            int sf_s = smem_sf + s * SF_STAGE_FUSE_M;
            int SFA_s = sf_s;
            int SFB0_s = SFA_s + SFA_SIZE;
            int SFB1_s = SFB0_s + SFB_SIZE;

            constexpr uint64_t sf_base = desc_encode(0) | (desc_encode(8 * 16) << 32ULL) | (1ULL << 46ULL);
            uint64_t sfa_d = sf_base + ((uint64_t)SFA_s >> 4ULL);
            uint64_t sfb0_d = sf_base + ((uint64_t)SFB0_s >> 4ULL);
            uint64_t sfb1_d = sf_base + ((uint64_t)SFB1_s >> 4ULL);

            #pragma unroll
            for (int kk = 0; kk < BLOCK_K / MMA_K; kk++) {
                tcgen05_cp_nvfp4(SFA_TMEM + kk * 4, sfa_d + (uint64_t)kk * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB0_TMEM + kk * 4, sfb0_d + (uint64_t)kk * (512ULL >> 4ULL));
                tcgen05_cp_nvfp4(SFB1_TMEM + kk * 4, sfb1_d + (uint64_t)kk * (512ULL >> 4ULL));
            }

            const uint64_t a_base_desc = mk(A_s);
            const uint64_t b0_base_desc = mk(B0_s);
            const uint64_t b1_base_desc = mk(B1_s);
            const uint64_t a_delta = mk(A_s + 32) - a_base_desc;
            const uint64_t b0_delta = mk(B0_s + 32) - b0_base_desc;
            const uint64_t b1_delta = mk(B1_s + 32) - b1_base_desc;
            #pragma unroll
            for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
                uint64_t ad = a_base_desc + (uint64_t)k2 * a_delta;
                uint64_t b0d = b0_base_desc + (uint64_t)k2 * b0_delta;
                uint64_t b1d = b1_base_desc + (uint64_t)k2 * b1_delta;
                int enable_d = (ik == 0 && k2 == 0) ? 0 : 1;

                tcgen05_mma_nvfp4(ad, b0d, i_desc0,
                    SFA_TMEM + k2 * 4, SFB0_TMEM + k2 * 4,
                    enable_d, d_tmem0);
                tcgen05_mma_nvfp4(ad, b1d, i_desc1,
                    SFA_TMEM + k2 * 4, SFB1_TMEM + k2 * 4,
                    enable_d, d_tmem1);
            }

            tcgen05_commit(mma_mbar + s * 8);
        }
        tcgen05_commit(done_mbar0);
        tcgen05_commit(done_mbar1);
    }
}

// ============================================================================
// TensorMap initialization
// ============================================================================
void check_cu(CUresult err) {
    if (err == CUDA_SUCCESS) return;
    const char *msg;
    if (cuGetErrorString(err, &msg) != CUDA_SUCCESS) msg = "unknown";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", msg);
}

void init_AB_tmap(CUtensorMap *tmap, const char *ptr, uint64_t height, uint64_t width,
                  uint32_t box_h, uint32_t box_w, CUtensorMapL2promotion l2) {
    constexpr uint32_t rank = 3;
    uint64_t gd[rank] = {256, height, width / 256};
    uint64_t gs[rank - 1] = {width / 2, 128};
    uint32_t bd[rank] = {256, box_h, box_w / 256};
    uint32_t es[rank] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank, (void*)ptr,
        gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

void init_SF_tmap(CUtensorMap *tmap, const char *ptr, uint64_t mn, uint64_t K,
                  CUtensorMapL2promotion l2) {
    constexpr uint32_t rank = 3;
    uint64_t kb = K / 64;
    uint64_t mb = (mn + 127) / 128;
    uint32_t tk = BLOCK_K / 64;
    uint64_t gd[rank] = {256, mb, kb};
    uint64_t gs[rank - 1] = {kb * 512, 512};
    uint32_t bd[rank] = {256, 1, tk};
    uint32_t es[rank] = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(
        tmap, CU_TENSOR_MAP_DATA_TYPE_UINT16, rank, (void*)ptr,
        gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

// ============================================================================
// LUT builder: fuse along M within each (group, coord_n)
// ============================================================================
inline int clamp_stages(int stages, int min_stages) {
    if (stages > MAX_STAGES) stages = MAX_STAGES;
    if (stages < min_stages) stages = min_stages;
    return stages;
}

struct TileTmp {
    uint16_t gidx;
    uint16_t coord_n;
    uint16_t coord_m_raw;
    uint16_t off_m;
    uint16_t off_n;
    uint16_t b0_bytes;
    uint16_t b1_bytes;
    uint16_t num_stages;
    uint16_t main_stg;
};

inline void write_tmp_to_lut(KernelParams& params, const TileTmp* tmp, int cursor, int idx) {
    TORCH_CHECK(0 <= idx && idx < params.total_tiles, "tmp index out of range");
    const TileTmp& t = tmp[idx];
    TORCH_CHECK(cursor < MAX_TILE_LUT, "tile LUT overflow");
    params.lut_gidx[cursor] = t.gidx;
    params.lut_coord_n[cursor] = t.coord_n;
    params.lut_coord_m_raw[cursor] = t.coord_m_raw;
    params.lut_off_m[cursor] = t.off_m;
    params.lut_off_n[cursor] = t.off_n;
    params.lut_b0_bytes[cursor] = t.b0_bytes;
    params.lut_b1_bytes[cursor] = t.b1_bytes;
    params.lut_num_stages[cursor] = t.num_stages;
    params.lut_main_stg[cursor] = t.main_stg;
}

inline bool is_bench1_shape_fused(const KernelParams& params) {
    if (params.num_groups != 8 || params.total_tiles != 256) return false;
    const int m_ref[8] = {80, 176, 128, 72, 64, 248, 96, 160};
    const int n_ref[8] = {4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096};
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        if (params.groups[g].M != m_ref[g] || params.groups[g].N != n_ref[g]) return false;
    }
    return true;
}

inline bool is_bench2_shape_fused(const KernelParams& params) {
    if (params.num_groups != 8 || params.total_tiles != 448) return false;
    const int m_ref[8] = {40, 76, 168, 72, 164, 148, 196, 160};
    const int n_ref[8] = {7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168};
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        if (params.groups[g].M != m_ref[g] || params.groups[g].N != n_ref[g]) return false;
    }
    return true;
}

inline void build_tile_lut(KernelParams& params, int schedule, int smem_avail) {
    TileTmp tmp[MAX_TILE_LUT];
    int total = 0;

    for (int g = 0; g < params.num_groups; g++) {
        const GroupInfo& gi = params.groups[g];
        const int b_tail_bytes = gi.mma_n_tail * (BLOCK_K / 2);

        for (int cn = 0; cn < gi.n_tiles; cn++) {
            for (int cm = 0; cm < gi.m_tiles; cm += 2) {
                const int off_n = cn * BLOCK_M;
                const int off_m0 = cm * BLOCK_N;

                if (cm + 1 < gi.m_tiles) {
                    const int cm1 = cm + 1;
                    const int b0_bytes = B_SIZE;
                    const bool is_tail1 = (cm1 == gi.m_tiles - 1) && (gi.mma_n_tail != BLOCK_N);
                    const int b1_bytes = is_tail1 ? b_tail_bytes : B_SIZE;

                    const int main_stg = A_SIZE + b0_bytes + b1_bytes;
                    int stages = smem_avail / (main_stg + SF_STAGE_FUSE_M);
                    stages = clamp_stages(stages, 4);

                    TORCH_CHECK(total < MAX_TILE_LUT, "tile LUT overflow");
                    tmp[total++] = {
                        (uint16_t)g,
                        (uint16_t)cn,
                        (uint16_t)(0x8000u | (uint16_t)cm),
                        (uint16_t)off_m0,
                        (uint16_t)off_n,
                        (uint16_t)b0_bytes,
                        (uint16_t)b1_bytes,
                        (uint16_t)stages,
                        (uint16_t)main_stg,
                    };
                } else {
                    const bool is_tail0 = (cm == gi.m_tiles - 1) && (gi.mma_n_tail != BLOCK_N);
                    const int b0_bytes = is_tail0 ? b_tail_bytes : B_SIZE;
                    const int main_stg = is_tail0 ? gi.main_stage_tail : (A_SIZE + B_SIZE);
                    int stages = is_tail0 ? gi.num_stages_tail : FULL_STAGES;
                    stages = clamp_stages(stages, FULL_STAGES);

                    TORCH_CHECK(total < MAX_TILE_LUT, "tile LUT overflow");
                    tmp[total++] = {
                        (uint16_t)g,
                        (uint16_t)cn,
                        (uint16_t)cm,
                        (uint16_t)off_m0,
                        (uint16_t)off_n,
                        (uint16_t)b0_bytes,
                        0,
                        (uint16_t)stages,
                        (uint16_t)main_stg,
                    };
                }
            }
        }
    }

    params.total_tiles = total;
    const int cap_ctas = (total > 128 && total <= 384) ? 128 : MAX_LAUNCH_CTAS;
    params.launch_ctas = (total < cap_ctas) ? total : cap_ctas;
    const bool bench1_shape = is_bench1_shape_fused(params);
    const bool bench2_shape = is_bench2_shape_fused(params);
    if (bench1_shape && V_BENCH1_LAUNCH_CTAS > 0) params.launch_ctas = V_BENCH1_LAUNCH_CTAS;
    if (bench2_shape && V_BENCH2_LAUNCH_CTAS > 0) params.launch_ctas = V_BENCH2_LAUNCH_CTAS;
    if (!bench1_shape && !bench2_shape && V_GENERIC_LAUNCH_CTAS > 0) params.launch_ctas = V_GENERIC_LAUNCH_CTAS;
    if (params.launch_ctas > params.total_tiles) params.launch_ctas = params.total_tiles;

    TORCH_CHECK(params.launch_ctas <= MAX_LAUNCH_CTAS, "launch_ctas exceeds LUT capacity");
    TORCH_CHECK(params.total_tiles <= MAX_TILE_LUT, "total_tiles exceeds LUT capacity");

    const bool bench1 = is_bench1_shape_fused(params);
    const bool bench2 = is_bench2_shape_fused(params);
    if ((bench1 || bench2) && V_USE_EXPLICIT_BENCH_PACK) {
        int per_cta_count[MAX_LAUNCH_CTAS] = {0};
        int per_cta_units[MAX_LAUNCH_CTAS][5] = {{0}};

        auto push_unit = [&](int bid, int unit_idx) {
            TORCH_CHECK(0 <= bid && bid < params.launch_ctas, "bid out of range");
            TORCH_CHECK(per_cta_count[bid] < 5, "per-CTA unit overflow");
            per_cta_units[bid][per_cta_count[bid]++] = unit_idx;
        };

        auto collect = [&](int* out, int& n, int g, bool fused) {
            for (int i = 0; i < params.total_tiles; i++) {
                const bool is_fused = (tmp[i].coord_m_raw & 0x8000u) != 0;
                if (is_fused == fused && (int)tmp[i].gidx == g) out[n++] = i;
            }
        };

        if (bench1) {
            TORCH_CHECK(params.launch_ctas == MAX_LAUNCH_CTAS, "bench1 fused mapping expects 148 CTAs");
            int p160[64], p176[64], p248[64];
            int s64[64], s72[64], s80[64], s96[64], s128[64];
            int np160 = 0, np176 = 0, np248 = 0;
            int ns64 = 0, ns72 = 0, ns80 = 0, ns96 = 0, ns128 = 0;
            collect(p160, np160, 7, true);
            collect(p176, np176, 1, true);
            collect(p248, np248, 5, true);
            collect(s64, ns64, 4, false);
            collect(s72, ns72, 3, false);
            collect(s80, ns80, 0, false);
            collect(s96, ns96, 6, false);
            collect(s128, ns128, 2, false);
            TORCH_CHECK(np160 == 32 && np176 == 32 && np248 == 32, "bench1 pair category count mismatch");
            TORCH_CHECK(ns64 == 32 && ns72 == 32 && ns80 == 32 && ns96 == 32 && ns128 == 32,
                        "bench1 single category count mismatch");

            int ip160 = 0, ip176 = 0, ip248 = 0;
            int is64 = 0, is72 = 0, is80 = 0, is96 = 0, is128 = 0;

            for (int bid = 0; bid <= 31; bid++) { push_unit(bid, p160[ip160++]); push_unit(bid, s64[is64++]); }
            for (int bid = 32; bid <= 55; bid++) { push_unit(bid, p176[ip176++]); push_unit(bid, s72[is72++]); }
            for (int bid = 56; bid <= 63; bid++) { push_unit(bid, p176[ip176++]); }
            for (int bid = 64; bid <= 95; bid++) { push_unit(bid, p248[ip248++]); }
            for (int bid = 96; bid <= 103; bid++) { push_unit(bid, s80[is80++]); push_unit(bid, s96[is96++]); }
            for (int bid = 104; bid <= 127; bid++) { push_unit(bid, s80[is80++]); push_unit(bid, s128[is128++]); }
            for (int bid = 128; bid <= 135; bid++) { push_unit(bid, s72[is72++]); push_unit(bid, s128[is128++]); }
            for (int bid = 136; bid <= 147; bid++) { push_unit(bid, s96[is96++]); push_unit(bid, s96[is96++]); }

            TORCH_CHECK(ip160 == np160 && ip176 == np176 && ip248 == np248, "bench1 pair fill mismatch");
            TORCH_CHECK(is64 == ns64 && is72 == ns72 && is80 == ns80 && is96 == ns96 && is128 == ns128,
                        "bench1 single fill mismatch");
        } else {
            TORCH_CHECK(params.launch_ctas == MAX_LAUNCH_CTAS, "bench2 fused mapping expects 148 CTAs");
            int p148[64], p160[64], p164[64], p168[64], p196[64];
            int s40[64], s72[64], s76[64];
            int np148 = 0, np160 = 0, np164 = 0, np168 = 0, np196 = 0;
            int ns40 = 0, ns72 = 0, ns76 = 0;
            collect(p148, np148, 5, true);
            collect(p160, np160, 7, true);
            collect(p164, np164, 4, true);
            collect(p168, np168, 2, true);
            collect(p196, np196, 6, true);
            collect(s40, ns40, 0, false);
            collect(s72, ns72, 3, false);
            collect(s76, ns76, 1, false);
            TORCH_CHECK(np148 == 56 && np160 == 56 && np164 == 56 && np168 == 56 && np196 == 56,
                        "bench2 pair category count mismatch");
            TORCH_CHECK(ns40 == 56 && ns72 == 56 && ns76 == 56, "bench2 single category count mismatch");

            int ip148 = 0, ip160 = 0, ip164 = 0, ip168 = 0, ip196 = 0;
            int is40 = 0, is72 = 0, is76 = 0;

            for (int bid = 0; bid <= 11; bid++) { push_unit(bid, p160[ip160++]); push_unit(bid, p196[ip196++]); }
            for (int bid = 12; bid <= 15; bid++) { push_unit(bid, p160[ip160++]); push_unit(bid, p196[ip196++]); push_unit(bid, s40[is40++]); }
            for (int bid = 16; bid <= 55; bid++) { push_unit(bid, p148[ip148++]); push_unit(bid, p196[ip196++]); push_unit(bid, s40[is40++]); }
            for (int bid = 56; bid <= 67; bid++) { push_unit(bid, p164[ip164++]); push_unit(bid, p168[ip168++]); push_unit(bid, s40[is40++]); }
            for (int bid = 68; bid <= 71; bid++) { push_unit(bid, p164[ip164++]); push_unit(bid, p168[ip168++]); push_unit(bid, s72[is72++]); }
            for (int bid = 72; bid <= 91; bid++) { push_unit(bid, p164[ip164++]); push_unit(bid, p164[ip164++]); push_unit(bid, s72[is72++]); }
            for (int bid = 92; bid <= 107; bid++) { push_unit(bid, p160[ip160++]); push_unit(bid, p168[ip168++]); push_unit(bid, s72[is72++]); }
            for (int bid = 108; bid <= 131; bid++) { push_unit(bid, p160[ip160++]); push_unit(bid, p168[ip168++]); push_unit(bid, s76[is76++]); }
            for (int bid = 132; bid <= 147; bid++) { push_unit(bid, p148[ip148++]); push_unit(bid, s76[is76++]); push_unit(bid, s76[is76++]); push_unit(bid, s72[is72++]); }

            TORCH_CHECK(ip148 == np148 && ip160 == np160 && ip164 == np164 && ip168 == np168 && ip196 == np196,
                        "bench2 pair fill mismatch");
            TORCH_CHECK(is40 == ns40 && is72 == ns72 && is76 == ns76, "bench2 single fill mismatch");
        }

        int cursor = 0;
        for (int bid = 0; bid < params.launch_ctas; bid++) {
            params.lut_worker_start[bid] = cursor;
            params.lut_worker_count[bid] = per_cta_count[bid];
            for (int k = 0; k < per_cta_count[bid]; k++) {
                write_tmp_to_lut(params, tmp, cursor, per_cta_units[bid][k]);
                cursor++;
            }
        }
        for (int bid = params.launch_ctas; bid < MAX_LAUNCH_CTAS; bid++) {
            params.lut_worker_start[bid] = 0;
            params.lut_worker_count[bid] = 0;
        }
        TORCH_CHECK(cursor == params.total_tiles, "explicit fused LUT size mismatch");
        return;
    }

    int cursor = 0;
    for (int bid = 0; bid < params.launch_ctas; bid++) {
        int logical_bid = bid;
        if (schedule == SCHED_F2_G2) {
            logical_bid = (bid * 17) % params.launch_ctas;
        } else if (schedule == SCHED_F1_G1) {
            logical_bid = params.launch_ctas - 1 - bid;
        }

        const int my_count = (params.total_tiles - logical_bid + params.launch_ctas - 1) / params.launch_ctas;
        params.lut_worker_start[bid] = cursor;
        params.lut_worker_count[bid] = my_count;

        for (int tile_iter = 0; tile_iter < my_count; tile_iter++) {
            int k = tile_iter;
            if (schedule == SCHED_REV || schedule == SCHED_F1_G1) {
                k = my_count - 1 - tile_iter;
            } else if (schedule == SCHED_F2_G2) {
                const int h = (my_count + 1) >> 1;
                k = (tile_iter < h) ? (tile_iter << 1) : (((tile_iter - h) << 1) + 1);
            }

            const int tile_id = logical_bid + k * params.launch_ctas;
            TORCH_CHECK(tile_id >= 0 && tile_id < params.total_tiles, "tile_id out of range");
            write_tmp_to_lut(params, tmp, cursor, tile_id);
            cursor++;
        }
    }

    for (int bid = params.launch_ctas; bid < MAX_LAUNCH_CTAS; bid++) {
        params.lut_worker_start[bid] = 0;
        params.lut_worker_count[bid] = 0;
    }
    TORCH_CHECK(cursor == params.total_tiles, "tile LUT size mismatch");
}

// ============================================================================
// Kernel
// ============================================================================
template <int SCHEDULE_ID, int PROFILE_ID>
__global__ __launch_bounds__(TB_SIZE)
void grouped_gemm_kernel(
    const __grid_constant__ KernelParams params,
    const __grid_constant__ TmapParamPackG8 tmap_pack_g8
) {
    struct EpMeta {
        half* c_ptr;
        int M, N;
        int off_m, off_n;
        int mma_n;
    };

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bid = blockIdx.x;
    if (bid >= params.launch_ctas) return;

    extern __shared__ __align__(1024) char smem_raw[];
    half* smem_ep = reinterpret_cast<half*>(smem_raw);
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_raw + EP_BUF_SIZE));
    const int smem_main = smem + TMAP_SMEM;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int64_t mbars[NUM_MBAR];
    __shared__ int32_t tmem_alloc_buf;
    __shared__ EpMeta ep_meta[2];

    const int mbar_base = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int tma_mbar = mbar_base;
    const int mma_mbar = tma_mbar + MAX_STAGES * 8;
    const int done_mbar0 = mma_mbar + MAX_STAGES * 8;
    const int done_mbar1 = done_mbar0 + 8;

    if (warp_id == 1) {
        int alloc_addr = static_cast<int>(__cvta_generic_to_shared(&tmem_alloc_buf));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(alloc_addr), "r"(TMEM_COLS));
    }
    __syncthreads();

    constexpr uint64_t cache_A =
        (PROFILE_ID == PROFILE_BENCH3 || PROFILE_ID == PROFILE_BENCH4) ? EVICT_FIRST :
        ((PROFILE_ID == PROFILE_GENERIC_G2) ? EVICT_NORMAL : 0ULL);
    constexpr uint64_t cache_B =
        (PROFILE_ID == PROFILE_BENCH3 || PROFILE_ID == PROFILE_GENERIC_G2) ? EVICT_FIRST :
        ((PROFILE_ID == PROFILE_BENCH4) ? EVICT_NORMAL : 0ULL);
    constexpr uint64_t cache_SF = cache_B;

    int pending_slot = -1;

    const int my_count = params.lut_worker_count[bid];
    if (my_count <= 0) return;
    const int worker_start = params.lut_worker_start[bid];

    for (int tile_iter = 0; tile_iter < my_count; tile_iter++) {
        const int lut_idx = worker_start + tile_iter;
        const int gidx = static_cast<int>(params.lut_gidx[lut_idx]);
        const int coord_n = static_cast<int>(params.lut_coord_n[lut_idx]);
        const uint16_t coord_m_raw = params.lut_coord_m_raw[lut_idx];
        const int is_fused = (coord_m_raw & 0x8000u) != 0;
        const int coord_m0 = static_cast<int>(coord_m_raw & 0x7FFFu);

        const GroupInfo& gi = params.groups[gidx];
        const int M = gi.M;
        const int N = gi.N;
        const int K = gi.K;
        const int num_k = K / BLOCK_K;

        const int off_m0 = static_cast<int>(params.lut_off_m[lut_idx]);
        const int off_n = static_cast<int>(params.lut_off_n[lut_idx]);

        const int b0_bytes = static_cast<int>(params.lut_b0_bytes[lut_idx]);
        const int b1_bytes = static_cast<int>(params.lut_b1_bytes[lut_idx]);
        const int main_stg = static_cast<int>(params.lut_main_stg[lut_idx]);
        const int num_stages = static_cast<int>(params.lut_num_stages[lut_idx]);

        const void *At = static_cast<const void*>(&(tmap_pack_g8.A_full[gidx]));
        const void *B0t = static_cast<const void*>(&(b0_bytes == B_SIZE ? tmap_pack_g8.B_full[gidx] : tmap_pack_g8.B_tail[gidx]));
        const void *B1t = static_cast<const void*>(&(b1_bytes == B_SIZE ? tmap_pack_g8.B_full[gidx] : tmap_pack_g8.B_tail[gidx]));
        const void *SFAt = static_cast<const void*>(&tmap_pack_g8.SFA[gidx]);
        const void *SFBt = static_cast<const void*>(&tmap_pack_g8.SFB[gidx]);

        if (is_fused) {
            const int off_m1 = off_m0 + BLOCK_N;
            const int coord_m1 = coord_m0 + 1;

            // Fused-M needs both D slots; drain any pending first.
            if (warp_id < NUM_EP_WARPS && pending_slot >= 0) {
             EpMeta meta = ep_meta[pending_slot];
            do_epilogue_transposed_smem(warp_id, lane_id,
            (pending_slot == 0) ? done_mbar0 : done_mbar1,
           (pending_slot == 0) ? D_TMEM0 : D_TMEM1,
          smem_ep, meta.c_ptr, meta.M, meta.N, meta.off_m, meta.off_n, meta.mma_n);
            }
            __syncthreads();
            pending_slot = -1;

            const int mma_n0 = b0_bytes / (BLOCK_K / 2);
            const int mma_n1 = b1_bytes / (BLOCK_K / 2);
            const uint32_t i_desc0 = (1U<<7U)|(1U<<10U)|(((uint32_t)mma_n0>>3U)<<17U)|(((uint32_t)BLOCK_M>>7U)<<27U);
            const uint32_t i_desc1 = (1U<<7U)|(1U<<10U)|(((uint32_t)mma_n1>>3U)<<17U)|(((uint32_t)BLOCK_M>>7U)<<27U);

            const int tma_expect_bytes = main_stg + SF_STAGE_FUSE_M;

            if (warp_id == 0 && lane_id == 0) {
                ep_meta[0] = {gi.c_ptr, M, N, off_m0, off_n, mma_n0};
                ep_meta[1] = {gi.c_ptr, M, N, off_m1, off_n, mma_n1};
            }

            switch (num_stages) {
                case 4: do_tile_compute_fuse_m_nocollector<4>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
                case 5: do_tile_compute_fuse_m_nocollector<5>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
                case 6: do_tile_compute_fuse_m_nocollector<6>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
                case 7: do_tile_compute_fuse_m_nocollector<7>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
                case 8: do_tile_compute_fuse_m_nocollector<8>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
                case 9: do_tile_compute_fuse_m_nocollector<9>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
                default: do_tile_compute_fuse_m_nocollector<4>(warp_id, smem_main, main_stg, b0_bytes,
                        At, B0t, B1t, SFAt, SFBt,
                        off_n, off_m0, off_m1, coord_n, coord_m0, coord_m1,
                        num_k, tma_mbar, mma_mbar,
                        done_mbar0, done_mbar1, tma_expect_bytes,
                        i_desc0, i_desc1, D_TMEM0, D_TMEM1,
                        cache_A, cache_B, cache_SF); break;
            }

            // Drain D0 now, keep D1 pending.
            if (warp_id < NUM_EP_WARPS) {
             EpMeta meta = ep_meta[0];
            do_epilogue_transposed_smem(warp_id, lane_id, done_mbar0, D_TMEM0,
           smem_ep, meta.c_ptr, meta.M, meta.N, meta.off_m, meta.off_n, meta.mma_n);
            }
            __syncthreads();
            pending_slot = 1;

        } else {
            // Single tile: use free slot, overlap draining pending slot.
            const int compute_slot = (pending_slot < 0) ? 0 : (1 - pending_slot);
            const int d_base = (compute_slot == 0) ? D_TMEM0 : D_TMEM1;
            const int dm = (compute_slot == 0) ? done_mbar0 : done_mbar1;

            const int mma_n = b0_bytes / (BLOCK_K / 2);
            const uint32_t i_desc = (1U<<7U)|(1U<<10U)|(((uint32_t)mma_n>>3U)<<17U)|(((uint32_t)BLOCK_M>>7U)<<27U);
            const int tma_expect_bytes = main_stg + SF_STAGE;

            if (warp_id == 0 && lane_id == 0) {
                ep_meta[compute_slot] = {gi.c_ptr, M, N, off_m0, off_n, mma_n};
            }

            switch (num_stages) {
                case 6: do_tile_compute<6>(warp_id, smem_main, main_stg,
                            At, B0t, SFAt, SFBt,
                            off_n, off_m0, coord_n, coord_m0,
                            num_k, tma_mbar, mma_mbar, dm,
                            tma_expect_bytes, i_desc, d_base,
                            cache_A, cache_B, cache_SF);
                        break;
                case 7: do_tile_compute<7>(warp_id, smem_main, main_stg,
                            At, B0t, SFAt, SFBt,
                            off_n, off_m0, coord_n, coord_m0,
                            num_k, tma_mbar, mma_mbar, dm,
                            tma_expect_bytes, i_desc, d_base,
                            cache_A, cache_B, cache_SF);
                        break;
                case 8: do_tile_compute<8>(warp_id, smem_main, main_stg,
                            At, B0t, SFAt, SFBt,
                            off_n, off_m0, coord_n, coord_m0,
                            num_k, tma_mbar, mma_mbar, dm,
                            tma_expect_bytes, i_desc, d_base,
                            cache_A, cache_B, cache_SF);
                        break;
                case 9: do_tile_compute<9>(warp_id, smem_main, main_stg,
                            At, B0t, SFAt, SFBt,
                            off_n, off_m0, coord_n, coord_m0,
                            num_k, tma_mbar, mma_mbar, dm,
                            tma_expect_bytes, i_desc, d_base,
                            cache_A, cache_B, cache_SF);
                        break;
                default: do_tile_compute<6>(warp_id, smem_main, main_stg,
                            At, B0t, SFAt, SFBt,
                            off_n, off_m0, coord_n, coord_m0,
                            num_k, tma_mbar, mma_mbar, dm,
                            tma_expect_bytes, i_desc, d_base,
                            cache_A, cache_B, cache_SF);
                        break;
            }

            if (warp_id < NUM_EP_WARPS && pending_slot >= 0) {
          EpMeta meta = ep_meta[pending_slot];
         do_epilogue_transposed_smem(warp_id, lane_id,
        (pending_slot == 0) ? done_mbar0 : done_mbar1,
       (pending_slot == 0) ? D_TMEM0 : D_TMEM1,
      smem_ep, meta.c_ptr, meta.M, meta.N, meta.off_m, meta.off_n, meta.mma_n);
            }
            __syncthreads();
            pending_slot = compute_slot;
        }
    }

    if (warp_id < NUM_EP_WARPS && pending_slot >= 0) {
     EpMeta meta = ep_meta[pending_slot];
    do_epilogue_transposed_smem(warp_id, lane_id,
    (pending_slot == 0) ? done_mbar0 : done_mbar1,
   (pending_slot == 0) ? D_TMEM0 : D_TMEM1,
  smem_ep, meta.c_ptr, meta.M, meta.N, meta.off_m, meta.off_n, meta.mma_n);
    }
    __syncthreads();

    if (warp_id == 0)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(TMEM_COLS));
}

template <int SCHEDULE_ID, int PROFILE_ID>
inline void launch_grouped_kernel(const KernelParams& params, const TmapParamPackG8& t, int smem_size) {
    cudaFuncSetAttribute(grouped_gemm_kernel<SCHEDULE_ID, PROFILE_ID>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(grouped_gemm_kernel<SCHEDULE_ID, PROFILE_ID>,
                         cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    grouped_gemm_kernel<SCHEDULE_ID, PROFILE_ID><<<params.launch_ctas, TB_SIZE, smem_size>>>(params, t);
}

void grouped_gemm_impl(at::TensorList A_list, at::TensorList B_list, at::TensorList C_list,
                       at::TensorList SFA_list, at::TensorList SFB_list) {
    int G = A_list.size();
    TORCH_CHECK(G <= MAX_GROUPS);
    if (G == 0) return;

    KernelParams params = {};
    params.num_groups = G;

    static int smem_size = 0;
    static int smem_avail = 0;
    if (!smem_size) {
        int dev;
        cudaGetDevice(&dev);
        int sm;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        smem_size = sm - 1024;
        smem_avail = smem_size - EP_BUF_SIZE - TMAP_SMEM;
    }

    int raw_total_tiles = 0;
    for (int g = 0; g < G; g++) {
        int Mi = A_list[g].size(0);
        int Ki = A_list[g].size(1) * 2;
        int Ni = B_list[g].size(0);

        int n_tiles = (Ni + BLOCK_M - 1) / BLOCK_M;
        int m_tiles = (Mi + BLOCK_N - 1) / BLOCK_N;

        int m_rem = Mi % BLOCK_N;
        int mma_n_tail = (m_rem == 0) ? BLOCK_N : ((m_rem + 7) & ~7);
        int b_bytes_tail = mma_n_tail * (BLOCK_K / 2);
        int main_stage_tail = A_SIZE + b_bytes_tail;
        int num_stages_tail = smem_avail / (main_stage_tail + SF_STAGE);
        if (num_stages_tail > MAX_STAGES) num_stages_tail = MAX_STAGES;
        if (num_stages_tail < FULL_STAGES) num_stages_tail = FULL_STAGES;

        params.groups[g] = {(half*)C_list[g].data_ptr(), Mi, Ni, Ki, m_tiles, n_tiles,
                            mma_n_tail, num_stages_tail, main_stage_tail};
        raw_total_tiles += m_tiles * n_tiles;
    }

    // Bench profile selection uses raw (unfused) tile counts for compatibility.
    int bench_profile = PROFILE_GENERIC_G8;
    if (params.num_groups == 8 && raw_total_tiles == 352) bench_profile = PROFILE_BENCH1;
    else if (params.num_groups == 8 && raw_total_tiles == 728) bench_profile = PROFILE_BENCH2;
    else if (params.num_groups == 2 && raw_total_tiles == 120) bench_profile = PROFILE_BENCH3;
    else if (params.num_groups == 2 && raw_total_tiles == 128) bench_profile = PROFILE_BENCH4;
    else if (params.num_groups == 2) bench_profile = PROFILE_GENERIC_G2;

    CUtensorMapL2promotion ab_l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    CUtensorMapL2promotion sf_l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    switch (bench_profile) {
        case PROFILE_BENCH1:
            ab_l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            sf_l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            break;
        case PROFILE_BENCH2:
            ab_l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
            sf_l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            break;
        case PROFILE_BENCH3:
        case PROFILE_BENCH4:
        case PROFILE_GENERIC_G2:
            ab_l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            sf_l2 = CU_TENSOR_MAP_L2_PROMOTION_NONE;
            break;
        default:
            break;
    }

    // Init tensor maps (transposed formulation: A_tmap uses original B, B_tmap uses original A)
    TmapParamPackG8 tp = {};
    bool uniform_nk = true;
    for (int g = 1; g < G; g++) {
        if (B_list[g].size(0) != B_list[0].size(0) || A_list[g].size(1) != A_list[0].size(1)) {
            uniform_nk = false;
            break;
        }
    }

    // Reuse transposed SFB map encodings by M-block class (mirrors pairmax's SFA reuse idea).
    int sfb_src[MAX_GROUPS];
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
            tp.A_full[g] = tp.A_full[0];
            check_cu(cuTensorMapReplaceAddress(&tp.A_full[g], (void*)B_list[g].data_ptr()));
            if (n_tail == 0) {
                tp.A_tail[g] = tp.A_full[g];
            } else {
                tp.A_tail[g] = tp.A_tail[0];
                check_cu(cuTensorMapReplaceAddress(&tp.A_tail[g], (void*)B_list[g].data_ptr()));
            }
        } else {
            init_AB_tmap(&tp.A_full[g], (const char*)B_list[g].data_ptr(), Ni, Ki, BLOCK_M, BLOCK_K, ab_l2);
            if (n_tail == 0) tp.A_tail[g] = tp.A_full[g];
            else init_AB_tmap(&tp.A_tail[g], (const char*)B_list[g].data_ptr(), Ni, Ki, n_tail, BLOCK_K, ab_l2);
        }

        int m_tail = Mi % BLOCK_N;
        int m_tail_box = (m_tail == 0) ? BLOCK_N : ((m_tail + 7) & ~7);
        init_AB_tmap(&tp.B_full[g], (const char*)A_list[g].data_ptr(), Mi, Ki, BLOCK_N, BLOCK_K, ab_l2);
        if (m_tail == 0) tp.B_tail[g] = tp.B_full[g];
        else init_AB_tmap(&tp.B_tail[g], (const char*)A_list[g].data_ptr(), Mi, Ki, m_tail_box, BLOCK_K, ab_l2);

        // Swap scale maps to match swapped A/B in MMA.
        if (g > 0 && uniform_nk) {
            tp.SFA[g] = tp.SFA[0];
            check_cu(cuTensorMapReplaceAddress(&tp.SFA[g], (void*)SFB_list[g].data_ptr()));
        } else {
            init_SF_tmap(&tp.SFA[g], (const char*)SFB_list[g].data_ptr(), Ni, Ki, sf_l2);
        }

        if (uniform_nk && sfb_src[g] >= 0) {
            tp.SFB[g] = tp.SFB[sfb_src[g]];
            check_cu(cuTensorMapReplaceAddress(&tp.SFB[g], (void*)SFA_list[g].data_ptr()));
        } else {
            init_SF_tmap(&tp.SFB[g], (const char*)SFA_list[g].data_ptr(), Mi, Ki, sf_l2);
        }
    }

    int schedule = SCHED_REV;
    if (params.num_groups == 8 && raw_total_tiles == 352) {
        schedule = SCHED_F2_G2;
    } else if (params.num_groups == 8 && raw_total_tiles == 728) {
        schedule = SCHED_F1_G1;
    } else if (params.num_groups == 2 && raw_total_tiles == 128) {
        schedule = SCHED_REV;
    } else if (params.num_groups == 2) {
        schedule = SCHED_BASE;
    }
    if (V_FORCE_SCHEDULE >= 0) schedule = V_FORCE_SCHEDULE;

    build_tile_lut(params, schedule, smem_avail);

    #define LAUNCH_FOR_PROFILE(PROFILE_ID)                                                   \
        switch (schedule) {                                                                  \
            case SCHED_BASE:                                                                 \
                launch_grouped_kernel<SCHED_BASE, PROFILE_ID>(params, tp, smem_size);       \
                break;                                                                       \
            case SCHED_F2_G2:                                                                \
                launch_grouped_kernel<SCHED_F2_G2, PROFILE_ID>(params, tp, smem_size);      \
                break;                                                                       \
            case SCHED_F1_G1:                                                                \
                launch_grouped_kernel<SCHED_F1_G1, PROFILE_ID>(params, tp, smem_size);      \
                break;                                                                       \
            case SCHED_REV:                                                                  \
            default:                                                                         \
                launch_grouped_kernel<SCHED_REV, PROFILE_ID>(params, tp, smem_size);        \
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

TORCH_LIBRARY(gg_ep_xor, m) {
    m.def("run(Tensor[] A, Tensor[] B, Tensor[] C, Tensor[] SFA, Tensor[] SFB) -> ()");
    m.impl("run", &grouped_gemm_impl);
}
"""

load_inline(
    "gg_ep_xor_mod",
    cpp_sources="",
    cuda_sources=cuda_src,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
)

_run = torch.ops.gg_ep_xor.run


def custom_kernel(data: input_t) -> output_t:
    abc, _, sf_reordered, _ = data
    a, b, c = zip(*abc)
    sfa, sfb = zip(*sf_reordered)
    _run(list(a), list(b), list(c), list(sfa), list(sfb))
    return list(c)

