from __future__ import annotations

from typing import List

import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

CPP_SRC = r"""
#include <torch/extension.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

void dispatch_group_gemm_raw(
  int G,
  const int64_t* packed_ptrs,
  const int* problem_sizes
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dispatch_pack_dims", [](py::sequence abc_pack, py::sequence sf_pack, py::sequence dims) {
    const int G = (int)abc_pack.size();

    alignas(64) int64_t ptrs[5 * 8];
    alignas(64) int ps[4 * 8];

    for (int i = 0; i < G; ++i) {
      auto abc = py::reinterpret_borrow<py::tuple>(abc_pack[i]);
      auto sf  = py::reinterpret_borrow<py::tuple>(sf_pack[i]);
      auto d   = py::reinterpret_borrow<py::tuple>(dims[i]);

      const at::Tensor a   = abc[0].cast<at::Tensor>();
      const at::Tensor b   = abc[1].cast<at::Tensor>();
      const at::Tensor c   = abc[2].cast<at::Tensor>();
      const at::Tensor sfa = sf[0].cast<at::Tensor>();
      const at::Tensor sfb = sf[1].cast<at::Tensor>();

      ptrs[5 * i + 0] = (int64_t)a.data_ptr();
      ptrs[5 * i + 1] = (int64_t)b.data_ptr();
      ptrs[5 * i + 2] = (int64_t)c.data_ptr();
      ptrs[5 * i + 3] = (int64_t)sfa.data_ptr();
      ptrs[5 * i + 4] = (int64_t)sfb.data_ptr();

      ps[4 * i + 0] = d[0].cast<int>();
      ps[4 * i + 1] = d[1].cast<int>();
      ps[4 * i + 2] = d[2].cast<int>();
      ps[4 * i + 3] = d[3].cast<int>();
    }

    dispatch_group_gemm_raw(G, ptrs, ps);
  });
}
"""

CUDA_SRC = r"""
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>

#include <stddef.h>
#include <stdint.h>
#include <torch/library.h>

__device__ inline constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

template <typename T>
__device__ __forceinline__ T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@!P1 bra.uni LAB_WAIT;\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ __forceinline__ void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::%2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc), "n"(CTA_GROUP));
}

template <int CTA_GROUP = 1>
__device__ __forceinline__ void tcgen05_mma_nvfp4(
  int d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d
) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::%7.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d), "n"(CTA_GROUP)
  );
}

struct SHAPE {
  static constexpr char _32x32b[]  = ".32x32b";
};

template <int NUM_REGS, const char *SHAPE_, int NUM>
__device__ __forceinline__ void tcgen05_ld(float *tmp, int row, int col) {
  const int addr = (row << 16) | col;
  if constexpr (NUM_REGS == 4) {
    asm volatile("tcgen05.ld.sync.aligned%5.x%6.b32 "
                "{%0, %1, %2, %3}, [%4];"
                : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
                : "r"(addr), "C"(SHAPE_), "n"(NUM));
  }
  if constexpr (NUM_REGS == 8) {
    asm volatile("tcgen05.ld.sync.aligned%9.x%10.b32 "
                "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7}, [%8];"
                : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                : "r"(addr), "C"(SHAPE_), "n"(NUM));
  }
  if constexpr (NUM_REGS == 16) {
    asm volatile("tcgen05.ld.sync.aligned%17.x%18.b32 "
                "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
                : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                  "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
                : "r"(addr), "C"(SHAPE_), "n"(NUM));
  }
  if constexpr (NUM_REGS == 32) {
    asm volatile("tcgen05.ld.sync.aligned%33.x%34.b32 "
                "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                "  %8,  %9, %10, %11, %12, %13, %14, %15, "
                " %16, %17, %18, %19, %20, %21, %22, %23, "
                " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                  "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                  "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                  "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
                : "r"(addr), "C"(SHAPE_), "n"(NUM));
  }
}

template <int num>
__device__ __forceinline__ void tcgen05_ld_32x32b(float *tmp, int row, int col) {
  tcgen05_ld<num, SHAPE::_32x32b, num>(tmp, row, col);
}

__device__ __forceinline__ void store_cs_32B(half *ptr,
    uint32_t h0, uint32_t h1, uint32_t h2, uint32_t h3,
    uint32_t h4, uint32_t h5, uint32_t h6, uint32_t h7) {
  asm volatile("{\n\t"
    ".reg .b64 d0, d1, d2, d3;\n\t"
    "mov.b64 d0, {%1, %2};\n\t"
    "mov.b64 d1, {%3, %4};\n\t"
    "mov.b64 d2, {%5, %6};\n\t"
    "mov.b64 d3, {%7, %8};\n\t"
    "st.cs.v4.b64 [%0], {d0, d1, d2, d3};\n\t"
    "}"
    :: "l"(ptr), "r"(h0), "r"(h1), "r"(h2), "r"(h3),
       "r"(h4), "r"(h5), "r"(h6), "r"(h7) : "memory");
}

__device__ __forceinline__ void tcgen05_ld_32x32b_pack16(uint32_t *d, int row, int col) {
  float tmp[16];
  tcgen05_ld_32x32b<16>(tmp, row, col);
  half2 h0 = __floats2half2_rn(tmp[ 0], tmp[ 1]);
  half2 h1 = __floats2half2_rn(tmp[ 2], tmp[ 3]);
  half2 h2 = __floats2half2_rn(tmp[ 4], tmp[ 5]);
  half2 h3 = __floats2half2_rn(tmp[ 6], tmp[ 7]);
  half2 h4 = __floats2half2_rn(tmp[ 8], tmp[ 9]);
  half2 h5 = __floats2half2_rn(tmp[10], tmp[11]);
  half2 h6 = __floats2half2_rn(tmp[12], tmp[13]);
  half2 h7 = __floats2half2_rn(tmp[14], tmp[15]);
  d[0] = *(uint32_t*)&h0;
  d[1] = *(uint32_t*)&h1;
  d[2] = *(uint32_t*)&h2;
  d[3] = *(uint32_t*)&h3;
  d[4] = *(uint32_t*)&h4;
  d[5] = *(uint32_t*)&h5;
  d[6] = *(uint32_t*)&h6;
  d[7] = *(uint32_t*)&h7;
}

static void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr = nullptr;
  cuGetErrorString(err, &error_msg_ptr);
  TORCH_CHECK(false, "cuTensorMap error: ", (error_msg_ptr ? error_msg_ptr : "unknown"));
}

struct __align__(16) Meta {
  uint64_t C[8];
  uint64_t SFA[8];
  uint64_t SFB[8];
  int M[8];
  int N[8];
  int K[8];
  int offsets[9];
  int num_groups;
  int tiles_count;
};

template <int BM, int BN, int SWIZZLE = 2>
__device__ __forceinline__ void decode_tile(
    const Meta& meta, int tile_id, int& group, int& off_m, int& off_n) {
  group = 0;
  #pragma unroll
  for (int g = 0; g < 8; ++g) {
    if (g < meta.num_groups && tile_id >= meta.offsets[g + 1]) group = g + 1;
  }
  const int local_id = tile_id - meta.offsets[group];
  const int tiles_m = (meta.M[group] + BM - 1) / BM;
  const int tiles_n = (meta.N[group] + BN - 1) / BN;
  const int swz = (tiles_n < SWIZZLE) ? tiles_n : SWIZZLE;
  const int block_sz = tiles_m * swz;
  const int block_id = local_id / block_sz;
  const int local    = local_id % block_sz;
  const int m_idx    = local / swz;
  const int n_local  = local % swz;
  const int n_idx    = block_id * swz + n_local;
  off_m = m_idx * BM;
  off_n = n_idx * BN;
}

struct __align__(8) TileInfo {
  int16_t group;
  int16_t off_m;
  int16_t off_n;
  int16_t pad;
};

struct __align__(16) MetaP {
  uint64_t C[8];
  uint64_t SFA[8];
  uint64_t SFB[8];
  int M[8];
  int N[8];
  int K[8];
  int num_groups;
  int tiles_count;
//  TileInfo tiles[768];
  uint32_t tiles[768];
};
static inline uint32_t pack_tile_u32(int group, int m_idx, int n_idx) {
  TORCH_CHECK((unsigned)group < 8, "group overflow");
  TORCH_CHECK((unsigned)m_idx < (1u << 14), "m_idx overflow");
  TORCH_CHECK((unsigned)n_idx < (1u << 15), "n_idx overflow");
  return (uint32_t)group | ((uint32_t)m_idx << 3) | ((uint32_t)n_idx << 17);
}

__device__ __forceinline__ void unpack_tile_u32(uint32_t t, int &group, int &off_m, int &off_n) {
  group = (int)(t & 0x7u);
  int m_idx = (int)((t >> 3) & 0x3FFFu);
  int n_idx = (int)((t >> 17) & 0x7FFFu);
  off_m = m_idx << 7;  // *128
  off_n = n_idx << 7;  // *128
}


struct __align__(64) DeviceBlob {
  CUtensorMap A[8];
  CUtensorMap B[8];
};

#define TMAP_KERNEL_PARAMS \
  const __grid_constant__ CUtensorMap kA0, const __grid_constant__ CUtensorMap kA1, \
  const __grid_constant__ CUtensorMap kA2, const __grid_constant__ CUtensorMap kA3, \
  const __grid_constant__ CUtensorMap kA4, const __grid_constant__ CUtensorMap kA5, \
  const __grid_constant__ CUtensorMap kA6, const __grid_constant__ CUtensorMap kA7, \
  const __grid_constant__ CUtensorMap kB0, const __grid_constant__ CUtensorMap kB1, \
  const __grid_constant__ CUtensorMap kB2, const __grid_constant__ CUtensorMap kB3, \
  const __grid_constant__ CUtensorMap kB4, const __grid_constant__ CUtensorMap kB5, \
  const __grid_constant__ CUtensorMap kB6, const __grid_constant__ CUtensorMap kB7

#define TMAP_LAUNCH_ARGS(blob) \
  (blob).A[0], (blob).A[1], (blob).A[2], (blob).A[3], \
  (blob).A[4], (blob).A[5], (blob).A[6], (blob).A[7], \
  (blob).B[0], (blob).B[1], (blob).B[2], (blob).B[3], \
  (blob).B[4], (blob).B[5], (blob).B[6], (blob).B[7]

__device__ __forceinline__
const CUtensorMap* tmap_select_A(int group,
    const CUtensorMap &A0, const CUtensorMap &A1, const CUtensorMap &A2, const CUtensorMap &A3,
    const CUtensorMap &A4, const CUtensorMap &A5, const CUtensorMap &A6, const CUtensorMap &A7) {
  switch (group) {
    case 0: return &A0; case 1: return &A1; case 2: return &A2; case 3: return &A3;
    case 4: return &A4; case 5: return &A5; case 6: return &A6; default: return &A7;
  }
}
__device__ __forceinline__
const CUtensorMap* tmap_select_B(int group,
    const CUtensorMap &B0, const CUtensorMap &B1, const CUtensorMap &B2, const CUtensorMap &B3,
    const CUtensorMap &B4, const CUtensorMap &B5, const CUtensorMap &B6, const CUtensorMap &B7) {
  switch (group) {
    case 0: return &B0; case 1: return &B1; case 2: return &B2; case 3: return &B3;
    case 4: return &B4; case 5: return &B5; case 6: return &B6; default: return &B7;
  }
}

#define TMAP_SELECT_AB(group) \
  const CUtensorMap *A_tmap = tmap_select_A(group, kA0, kA1, kA2, kA3, kA4, kA5, kA6, kA7); \
  const CUtensorMap *B_tmap = tmap_select_B(group, kB0, kB1, kB2, kB3, kB4, kB5, kB6, kB7)

static inline void tmap_replace_addr(CUtensorMap *tmap, uint64_t new_addr) {
  reinterpret_cast<uint64_t*>(tmap)[0] = new_addr;
}

static inline void tmap_replace_global_dim1(CUtensorMap *tmap, uint32_t new_dim_minus1) {
  reinterpret_cast<uint32_t*>(tmap)[9] = new_dim_minus1;
}

static inline void build_A_tmap(CUtensorMap *tmap, uint64_t addr, int M, int K) {
  uint32_t *u = reinterpret_cast<uint32_t*>(tmap);
  memset(u, 0, 128);
  reinterpret_cast<uint64_t*>(u)[0] = addr;
  u[2]  = 0x000665a0u;
  u[3]  = (uint32_t)(K / 32);
  u[4]  = 8u;
  u[8]  = 0xffu;
  u[9]  = (uint32_t)(M - 1);
  u[10] = (uint32_t)(K / 256 - 1);
  u[13] = 0xff000000u;
  u[14] = 0x7fu;
  u[18] = 0x400u;
}

static inline void build_B_tmap(CUtensorMap *tmap, uint64_t addr, int N, int K) {
  uint32_t *u = reinterpret_cast<uint32_t*>(tmap);
  memset(u, 0, 128);
  reinterpret_cast<uint64_t*>(u)[0] = addr;
  u[2]  = 0x000665a0u;  // L2 promotion
  u[3]  = (uint32_t)(K / 32);
  u[4]  = 8u;
  u[8]  = 0xffu;
  u[9]  = (uint32_t)(N - 1);
  u[10] = (uint32_t)(K / 256 - 1);
  u[13] = 0xff000000u;
  u[14] = 0x7fu;
  u[18] = 0x400u;
}

static void init_AB_tmap(
  CUtensorMap *tmap,
  const void *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width,
  CUtensorMapL2promotion l2promo = CU_TENSOR_MAP_L2_PROMOTION_NONE
) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank]       = {256ULL, global_height, global_width / 256ULL};
  uint64_t globalStrides[rank-1] = {global_width / 2ULL, 128ULL};
  uint32_t boxDim[rank]          = {256U, shared_height, shared_width / 256U};
  uint32_t elementStrides[rank]  = {1U, 1U, 1U};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    rank,
    const_cast<void *>(ptr),
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    l2promo,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 256;

constexpr uint64_t EVICT_FIRST  = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST   = 0x14F0000000000000ULL;

constexpr int STAGE_A_SZ   = BLOCK_M * BLOCK_K / 2;
constexpr int STAGE_B_SZ   = BLOCK_N * BLOCK_K / 2;
constexpr int STAGE_SFA_SZ = 128 * BLOCK_K / 16;
constexpr int STAGE_SFB_SZ = 128 * BLOCK_K / 16;
constexpr int STAGE_SZ     = STAGE_A_SZ + STAGE_B_SZ + STAGE_SFA_SZ + STAGE_SFB_SZ;

template <uint64_t CACHE_A = EVICT_LAST, uint64_t CACHE_B = EVICT_FIRST>
__device__ __forceinline__ void issue_tma_loads(
    int smem, int stage_id, int iter_k,
    int off_m, int off_n,
    const CUtensorMap *A_tmap, const CUtensorMap *B_tmap,
    const char *SFA_base, const char *SFB_base,
    int tma_mbar_addr)
{
  const int mbar_addr = tma_mbar_addr + stage_id * 8;
  const int A_smem  = smem + stage_id * STAGE_SZ;
  const int B_smem  = A_smem + STAGE_A_SZ;
  const int SFA_smem = B_smem + STAGE_B_SZ;
  const int SFB_smem = SFA_smem + STAGE_SFA_SZ;

  const int sf_byte = iter_k << 11;
  tma_gmem2smem(SFB_smem, SFB_base + sf_byte, STAGE_SFB_SZ, mbar_addr, CACHE_B);
  tma_gmem2smem(SFA_smem, SFA_base + sf_byte, STAGE_SFA_SZ, mbar_addr, CACHE_A);
  tma_3d_gmem2smem(B_smem, B_tmap, 0, off_n, iter_k, mbar_addr, CACHE_B);
  tma_3d_gmem2smem(A_smem, A_tmap, 0, off_m, iter_k, mbar_addr, CACHE_A);

  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(STAGE_SZ) : "memory");
}

constexpr int NUM_STAGES = 6;

constexpr int ACCUM_STRIDE_TMEM = 128;
constexpr int SCALE_BASE_TMEM   = 2 * ACCUM_STRIDE_TMEM;
constexpr int SFA_TMEM          = SCALE_BASE_TMEM;
constexpr int SFB_TMEM          = SFA_TMEM + 4 * (BLOCK_K / MMA_K);
constexpr int TMEM_ALLOC        = 512;

constexpr int NUM_SMS_TARGET = 148;
constexpr int NUM_SMS_TARGET_N4096_K7168 = 118;
constexpr int NUM_SMS_TARGET_N7168_K2048 = 146;

template <int NUM_K_ITERS = 0, uint64_t CACHE_A = EVICT_LAST, uint64_t CACHE_B = EVICT_FIRST>
__global__ __launch_bounds__(BLOCK_M + 2 * WARP_SIZE, 1)
void cutlass_grouped_kernel(TMAP_KERNEL_PARAMS, const __grid_constant__ Meta kmeta) {
  
  const Meta& meta = kmeta;
//  const Meta *meta = &kmeta;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int lane_id = tid % WARP_SIZE;
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = STAGE_A_SZ;
  constexpr int B_size = STAGE_B_SZ;
  constexpr int SFA_size = STAGE_SFA_SZ;
  constexpr int SFB_size = STAGE_SFB_SZ;
  constexpr int STAGE_SIZE = STAGE_SZ;
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);
  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++) mbarrier_init(tma_mbar_addr + i * 8, 1);
//    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("fence.cta;");
  }
  if (warp_id == NUM_WARPS - 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                :: "r"(smem), "r"(BLOCK_N * 2));
  }
  int group, off_m, off_n;
  decode_tile<BLOCK_M, BLOCK_N>(meta, blockIdx.x, group, off_m, off_n);
  const int M = meta.M[group];
  const int N = meta.N[group];
  const int K = meta.K[group];

  const int num_iters = (NUM_K_ITERS > 0) ? NUM_K_ITERS : (K / BLOCK_K);

  if (warp_id == NUM_WARPS - 2) {
    if (elect_sync()) {
      TMAP_SELECT_AB(group);
      const char *SFA_ptr = reinterpret_cast<const char *>(meta.SFA[group]);
      const char *SFB_ptr = reinterpret_cast<const char *>(meta.SFB[group]);
      const int rest_k = K / 64;
      const int tileA = off_m >> 7;
      const int tileB = off_n >> 7;
      const char *SFA_base = SFA_ptr + (tileA * rest_k) * 512;
      const char *SFB_base = SFB_ptr + (tileB * rest_k) * 512;

      #pragma unroll 1
      for (int iter_k = 0; iter_k < NUM_STAGES && iter_k < num_iters; iter_k++)
        issue_tma_loads<CACHE_A, CACHE_B>(smem, iter_k, iter_k, off_m, off_n, A_tmap, B_tmap, SFA_base, SFB_base, tma_mbar_addr);
      #pragma unroll
      for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
        const int stage_id = iter_k % NUM_STAGES;
        const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
        mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        issue_tma_loads<CACHE_A, CACHE_B>(smem, stage_id, iter_k, off_m, off_n, A_tmap, B_tmap, SFA_base, SFB_base, tma_mbar_addr);
      }
    }
  } else if (warp_id == NUM_WARPS - 1) {
    if (elect_sync()) {
      constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | ((uint32_t)128 >> 7U << 27U);
      const int scaleA_base = SFA_tmem;
      const int scaleB_base = SFB_tmem;

      constexpr auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      constexpr auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        const int stage_id = iter_k % NUM_STAGES;
        const int tma_phase = (iter_k / NUM_STAGES) % 2;
        mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int B_smem = A_smem + A_size;
        const int SFA_smem = B_smem + B_size;
        const int SFB_smem = SFA_smem + SFA_size;

        constexpr uint64_t SF_desc = make_desc_SF(0);
        uint64_t sfa_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
        uint64_t sfb_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);
        uint64_t a_desc = make_desc_AB(A_smem);
        uint64_t b_desc = make_desc_AB(B_smem);

        uint64_t sfa_d[4], sfb_d[4], a_d[4], b_d[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
          sfa_d[k] = sfa_desc; sfb_d[k] = sfb_desc;
          a_d[k] = a_desc; b_d[k] = b_desc;
          sfa_desc += (512ULL >> 4ULL);
          sfb_desc += (512ULL >> 4ULL);
          a_desc   += (32ULL >> 4ULL);
          b_desc   += (32ULL >> 4ULL);
        }

        tcgen05_cp_nvfp4(SFA_tmem + 0 * 4, sfa_d[0]);
        tcgen05_cp_nvfp4(SFB_tmem + 0 * 4, sfb_d[0]);
        tcgen05_cp_nvfp4(SFA_tmem + 1 * 4, sfa_d[1]);
        tcgen05_cp_nvfp4(SFB_tmem + 1 * 4, sfb_d[1]);
        tcgen05_mma_nvfp4(0, a_d[0], b_d[0], i_desc,
                          scaleA_base + 0 * 4, scaleB_base + 0 * 4, iter_k);

        tcgen05_cp_nvfp4(SFA_tmem + 2 * 4, sfa_d[2]);
        tcgen05_cp_nvfp4(SFB_tmem + 2 * 4, sfb_d[2]);
        tcgen05_mma_nvfp4(0, a_d[1], b_d[1], i_desc,
                          scaleA_base + 1 * 4, scaleB_base + 1 * 4, 1);

        tcgen05_cp_nvfp4(SFA_tmem + 3 * 4, sfa_d[3]);
        tcgen05_cp_nvfp4(SFB_tmem + 3 * 4, sfb_d[3]);
        tcgen05_mma_nvfp4(0, a_d[2], b_d[2], i_desc,
                          scaleA_base + 2 * 4, scaleB_base + 2 * 4, 1);

        tcgen05_mma_nvfp4(0, a_d[3], b_d[3], i_desc,
                          scaleA_base + 3 * 4, scaleB_base + 3 * 4, 1);

        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                    :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
      }

      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                  :: "r"(mainloop_mbar_addr) : "memory");
    }
  } else if (tid < BLOCK_M) {
    mbarrier_wait(mainloop_mbar_addr, 0);
//    asm volatile("tcgen05.fence::after_thread_sync;");

    half *C_ptr = reinterpret_cast<half *>(meta.C[group]);
    const int row = off_m + warp_id * 32 + lane_id;
    const bool row_valid = (row < M);
    half *row_ptr = row_valid ? C_ptr + row * N + off_n : nullptr;
    {
      uint32_t bufs[2][8];
      int cur = 0;
      tcgen05_ld_32x32b_pack16(bufs[0], warp_id * 32, 0);
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      #pragma unroll
      for (int col_base = 16; col_base < BLOCK_N; col_base += 16) {
        int nxt = cur ^ 1;
        tcgen05_ld_32x32b_pack16(bufs[nxt], warp_id * 32, col_base);
        if (row_valid) store_cs_32B(row_ptr + col_base - 16,
          bufs[cur][0], bufs[cur][1], bufs[cur][2], bufs[cur][3],
          bufs[cur][4], bufs[cur][5], bufs[cur][6], bufs[cur][7]);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        cur = nxt;
      }
      if (row_valid) store_cs_32B(row_ptr + BLOCK_N - 16,
        bufs[cur][0], bufs[cur][1], bufs[cur][2], bufs[cur][3],
        bufs[cur][4], bufs[cur][5], bufs[cur][6], bufs[cur][7]);
    }

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
    if (warp_id == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 2));
  }
}

template <int NUM_SMS_LIMIT, int NUM_K_ITERS = 0, uint64_t CACHE_A = EVICT_LAST, uint64_t CACHE_B = EVICT_FIRST>
__global__ __launch_bounds__(BLOCK_M + 2 * WARP_SIZE, 1)
void cutlass_grouped_kernel_persistent(TMAP_KERNEL_PARAMS, const __grid_constant__ MetaP kmeta) {
//  const MetaP *meta = &kmeta;
  const MetaP& meta = kmeta;
  const int tid = threadIdx.x;
  const int lane_id = tid & 31;
  const int warp_id = tid >> 5;
  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = STAGE_A_SZ;
  constexpr int B_size = STAGE_B_SZ;
  constexpr int SFA_size = STAGE_SFA_SZ;
  constexpr int SFB_size = STAGE_SFB_SZ;
  constexpr int STAGE_SIZE = STAGE_SZ;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 2];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop0_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int mainloop1_mbar_addr = mainloop0_mbar_addr + 8;

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 2; i++) mbarrier_init(tma_mbar_addr + i * 8, 1);
//    asm volatile("fence.mbarrier_init.release.cluster;");
    asm volatile("fence.cta;");
  }
  if (warp_id == NUM_WARPS - 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                :: "r"(smem), "r"(TMEM_ALLOC));
  }

  int iter = 0;
  int global_iter_base = 0;
  int group_prev = 0, off_m_prev = 0, off_n_prev = 0;

  for (int tile_id = (int)blockIdx.x; tile_id < meta.tiles_count; tile_id += (int)gridDim.x, iter++) {
    const int cur_buf = (iter & 1);
    const int cur_d_tmem = cur_buf * ACCUM_STRIDE_TMEM;
    const int cur_mainloop_mbar = (cur_buf == 0) ? mainloop0_mbar_addr : mainloop1_mbar_addr;

    if (iter && tid < BLOCK_M) {
      const int prev_iter = iter - 1;
      const int pbuf = (prev_iter & 1);
      const int prev_d_tmem = pbuf * ACCUM_STRIDE_TMEM;
      const int prev_mainloop_mbar = (pbuf == 0) ? mainloop0_mbar_addr : mainloop1_mbar_addr;
      const int prev_seq = (prev_iter >> 1);
      const int prev_phase = (prev_seq & 1);

      const int group_p = group_prev;
      const int off_m_p = off_m_prev;
      const int off_n_p = off_n_prev;
      const int M_p = meta.M[group_p];
      const int N_p = meta.N[group_p];
      half *C_ptr_p = reinterpret_cast<half *>(meta.C[group_p]);

      mbarrier_wait(prev_mainloop_mbar, prev_phase);
//      asm volatile("tcgen05.fence::after_thread_sync;");

      
      {
        const int row = off_m_p + warp_id * 32 + lane_id;
        const bool row_valid = (row < M_p);
        half *row_ptr = row_valid ? C_ptr_p + row * N_p + off_n_p : nullptr;
        uint32_t bufs[2][8];
        int cur = 0;
        tcgen05_ld_32x32b_pack16(bufs[0], warp_id * 32, prev_d_tmem + 0);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        #pragma unroll
        for (int col_base = 16; col_base < BLOCK_N; col_base += 16) {
          int nxt = cur ^ 1;
          tcgen05_ld_32x32b_pack16(bufs[nxt], warp_id * 32, prev_d_tmem + col_base);
          if (row_valid) store_cs_32B(row_ptr + col_base - 16,
            bufs[cur][0], bufs[cur][1], bufs[cur][2], bufs[cur][3],
            bufs[cur][4], bufs[cur][5], bufs[cur][6], bufs[cur][7]);
          asm volatile("tcgen05.wait::ld.sync.aligned;");
          cur = nxt;
        }
        if (row_valid) store_cs_32B(row_ptr + BLOCK_N - 16,
          bufs[cur][0], bufs[cur][1], bufs[cur][2], bufs[cur][3],
          bufs[cur][4], bufs[cur][5], bufs[cur][6], bufs[cur][7]);
      }
    }

//    const TileInfo _ti = meta.tiles[tile_id];
//    const int group = _ti.group;
//    const int off_m = _ti.off_m;
//    const int off_n = _ti.off_n;
    const uint32_t t = meta.tiles[tile_id];
    int group, off_m, off_n;
    unpack_tile_u32(t, group, off_m, off_n);
    const int M = meta.M[group];
    const int N = meta.N[group];
    const int K = meta.K[group];
    const int num_iters = (NUM_K_ITERS > 0) ? NUM_K_ITERS : (K / BLOCK_K);

    if (warp_id == NUM_WARPS - 2 && elect_sync()) {
      TMAP_SELECT_AB(group);
      const char *SFA_ptr = reinterpret_cast<const char *>(meta.SFA[group]);
      const char *SFB_ptr = reinterpret_cast<const char *>(meta.SFB[group]);
      const int rest_k = K / 64;
      const int tileA = off_m >> 7;
      const int tileB = off_n >> 7;
      const char *SFA_base = SFA_ptr + (tileA * rest_k) * 512;
      const char *SFB_base = SFB_ptr + (tileB * rest_k) * 512;

      {
        int stage_id = global_iter_base % NUM_STAGES;
        int phase_cnt = global_iter_base / NUM_STAGES;
        #pragma unroll 1
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          const int giter = global_iter_base + iter_k;
          if (giter >= NUM_STAGES) {
            mbarrier_wait(mma_mbar_addr + stage_id * 8, (phase_cnt - 1) & 1);
          }
          issue_tma_loads<CACHE_A, CACHE_B>(smem, stage_id, iter_k, off_m, off_n, A_tmap, B_tmap, SFA_base, SFB_base, tma_mbar_addr);
          if (++stage_id == NUM_STAGES) { stage_id = 0; phase_cnt++; }
        }
      }
    } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
      constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)BLOCK_N >> 3U << 17U) | ((uint32_t)128 >> 7U << 27U);
      const int scaleA_base = SFA_TMEM;
      const int scaleB_base = SFB_TMEM;

      constexpr auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      constexpr auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      {
        int stage_id = global_iter_base % NUM_STAGES;
        int phase_cnt = global_iter_base / NUM_STAGES;
        for (int iter_k = 0; iter_k < num_iters; iter_k++) {
          const int tma_phase = (phase_cnt & 1);

          mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

          const int A_smem = smem + stage_id * STAGE_SIZE;
          const int B_smem = A_smem + A_size;
          const int SFA_smem = B_smem + B_size;
          const int SFB_smem = SFA_smem + SFA_size;

          constexpr uint64_t SF_desc = make_desc_SF(0);
          uint64_t sfa_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
          uint64_t sfb_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);
          uint64_t a_desc = make_desc_AB(A_smem);
          uint64_t b_desc = make_desc_AB(B_smem);

          {
            uint64_t sfa_d[4], sfb_d[4], a_d[4], b_d[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
              sfa_d[k] = sfa_desc; sfb_d[k] = sfb_desc;
              a_d[k] = a_desc; b_d[k] = b_desc;
              sfa_desc += (512ULL >> 4ULL);
              sfb_desc += (512ULL >> 4ULL);
              a_desc += (32ULL >> 4ULL);
              b_desc += (32ULL >> 4ULL);
            }

            tcgen05_cp_nvfp4(SFA_TMEM + 0 * 4, sfa_d[0]);
            tcgen05_cp_nvfp4(SFB_TMEM + 0 * 4, sfb_d[0]);
            tcgen05_cp_nvfp4(SFA_TMEM + 1 * 4, sfa_d[1]);
            tcgen05_cp_nvfp4(SFB_TMEM + 1 * 4, sfb_d[1]);
            tcgen05_mma_nvfp4(cur_d_tmem, a_d[0], b_d[0], i_desc,
                              scaleA_base + 0 * 4, scaleB_base + 0 * 4, iter_k);

            tcgen05_cp_nvfp4(SFA_TMEM + 2 * 4, sfa_d[2]);
            tcgen05_cp_nvfp4(SFB_TMEM + 2 * 4, sfb_d[2]);
            tcgen05_mma_nvfp4(cur_d_tmem, a_d[1], b_d[1], i_desc,
                              scaleA_base + 1 * 4, scaleB_base + 1 * 4, 1);

            tcgen05_cp_nvfp4(SFA_TMEM + 3 * 4, sfa_d[3]);
            tcgen05_cp_nvfp4(SFB_TMEM + 3 * 4, sfb_d[3]);
            tcgen05_mma_nvfp4(cur_d_tmem, a_d[2], b_d[2], i_desc,
                              scaleA_base + 2 * 4, scaleB_base + 2 * 4, 1);

            tcgen05_mma_nvfp4(cur_d_tmem, a_d[3], b_d[3], i_desc,
                              scaleA_base + 3 * 4, scaleB_base + 3 * 4, 1);
          }

          asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                      :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
          if (++stage_id == NUM_STAGES) { stage_id = 0; phase_cnt++; }
        }
      }

      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                  :: "r"(cur_mainloop_mbar) : "memory");
    }
    group_prev = group; off_m_prev = off_m; off_n_prev = off_n;
    global_iter_base += num_iters;
  }

  if (iter && tid < BLOCK_M) {
    const int prev_iter = iter - 1;
    const int pbuf = (prev_iter & 1);
    const int prev_d_tmem = pbuf * ACCUM_STRIDE_TMEM;
    const int prev_mainloop_mbar = (pbuf == 0) ? mainloop0_mbar_addr : mainloop1_mbar_addr;
    const int prev_seq = (prev_iter >> 1);
    const int prev_phase = (prev_seq & 1);

    const int group_p = group_prev;
    const int off_m_p = off_m_prev;
    const int off_n_p = off_n_prev;
    const int M_p = meta.M[group_p];
    const int N_p = meta.N[group_p];
    half *C_ptr_p = reinterpret_cast<half *>(meta.C[group_p]);

    mbarrier_wait(prev_mainloop_mbar, prev_phase);
//    asm volatile("tcgen05.fence::after_thread_sync;");

    
    {
      const int row = off_m_p + warp_id * 32 + lane_id;
      const bool row_valid = (row < M_p);
      half *row_ptr = row_valid ? C_ptr_p + row * N_p + off_n_p : nullptr;
      uint32_t bufs[2][8];
      int cur = 0;
      tcgen05_ld_32x32b_pack16(bufs[0], warp_id * 32, prev_d_tmem + 0);
      asm volatile("tcgen05.wait::ld.sync.aligned;");
      #pragma unroll
      for (int col_base = 16; col_base < BLOCK_N; col_base += 16) {
        int nxt = cur ^ 1;
        tcgen05_ld_32x32b_pack16(bufs[nxt], warp_id * 32, prev_d_tmem + col_base);
        if (row_valid) store_cs_32B(row_ptr + col_base - 16,
          bufs[cur][0], bufs[cur][1], bufs[cur][2], bufs[cur][3],
          bufs[cur][4], bufs[cur][5], bufs[cur][6], bufs[cur][7]);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
        cur = nxt;
      }
      if (row_valid) store_cs_32B(row_ptr + BLOCK_N - 16,
        bufs[cur][0], bufs[cur][1], bufs[cur][2], bufs[cur][3],
        bufs[cur][4], bufs[cur][5], bufs[cur][6], bufs[cur][7]);
    }
  }

  if (tid < BLOCK_M) {
    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
  }
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(TMEM_ALLOC));
  }
}

struct __align__(64) TmapCache {
  int lastN[8] = {};
  int lastK[8] = {};
  CUtensorMap B_template[8];
  DeviceBlob hBlob;
};

static inline void populate_tmaps(
  TmapCache &cache, int G,
  const int64_t* packed_ptrs, const int* ps_ptr,
  uint64_t *C, uint64_t *SFA, uint64_t *SFB,
  int *M, int *N, int *K
) {
  for (int i = 0; i < G; i++) {
    M[i] = ps_ptr[i * 4 + 0];
    N[i] = ps_ptr[i * 4 + 1];
    K[i] = ps_ptr[i * 4 + 2];

    const int64_t* p = packed_ptrs + i * 5;
    const uint64_t Ap = (uint64_t)p[0];
    const uint64_t Bp = (uint64_t)p[1];

    C[i]   = (uint64_t)p[2];
    SFA[i] = (uint64_t)p[3];
    SFB[i] = (uint64_t)p[4];

    build_A_tmap(&cache.hBlob.A[i], Ap, M[i], K[i]);

    const bool bk_changed = (cache.lastN[i] != N[i]) || (cache.lastK[i] != K[i]);
    if (bk_changed) {
      cache.lastN[i] = N[i]; cache.lastK[i] = K[i];
      build_B_tmap(&cache.B_template[i], Bp, N[i], K[i]);
      cache.hBlob.B[i] = cache.B_template[i];
    }
    tmap_replace_addr(&cache.hBlob.B[i], Bp);
  }
}

void dispatch_group_gemm_raw(
  int G,
  const int64_t* packed_ptrs,
  const int* ps_ptr
) {
  thread_local TmapCache cache;

  constexpr int tb = BLOCK_M + 2 * WARP_SIZE;
  constexpr int smem_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2) * NUM_STAGES
                          + 128 * (BLOCK_K / 16) * 2 * NUM_STAGES;
  static_assert(smem_size > 48'000);

  static int inited = 0;
  if (!inited) {
    cudaFuncSetAttribute((cutlass_grouped_kernel<0>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute((cutlass_grouped_kernel<16>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute((cutlass_grouped_kernel<6>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute((cutlass_grouped_kernel_persistent<NUM_SMS_TARGET>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute((cutlass_grouped_kernel_persistent<NUM_SMS_TARGET_N4096_K7168, 28>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute((cutlass_grouped_kernel_persistent<NUM_SMS_TARGET_N7168_K2048, 8, EVICT_LAST, EVICT_LAST>), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    inited = 1;
  }

  const bool use_persistent = (G == 8 && ps_ptr[3] == 1);

  if (use_persistent) {
    MetaP hmeta;
    hmeta.num_groups = G;
    populate_tmaps(cache, G, packed_ptrs, ps_ptr,
                   hmeta.C, hmeta.SFA, hmeta.SFB, hmeta.M, hmeta.N, hmeta.K);

    int total_tiles = 0;
    for (int i = 0; i < G; i++) {
      int tm = (hmeta.M[i] + BLOCK_M - 1) / BLOCK_M;
      int tn = (hmeta.N[i] + BLOCK_N - 1) / BLOCK_N;
      total_tiles += tm * tn;
    }
    if (total_tiles == 0) return;
    hmeta.tiles_count = total_tiles;

    constexpr int SWIZZLE = 2;
    int idx = 0;
    int max_tiles_n = 0;
    for (int g = 0; g < G; g++) {
      int tn = (hmeta.N[g] + BLOCK_N - 1) / BLOCK_N;
      if (tn > max_tiles_n) max_tiles_n = tn;
    }
    for (int nb = 0; nb < max_tiles_n; nb += SWIZZLE) {
      for (int g = 0; g < G; g++) {
        int tiles_n_g = (hmeta.N[g] + BLOCK_N - 1) / BLOCK_N;
        if (nb >= tiles_n_g) continue;
        int tiles_m_g = (hmeta.M[g] + BLOCK_M - 1) / BLOCK_M;
        int swz = (tiles_n_g - nb < SWIZZLE) ? (tiles_n_g - nb) : SWIZZLE;
        for (int m = 0; m < tiles_m_g; m++) {
          for (int ns = 0; ns < swz; ns++) {
            hmeta.tiles[idx++] = pack_tile_u32(g, m, nb + ns);
          }
        }
      }
    }

    const int N0 = hmeta.N[0];
    const int K0 = hmeta.K[0];
    if (N0 == 4096 && K0 == 7168) {
      int grid_x = NUM_SMS_TARGET_N4096_K7168;
      if (grid_x > total_tiles) grid_x = total_tiles;
      cutlass_grouped_kernel_persistent<NUM_SMS_TARGET_N4096_K7168, 28><<<grid_x, tb, smem_size>>>(TMAP_LAUNCH_ARGS(cache.hBlob), hmeta);
    } else if (N0 == 7168 && K0 == 2048) {
      int grid_x = NUM_SMS_TARGET_N7168_K2048;
      if (grid_x > total_tiles) grid_x = total_tiles;
      cutlass_grouped_kernel_persistent<NUM_SMS_TARGET_N7168_K2048, 8, EVICT_LAST, EVICT_LAST><<<grid_x, tb, smem_size>>>(TMAP_LAUNCH_ARGS(cache.hBlob), hmeta);
    } else {
      int grid_x = NUM_SMS_TARGET;
      if (grid_x > total_tiles) grid_x = total_tiles;
      cutlass_grouped_kernel_persistent<NUM_SMS_TARGET><<<grid_x, tb, smem_size>>>(TMAP_LAUNCH_ARGS(cache.hBlob), hmeta);
    }
  } else {
    Meta hmeta;
    hmeta.offsets[0] = 0;
    hmeta.num_groups = G;
    populate_tmaps(cache, G, packed_ptrs, ps_ptr,
                   hmeta.C, hmeta.SFA, hmeta.SFB, hmeta.M, hmeta.N, hmeta.K);

    for (int i = 0; i < G; i++) {
      int tm = (hmeta.M[i] + BLOCK_M - 1) / BLOCK_M;
      int tn = (hmeta.N[i] + BLOCK_N - 1) / BLOCK_N;
      hmeta.offsets[i + 1] = hmeta.offsets[i] + tm * tn;
    }
    const int total_tiles = hmeta.offsets[G];
    if (total_tiles == 0) return;
    hmeta.tiles_count = total_tiles;

    const int K0 = hmeta.K[0];
    if (K0 == 4096) {
      cutlass_grouped_kernel<16><<<total_tiles, tb, smem_size>>>(TMAP_LAUNCH_ARGS(cache.hBlob), hmeta);
    } else if (K0 == 1536) {
      cutlass_grouped_kernel<6><<<total_tiles, tb, smem_size>>>(TMAP_LAUNCH_ARGS(cache.hBlob), hmeta);
    } else {
      cutlass_grouped_kernel<0><<<total_tiles, tb, smem_size>>>(TMAP_LAUNCH_ARGS(cache.hBlob), hmeta);
    }
  }
}
"""

EXT_NAME = "nvfp4_group_gemm_v13b_swizzle2_ext"

_EXT = None

def _ensure_built():
    global _EXT
    if _EXT is not None:
        return
    _EXT = load_inline(
        name=EXT_NAME,
        cpp_sources=CPP_SRC,
        cuda_sources=[CUDA_SRC],
        functions=None,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
            "-gencode=arch=compute_100a,code=sm_100a",
        ],
        extra_ldflags=["-lcuda"],
        with_cuda=True,
        verbose=False,
    )

def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sf_tensors, problem_sizes = data
    _ensure_built()
    _EXT.dispatch_pack_dims(abc_tensors, sf_tensors, problem_sizes)
    return [t[2] for t in abc_tensors]
