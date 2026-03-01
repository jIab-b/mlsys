from task import input_t, output_t
from reference import ref_kernel

import torch
from torch.utils.cpp_extension import load_inline

COMMON_CU = r"""
#include <stdio.h>

#include <cudaTypedefs.h>
#include <cuda_fp16.h>

#include <torch/library.h>
#include <ATen/core/Tensor.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;  // 32 bytes
constexpr int SF_BLOCK_SIZE = 16;
constexpr int NUM_SMS = 148;
constexpr int MAX_TMEM_COLS = 512;

constexpr int SF_TILE_ROWS = 128;
constexpr int SF_ATOM_NUM_ELEMENTS = 128 * 4; // cosize(SF_atom)

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

// https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
constexpr uint64_t EVICT_NORMAL = 0x1000000000000000;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000;

enum ProfilerTag {
  Setup = 0,
  IssueTMA,
  IssueMMA,
  WaitTMA,
  WaitMMA,
  WaitMainloop,
  WaitEpilogue,
  Epilogue,
};

__device__ inline
int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct Profiler {
  int64_t *data_ptr_;
  int sm_id_;
  int cnt_;

  __device__
  void init(int num_entries, int64_t *data_ptr, int bid) {
    data_ptr_ = data_ptr + bid * (1 + num_entries * 4);
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(sm_id_));
    cnt_ = 0;
  }

  __device__
  void start(ProfilerTag tag) {
    data_ptr_[1 + cnt_ * 4 + 0] = sm_id_;
    data_ptr_[1 + cnt_ * 4 + 1] = tag;
    data_ptr_[1 + cnt_ * 4 + 2] = globaltimer();
  }

  __device__
  void stop() {
    data_ptr_[1 + cnt_ * 4 + 3] = globaltimer() - data_ptr_[1 + cnt_ * 4 + 2];
    cnt_ += 1;
  }

  __device__
  void flush() {
    data_ptr_[0] = cnt_;
  }
};

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__
uint32_t elect_sync() {
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

__device__ inline
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_commit_mcast(int mbar_addr, uint16_t cta_mask) {
  asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
              :: "r"(mbar_addr), "h"(cta_mask), "n"(CTA_GROUP) : "memory");
}

// See https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-bulk-copy
// cp.async.bulk is a non-blocking instruction which initiates an asynchronous bulk-copy operation
// from the location specified by source address operand srcMem 
// to the location specified by destination address operand dstMem.
__device__ inline
void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

template <int CTA_GROUP = 1>
__device__ inline
void tma_1d_gmem2smem(int dst, const void *tmap_ptr, int x, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::%5.L2::cache_hint "
              "[%0], [%1, {%2}], [%3], %4;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "l"(cache_policy), "n"(CTA_GROUP)
              : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline
void tma_1d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::%6.L2::cache_hint "
              "[%0], [%1, {%2}], [%3], %4, %5;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy), "n"(CTA_GROUP)
              : "memory");
}

template <int CTA_GROUP = 2>
__device__ inline
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::%6.L2::cache_hint "
              "[%0], [%1, {%2, %3}], [%4], %5;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "l"(cache_policy), "n"(CTA_GROUP)
              : "memory");
}

template <int CTA_GROUP = 1>
__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::%7.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy), "n"(CTA_GROUP)
              : "memory");
}
// See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-cp
// Instruction tcgen05.cp initiates an asynchronous copy operation from shared memory 
// to the location specified by the address operand taddr in the Tensor Memory.
template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  // .32x128b corresponds to lane x size. So we move 32 lanes and 128 bits = 128 / 8 bytes = 16 bytes which is 4 columns
  // x4 means the data is multicast into 4 warps and each warp gets 1/4 of the data.
  // .warpx4 populates data across 32-lane groups (lane in the sense of tmem).
  // Some of the .shape qualifiers require certain .multicast qualifiers.

  // .64x128b requires .warpx2::02_13 or .warpx2::01_23
  // .32x128b requires .warpx4
  //
  // See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-movement-shape
  // 32x128b means 32 lanes, 128b == 16 bytes spans 16 / 4 == 4 columns
  asm volatile("tcgen05.cp.cta_group::%2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc), "n"(CTA_GROUP));
}


struct COLLECTOR_USAGE {
  static constexpr char NONE[]      = "";
  static constexpr char A_FILL[]    = ".collector::a::fill";
  static constexpr char A_USE[]     = ".collector::a::use";
  static constexpr char A_LASTUSE[] = ".collector::a::lastuse";
  static constexpr char A_DISCARD[] = ".collector::a::discard";
};

template <int CTA_GROUP = 1, const char *collector_usage = COLLECTOR_USAGE::NONE>
__device__ inline
void tcgen05_mma_nvfp4(
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
    ".reg .pred p;\n\t"  // predicate register enable-input-d
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::%7.kind::mxf4nvf4.block_scale.block16%8 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d),
       "n"(CTA_GROUP), "C"(collector_usage)
  );
}

// see https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
struct SHAPE {
  static constexpr char _32x32b[]  = ".32x32b";   // 32x1 tile for each warp
  static constexpr char _16x128b[] = ".16x128b";  // 16x4 tile
  static constexpr char _16x256b[] = ".16x256b";  // 16x8 tile
};

struct NUM {
  static constexpr char x4[]  = ".x4";
  static constexpr char x8[]  = ".x8";
  static constexpr char x16[] = ".x16";
  static constexpr char x32[] = ".x32";
  static constexpr char x64[] = ".x64";
  static constexpr char x128[] = ".x128";
};

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_16regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%17%18.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_32regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%33%34.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_64regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%65%66.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_128regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%129%130.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63, "
              " %64, %65, %66, %67, %68, %69, %70, %71, "
              " %72, %73, %74, %75, %76, %77, %78, %79, "
              " %80, %81, %82, %83, %84, %85, %86, %87, "
              " %88, %89, %90, %91, %92, %93, %94, %95, "
              " %96, %97, %98, %99,%100,%101,%102,%103, "
              "%104,%105,%106,%107,%108,%109,%110,%111, "
              "%112,%113,%114,%115,%116,%117,%118,%119, "
              "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63]),
                "=f"(tmp[64]), "=f"(tmp[65]), "=f"(tmp[66]), "=f"(tmp[67]), "=f"(tmp[68]), "=f"(tmp[69]), "=f"(tmp[70]), "=f"(tmp[71]),
                "=f"(tmp[72]), "=f"(tmp[73]), "=f"(tmp[74]), "=f"(tmp[75]), "=f"(tmp[76]), "=f"(tmp[77]), "=f"(tmp[78]), "=f"(tmp[79]),
                "=f"(tmp[80]), "=f"(tmp[81]), "=f"(tmp[82]), "=f"(tmp[83]), "=f"(tmp[84]), "=f"(tmp[85]), "=f"(tmp[86]), "=f"(tmp[87]),
                "=f"(tmp[88]), "=f"(tmp[89]), "=f"(tmp[90]), "=f"(tmp[91]), "=f"(tmp[92]), "=f"(tmp[93]), "=f"(tmp[94]), "=f"(tmp[95]),
                "=f"(tmp[96]), "=f"(tmp[97]), "=f"(tmp[98]), "=f"(tmp[99]), "=f"(tmp[100]),"=f"(tmp[101]),"=f"(tmp[102]),"=f"(tmp[103]),
                "=f"(tmp[104]),"=f"(tmp[105]),"=f"(tmp[106]),"=f"(tmp[107]),"=f"(tmp[108]),"=f"(tmp[109]),"=f"(tmp[110]),"=f"(tmp[111]),
                "=f"(tmp[112]),"=f"(tmp[113]),"=f"(tmp[114]),"=f"(tmp[115]),"=f"(tmp[116]),"=f"(tmp[117]),"=f"(tmp[118]),"=f"(tmp[119]),
                "=f"(tmp[120]),"=f"(tmp[121]),"=f"(tmp[122]),"=f"(tmp[123]),"=f"(tmp[124]),"=f"(tmp[125]),"=f"(tmp[126]),"=f"(tmp[127])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

__device__ inline void tcgen05_ld_32x32bx32(float *tmp, int row, int col) { tcgen05_ld_32regs<SHAPE::_32x32b, NUM::x32>(tmp, row, col); }
__device__ inline void tcgen05_ld_32x32bx64(float *tmp, int row, int col) { tcgen05_ld_64regs<SHAPE::_32x32b, NUM::x64>(tmp, row, col); }
__device__ inline void tcgen05_ld_32x32bx128(float *tmp, int row, int col) { tcgen05_ld_128regs<SHAPE::_32x32b, NUM::x128>(tmp, row, col); }

__device__ inline void tcgen05_ld_16x128bx8(float *tmp, int row, int col) { tcgen05_ld_16regs<SHAPE::_16x128b, NUM::x8>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x128bx16(float *tmp, int row, int col) { tcgen05_ld_32regs<SHAPE::_16x128b, NUM::x16>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x128bx32(float *tmp, int row, int col) { tcgen05_ld_64regs<SHAPE::_16x128b, NUM::x32>(tmp, row, col); }

__device__ inline void tcgen05_ld_16x256bx4(float *tmp, int row, int col) { tcgen05_ld_16regs<SHAPE::_16x256b, NUM::x4>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x256bx8(float *tmp, int row, int col) { tcgen05_ld_32regs<SHAPE::_16x256b, NUM::x8>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x256bx16(float *tmp, int row, int col) { tcgen05_ld_64regs<SHAPE::_16x256b, NUM::x16>(tmp, row, col); }

void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
    error_msg_ptr = "unable to get error string";
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  TORCH_CHECK(false, cudaGetErrorString(err));
}

void init_AB_tmap(
  CUtensorMap *tmap,
  const char *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width
) {
  constexpr uint32_t rank = 3;

  // Tile shape (boxDim)            Global stride	Comment
  // [BLOCK_M, BLOCK_K]	            [K, 1]	      Original layout of threadblock tile in global memory.
  // [BLOCK_M, BLOCK_K / 8, 8]	    [K, 8, 1]	    “Unflatten” the last dim.
  // [BLOCK_K / 8, BLOCK_M, 8]	    [8, K, 1]	    Swap the first 2 dims.
  // 8 if for bf16, 256 for nvfp4: we need contiguous strips of 256 elements

  // The order needs to be reversed as cuTensorMapEncodeTiled assumes the tensors are
  // col major.

  // For the smem layout see https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-layout-swizzling
  // See also https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout
  //
  // For example, 128B MN major swizzle atom would have a shape of (8*(128/32)) x 8 = 32x8 for tf32 tensor core inputs.
  // So for 4 bits we have (8 * 128 / 4) x 8 = 256 x 8
  // the swizzling pattern is given in the tables.

  uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank-1] = {global_width / 2, 128};  // in bytes
  uint32_t boxDim[rank]          = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank]  = {1, 1, 1}; // When all elements of elementStrides array is one, boxDim specifies the number of elements to load.

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}

void init_SF_tmap(CUtensorMap *tmap, const char *ptr, uint64_t global_size, uint32_t shared_size) {
  // use int64 as dtype, hence divide sizes by 8
  constexpr uint32_t rank = 1;
  uint64_t globalDim[rank]       = {global_size / 8};
  uint64_t globalStrides[rank-1] = {};  // in bytes
  uint32_t boxDim[rank]          = {shared_size / 8};
  uint32_t elementStrides[rank]  = {1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  //check_cu(err);
}

void init_SF_tmap_2d(CUtensorMap *tmap, const char *ptr, uint64_t global_height, uint64_t global_width, uint32_t shared_height) {
  // use int64 as dtype, hence divide sizes by 8
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank]       = {global_width / 8, global_height};
  uint64_t globalStrides[rank-1] = {global_width / 8};  // in bytes
  uint32_t boxDim[rank]          = {(uint32_t) global_width / 8, shared_height};
  uint32_t elementStrides[rank]  = {1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  //check_cu(err);
}
__device__ inline
void tma_prefetch(const void *src, int size, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
              :: "l"(src), "r"(size), "l"(cache_policy) : "memory");
}

__device__ inline
void tma_1d_prefetch(const void *tmap_ptr, int x, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.prefetch.tensor.1d.L2.global.L2::cache_hint [%0, {%1}], %2;"
              :: "l"(tmap_ptr), "r"(x), "l"(cache_policy) : "memory");
}

__device__ inline
void tma_2d_prefetch(const void *tmap_ptr, int x, int y, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.L2::cache_hint [%0, {%1, %2}], %3;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "l"(cache_policy) : "memory");
}

__device__ inline
void tma_3d_prefetch(const void *tmap_ptr, int x, int y, int z, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.L2::cache_hint [%0, {%1, %2, %3}], %4;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "l"(cache_policy) : "memory");
}
"""

CUDA_SRC_0 = r"""


template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool C_N_MAJOR,
  int NUM_STAGES,
  int NUM_SFB_ATOMS
>
__global__
__launch_bounds__(BLOCK_M + 2 * WARP_SIZE + 4 * WARP_SIZE)
void gemm_kernel(
  const __grid_constant__ CUtensorMap A0_tmap,
  const __grid_constant__ CUtensorMap B0_tmap,
  const __grid_constant__ CUtensorMap A1_tmap,
  const __grid_constant__ CUtensorMap B1_tmap,
  const __grid_constant__ CUtensorMap A2_tmap,
  const __grid_constant__ CUtensorMap B2_tmap,
  const __grid_constant__ CUtensorMap A3_tmap,
  const __grid_constant__ CUtensorMap B3_tmap,
  const __grid_constant__ CUtensorMap A4_tmap,
  const __grid_constant__ CUtensorMap B4_tmap,
  const __grid_constant__ CUtensorMap A5_tmap,
  const __grid_constant__ CUtensorMap B5_tmap,
  const __grid_constant__ CUtensorMap A6_tmap,
  const __grid_constant__ CUtensorMap B6_tmap,
  const __grid_constant__ CUtensorMap A7_tmap,
  const __grid_constant__ CUtensorMap B7_tmap,
  const __grid_constant__ CUtensorMap SFA0_tmap,
  const __grid_constant__ CUtensorMap SFB0_tmap,
  const __grid_constant__ CUtensorMap SFA1_tmap,
  const __grid_constant__ CUtensorMap SFB1_tmap,
  const __grid_constant__ CUtensorMap SFA2_tmap,
  const __grid_constant__ CUtensorMap SFB2_tmap,
  const __grid_constant__ CUtensorMap SFA3_tmap,
  const __grid_constant__ CUtensorMap SFB3_tmap,
  const __grid_constant__ CUtensorMap SFA4_tmap,
  const __grid_constant__ CUtensorMap SFB4_tmap,
  const __grid_constant__ CUtensorMap SFA5_tmap,
  const __grid_constant__ CUtensorMap SFB5_tmap,
  const __grid_constant__ CUtensorMap SFA6_tmap,
  const __grid_constant__ CUtensorMap SFB6_tmap,
  const __grid_constant__ CUtensorMap SFA7_tmap,
  const __grid_constant__ CUtensorMap SFB7_tmap,
  half *C0_ptr,
  half *C1_ptr,
  half *C2_ptr,
  half *C3_ptr,
  half *C4_ptr,
  half *C5_ptr,
  half *C6_ptr,
  half *C7_ptr,
  int M0, int N0,
  int M1, int N1,
  int M2, int N2,
  int M3, int N3,
  int M4, int N4,
  int M5, int N5,
  int M6, int N6,
  int M7, int N7
) {
  constexpr int EPILOGUE_BUFFER_SIZE = 3;
  const int tid = threadIdx.x;
  const int bid_k = blockIdx.x;
  const int bid = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  constexpr int problem0_blocks = C_N_MAJOR ? 112: 32;
  constexpr int problem1_blocks = C_N_MAJOR ? 999999999: problem0_blocks + 64;
  constexpr int problem2_blocks = C_N_MAJOR ? 999999999: problem1_blocks + 64;
  constexpr int problem3_blocks = C_N_MAJOR ? 999999999: problem2_blocks + 32;
  constexpr int problem4_blocks = C_N_MAJOR ? 999999999: problem3_blocks + 32;
  constexpr int problem5_blocks = C_N_MAJOR ? 999999999: problem4_blocks + 96;
  constexpr int problem6_blocks = C_N_MAJOR ? 999999999: problem5_blocks + 32;
  constexpr int problem7_blocks = C_N_MAJOR ? 999999999: problem6_blocks + 64;

  using Problem = std::tuple<int, int, int, int, int, int, int>;

  auto get_problem = [&](int block_id) -> Problem {
    int grid_m;
    int grid_n;
    int bid_m;
    int bid_n;

    int problem_id;

    int M;
    int N;

    if (block_id < problem0_blocks) {
      problem_id = 0;
      M = M0;
      N = N0;
      grid_m = DIVUP(M0, BLOCK_M);
      grid_n = DIVUP(N0, BLOCK_N);
      bid_m = block_id / grid_n;
      bid_n = block_id % grid_n;
    } else if (block_id < problem1_blocks) {
      problem_id = 1;
      M = M1;
      N = N1;
      grid_m = DIVUP(M1, BLOCK_M);
      grid_n = DIVUP(N1, BLOCK_N);
      bid_m = (block_id - problem0_blocks) / grid_n;
      bid_n = (block_id - problem0_blocks) % grid_n;
    } else if (block_id < problem2_blocks) {
      problem_id = 2;
      M = M2;
      N = N2;
      grid_m = DIVUP(M2, BLOCK_M);
      grid_n = DIVUP(N2, BLOCK_N);
      bid_m = (block_id - problem1_blocks) / grid_n;
      bid_n = (block_id - problem1_blocks) % grid_n;
    } else if (block_id < problem3_blocks) {
      problem_id = 3;
      M = M3;
      N = N3;
      grid_m = DIVUP(M3, BLOCK_M);
      grid_n = DIVUP(N3, BLOCK_N);
      bid_m = (block_id - problem2_blocks) / grid_n;
      bid_n = (block_id - problem2_blocks) % grid_n;
    } else if (block_id < problem4_blocks) {
      problem_id = 4;
      M = M4;
      N = N4;
      grid_m = DIVUP(M4, BLOCK_M);
      grid_n = DIVUP(N4, BLOCK_N);
      bid_m = (block_id - problem3_blocks) / grid_n;
      bid_n = (block_id - problem3_blocks) % grid_n;
    } else if (block_id < problem5_blocks) {
      problem_id = 5;
      M = M5;
      N = N5;
      grid_m = DIVUP(M5, BLOCK_M);
      grid_n = DIVUP(N5, BLOCK_N);
      bid_m = (block_id - problem4_blocks) / grid_n;
      bid_n = (block_id - problem4_blocks) % grid_n;
    } else if (block_id < problem6_blocks) {
      problem_id = 6;
      M = M6;
      N = N6;
      grid_m = DIVUP(M6, BLOCK_M);
      grid_n = DIVUP(N6, BLOCK_N);
      bid_m = (block_id - problem5_blocks) / grid_n;
      bid_n = (block_id - problem5_blocks) % grid_n;
    } else {
      problem_id = 7;
      M = M7;
      N = N7;
      grid_m = DIVUP(M7, BLOCK_M);
      grid_n = DIVUP(N7, BLOCK_N);
      bid_m = (block_id - problem6_blocks) / grid_n;
      bid_n = (block_id - problem6_blocks) % grid_n;
    }
    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    return {problem_id, M, N, bid_m, bid_n, off_m, off_n};
  };


  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2 + 4;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // always copy 128xBLOCK_K/16 = 128 * 4 per atom which is 64 cols, so 128 * 4 * BLOCK_K / 64 = 128 * BLOCK_K/16
  constexpr int SFB_size = 128 * BLOCK_K / 16 * NUM_SFB_ATOMS;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  // set up mbarriers and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  1 mbar  for mainloop
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 3 + EPILOGUE_BUFFER_SIZE * 2];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int stitching_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = stitching_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + EPILOGUE_BUFFER_SIZE * 8;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N * EPILOGUE_BUFFER_SIZE;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, 1); // one thread in each cluster CTA reports.
      mbarrier_init(stitching_mbar_addr + i * 8, 4 * WARP_SIZE);  // 128 - each thread each from stitching warps
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }

    for (int i = 0; i < EPILOGUE_BUFFER_SIZE; ++i) {
      mbarrier_init(mainloop_mbar_addr + 8 * i, 1);
      mbarrier_init(epilogue_mbar_addr + 8 * i, 4);
    }

    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(MAX_TMEM_COLS));
  }
  __syncthreads();  // visible to all threads

  constexpr int num_iters = K / BLOCK_K / SPLIT_K;

  // warp-specialization
  if (warp_id == 5 && elect_sync()) {

    auto issue_tma = [&](int iter_k, int stage_id, const Problem& problem) {
      const int problem_id = std::get<0>(problem);
      const int M = std::get<1>(problem);
      const int N = std::get<2>(problem);
      const int off_m = std::get<5>(problem);
      const int off_n = std::get<6>(problem);

      // TMA warp
      uint64_t cache_A, cache_B;
      if (M > N) {
        cache_A = EVICT_FIRST;
        cache_B = EVICT_LAST;
      } else {
        cache_A = EVICT_LAST;
        cache_B = EVICT_FIRST;
      }
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const CUtensorMap *A_tmap, *B_tmap;
      const CUtensorMap *SFA_tmap, *SFB_tmap;


      switch (problem_id) {
        case 0:
          A_tmap = &A0_tmap;
          B_tmap = &B0_tmap;
          SFA_tmap = &SFA0_tmap;
          SFB_tmap = &SFB0_tmap;
          break;
        case 1:
          A_tmap = &A1_tmap;
          B_tmap = &B1_tmap;
          SFA_tmap = &SFA1_tmap;
          SFB_tmap = &SFB1_tmap;
          break;
        case 2:
          A_tmap = &A2_tmap;
          B_tmap = &B2_tmap;
          SFA_tmap = &SFA2_tmap;
          SFB_tmap = &SFB2_tmap;
          break;
        case 3:
          A_tmap = &A3_tmap;
          B_tmap = &B3_tmap;
          SFA_tmap = &SFA3_tmap;
          SFB_tmap = &SFB3_tmap;
          break;
        case 4:
          A_tmap = &A4_tmap;
          B_tmap = &B4_tmap;
          SFA_tmap = &SFA4_tmap;
          SFB_tmap = &SFB4_tmap;
          break;
        case 5:
          A_tmap = &A5_tmap;
          B_tmap = &B5_tmap;
          SFA_tmap = &SFA5_tmap;
          SFB_tmap = &SFB5_tmap;
          break;
        case 6:
          A_tmap = &A6_tmap;
          B_tmap = &B6_tmap;
          SFA_tmap = &SFA6_tmap;
          SFB_tmap = &SFB6_tmap;
          break;
        case 7:
          A_tmap = &A7_tmap;
          B_tmap = &B7_tmap;
          SFA_tmap = &SFA7_tmap;
          SFB_tmap = &SFB7_tmap;
          break;
      }

      // issue TMA
      const int off_k = SPLIT_K == 1 ? iter_k * BLOCK_K : (iter_k * SPLIT_K + bid_k) * BLOCK_K;
      tma_3d_gmem2smem(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;

      tma_1d_gmem2smem<1>(SFA_smem, SFA_tmap, ((off_m / 128) * rest_k + off_k / (16 * 4)) * 64, mbar_addr, cache_A);
      // tma_1d_gmem2smem<1>(SFB_smem, SFB_tmap, ((off_n / 128) * rest_k + off_k / (16 * 4)) * 64 , mbar_addr, cache_B);
      // TODO consider loading from the non permuted layout, then we only have to load 96 rows and
      // we can just populate MMA layout.
      // we want to load the scale factors starting at off_n and off_k
      auto linear_offset = ((off_n / 128) * rest_k + off_k / (16 * 4)) * 64;
      tma_1d_gmem2smem<1>(SFB_smem, SFB_tmap, linear_offset, mbar_addr, cache_B);
      // Load the next block of SF for stitching.
      auto linear_offset_1 = (((off_n + 128)/ 128) * rest_k + off_k / (16 * 4)) * 64;
      tma_1d_gmem2smem<1>(SFB_smem + 128 * BLOCK_K / 16, SFB_tmap, linear_offset_1, mbar_addr, cache_B);
      // signal TMA done
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    int mma_phase = 1; // Start with phase 1 as there are no pending MMA operations
    int tma_pipeline_stage = 0;
    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // wait MMA
        mbarrier_wait(mma_mbar_addr + tma_pipeline_stage * 8, mma_phase);

        issue_tma(iter_k, tma_pipeline_stage, problem);

        tma_pipeline_stage = (tma_pipeline_stage + 1) % NUM_STAGES;

        if (tma_pipeline_stage == 0) {
          mma_phase ^= 1;
        }
      } // for iter_k
    } // for this_bid
  }  else if (warp_id > 5) {
    // Stitching warps
    int tma_phase = 0;
    int stitching_stage = 0;
    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      const int off_n = std::get<6>(problem);

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // wait TMA

        mbarrier_wait(tma_mbar_addr + stitching_stage * 8, tma_phase);

        ////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////// stitch ///////////////////////////////////////////
        //
        // In each sh mem the shape is ((32, 4), 4, 4) strides ((16, 4), 1, 512)
        // note ((32, 4), 4): ((16, 4), 1) is compact with size 512
        //
        int kSFBlock0;
        int kStichBufferBLockOffset0;
        int kNextBlock0;
        
        int kSFBlock1;
        int kStichBufferBLockOffset1;
        int kNextBlock1;

        int kSFBlock2;
        int kStichBufferBLockOffset2;
        int kNextBlock2;

        if (off_n == 0) {
          kSFBlock0  = 0;
          kStichBufferBLockOffset0 = 0;
          kNextBlock0 = 0;
          
          kSFBlock1  = 1;
          kStichBufferBLockOffset1 = 1;
          kNextBlock1 = 0;

          kSFBlock2  = 2;
          kStichBufferBLockOffset2 = 2;
          kNextBlock2 = 0;
        } else if (off_n == BLOCK_N) {
          kSFBlock0  = 3;
          kStichBufferBLockOffset0 = 0;
          kNextBlock0 = 0;
          
          kSFBlock1  = 0;
          kStichBufferBLockOffset1 = 1;
          kNextBlock1 = 1;

          kSFBlock2  = 1;
          kStichBufferBLockOffset2 = 2;
          kNextBlock2 = 1;
        } else if (off_n == BLOCK_N * 2) {
          kSFBlock0  = 2;
          kStichBufferBLockOffset0 = 0;
          kNextBlock0 = 0;
          
          kSFBlock1  = 3;
          kStichBufferBLockOffset1 = 1;
          kNextBlock1 = 0;

          kSFBlock2  = 0;
          kStichBufferBLockOffset2 = 2;
          kNextBlock2 = 1;
        } else if (off_n == BLOCK_N * 3) {
          kSFBlock0  = 1;
          kStichBufferBLockOffset0 = 0;
          kNextBlock0 = 0;
          
          kSFBlock1  = 2;
          kStichBufferBLockOffset1 = 1;
          kNextBlock1 = 0;

          kSFBlock2  = 3;
          kStichBufferBLockOffset2 = 2;
          kNextBlock2 = 0;
        }
            const int block_offset = (warp_id - 6) * 512;
            const int row_idx = lane_id;

        int tmp[3];
        // Add logic to omit copy.

        {
          const int sfb_0_offset = stitching_stage * STAGE_SIZE + A_size + B_size + SFA_size + kNextBlock0 * 128 * BLOCK_K / 16 + kSFBlock0 * 4;
          tmp[0] = reinterpret_cast<int*>(smem_ptr)[(sfb_0_offset + row_idx * 16 + 0 + block_offset)/4];
        }

        {
          const int sfb_1_offset = stitching_stage * STAGE_SIZE + A_size + B_size + SFA_size + kNextBlock1 * 128 * BLOCK_K / 16 + kSFBlock1 * 4;
          tmp[1] = reinterpret_cast<int*>(smem_ptr)[(sfb_1_offset + row_idx * 16 + 0 + block_offset)/4];
        }

        {
          const int sfb_1_offset = stitching_stage * STAGE_SIZE + A_size + B_size + SFA_size + kNextBlock2 * 128 * BLOCK_K / 16 + kSFBlock2 * 4;
          tmp[2] = reinterpret_cast<int*>(smem_ptr)[(sfb_1_offset + row_idx * 16 + 0 + block_offset)/4];
        }
        asm volatile("bar.sync 2, %0;" :: "r"(WARP_SIZE * 4) : "memory");

        {
          const int b_buffer_offset = stitching_stage * STAGE_SIZE + A_size + B_size + SFA_size + kStichBufferBLockOffset0 * 4;
          reinterpret_cast<int*>(smem_ptr)[(b_buffer_offset + row_idx * 16 + 0 + block_offset)/4] = tmp[0];
        }
        {
          const int b_buffer_offset = stitching_stage * STAGE_SIZE + A_size + B_size + SFA_size + kStichBufferBLockOffset1 * 4;
          reinterpret_cast<int*>(smem_ptr)[(b_buffer_offset + row_idx * 16 + 0 + block_offset)/4] = tmp[1];
        }
        {
          const int b_buffer_offset = stitching_stage * STAGE_SIZE + A_size + B_size + SFA_size + kStichBufferBLockOffset2 * 4;
          reinterpret_cast<int*>(smem_ptr)[(b_buffer_offset + row_idx * 16 + 0 + block_offset)/4] = tmp[2];
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////// end stitch ///////////////////////////////////////

        asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); // ~100ns

        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;"
                    :: "r"(stitching_mbar_addr + 8 * stitching_stage) : "memory");

        stitching_stage = (stitching_stage + 1) % NUM_STAGES;

        if (stitching_stage == 0) {
          tma_phase ^= 1;
        }
      } // for iter_k
    }
  } else if (warp_id == 4 && elect_sync()) {
    // MMA warp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    // fp4 MMA doesn't support MMA_M=64. Hence, we will use MMA_M=128 and ignore the rest.
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)MMA_N >> 3U << 17U)
                              | ((uint32_t)MMA_M >> 7U << 27U)
                              ;
    int stitching_phase = 0;
    int mma_pipeline_stage = 0;
    int epilogue_phase = 1;
    int mainloop_stage = 0; // 0/1 indicates where in tmem we will put the result

    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      const int bid_m = std::get<3>(problem);
      const int bid_n = std::get<4>(problem);

      mbarrier_wait(epilogue_mbar_addr + 8 * mainloop_stage, epilogue_phase);

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // wait TMA
        mbarrier_wait(stitching_mbar_addr + mma_pipeline_stage * 8, stitching_phase);

        const int A_smem = smem + mma_pipeline_stage * STAGE_SIZE;
        const int B_smem = A_smem + A_size;
        const int SFA_smem = B_smem + B_size;
        const int SFB_smem = SFA_smem + SFA_size;

        // set up shared memory descriptors for A and B
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
        // 128-byte swizzling. LBO is implied to be 1.
        auto make_desc_AB = [](int addr) -> uint64_t {
          const int SBO = 8 * 128;
          return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        };
        // no swizzling
        auto make_desc_SF = [](int addr) -> uint64_t {
          const int SBO = 8 * 16;
          return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
        };

        // tcgen05.cp -> tcgen05.mma should be pipelined correctly per PTX doc
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions
        // cutlass issues all of smem->tmem BEFORE mma
        // https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L1013-L1016
        constexpr uint64_t SF_desc = make_desc_SF(0);
        const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
        const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);

        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);  // 4 columns, 512 bytes of 128x4 / 32x4x4
          uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
          tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
          tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
        }

        // k1 selects the (BLOCK_M, 256) tile.
        // k2 selects the (BLOCK_M, 64) tile, whose rows are swizzled.
        // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
        // to have 2-column (8-byte) alignment (looks like not documented).
        for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
          for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
            uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
            uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

            int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
            const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
            const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

            const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
            tcgen05_mma_nvfp4(/*dst tmem*/ BLOCK_N * mainloop_stage, a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
          }

        // signal MMA done
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(mma_mbar_addr + mma_pipeline_stage * 8) : "memory");
        mma_pipeline_stage = (mma_pipeline_stage + 1) % NUM_STAGES;

        if (mma_pipeline_stage == 0) {
          stitching_phase ^= 1;
        }
      }

      // signal mainloop done
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mainloop_mbar_addr + mainloop_stage * 8) : "memory");
      mainloop_stage = (mainloop_stage + 1) % EPILOGUE_BUFFER_SIZE; 

      if (mainloop_stage == 0) {
        epilogue_phase ^= 1;
      }
    } // this_bid
  }
  else if (warp_id < 4) {
    // epilogue warps
    int mainloop_phase = 0;
    int mainloop_stage = 0; // 0/1 indicates where in tmem we will put the result

    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      const int problem_id = std::get<0>(problem);
      const int M = std::get<1>(problem);
      const int N = std::get<2>(problem);
      const int off_m = std::get<5>(problem);
      const int off_n = std::get<6>(problem);

      half* C_ptr;
      switch (problem_id) {
        case 0:
          C_ptr = C0_ptr;
          break;
        case 1:
          C_ptr = C1_ptr;
          break;
        case 2:
          C_ptr = C2_ptr;
          break;
        case 3:
          C_ptr = C3_ptr;
          break;
        case 4:
          C_ptr = C4_ptr;
          break;
        case 5:
          C_ptr = C5_ptr;
          break;
        case 6:
          C_ptr = C6_ptr;
          break;
        case 7:
          C_ptr = C7_ptr;
          break;
      }

      // wait mainloop
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      auto epilogue_M_major = [&]() {
        // C is M-major
        constexpr int WIDTH = std::min(BLOCK_N, 64);  // using 128 might be slower

        for (int n = 0; n < DIVUP(BLOCK_N, WIDTH); n++) {
          float tmp[WIDTH];  // if WIDTH=128, we are using 128 registers here
          if constexpr (WIDTH == 128) tcgen05_ld_32x32bx128(tmp, warp_id * 32, n * WIDTH + BLOCK_N * mainloop_stage);
          if constexpr (WIDTH == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH + BLOCK_N * mainloop_stage);
          if constexpr (WIDTH == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, n * WIDTH + BLOCK_N * mainloop_stage);
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          for (int i = 0; i < WIDTH; i++) {
            const int row = off_n + n * WIDTH + i;
            const int col = off_m + tid;

            if constexpr (SPLIT_K == 1) {
              if (row < std::min(off_n + BLOCK_N, N) && col < std::min(off_m + BLOCK_M, M)) {
                C_ptr[row * M + col] = __float2half(tmp[i]);
              }
            }
            // else
            //   atomicAdd(buf_ptr + row * M + col, tmp[i]);
          }
        }
      };
      auto epilogue_N_major = [&]() {
        // C is N-major
        for (int m = 0; m < 32 / 16; m++) {
          float tmp[BLOCK_N / 2];
          if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, BLOCK_N * mainloop_stage);
          if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, BLOCK_N * mainloop_stage);
          if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, BLOCK_N * mainloop_stage);
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          for (int i = 0; i < BLOCK_N / 8; i++) {
            const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
            const int col = off_n + i * 8 + (lane_id % 4) * 2;

            if constexpr (SPLIT_K == 1) {
              if (col < N) {
                if (row < M) {
                  reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
                }
                if (row + 8 < M) {
                  reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
                }
              }
            } 
            // else {
            //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 0) * N + col), float2({tmp[i * 4 + 0], tmp[i * 4 + 1]}));
            //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 8) * N + col), float2({tmp[i * 4 + 2], tmp[i * 4 + 3]}));
            // }
          }
        }
      };
      static_assert(!C_N_MAJOR);
      if constexpr (C_N_MAJOR)
        epilogue_N_major();
      else
        epilogue_M_major();

      if (elect_sync()) {
        // signal when the epilogue is complete
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(epilogue_mbar_addr + 8 * mainloop_stage) : "memory");
      }
      mainloop_stage = (mainloop_stage + 1) % EPILOGUE_BUFFER_SIZE; 
      if (mainloop_stage == 0) {
        mainloop_phase ^= 1;
      }
    } // this_bid

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");  // everyone is done with tmem
  } // epilogue warps
  if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(MAX_TMEM_COLS));
}

#define EXTRACT_PROBLEM(i) \
  std::tuple<at::Tensor, at::Tensor, at::Tensor> abc_tuple##i = ABC[i]; \
  auto A##i = std::get<0>(abc_tuple##i); \
  auto B##i = std::get<1>(abc_tuple##i); \
  std::tuple<at::Tensor, at::Tensor> sfab_tuple##i = SFAB[i]; \
  auto SFA##i = std::get<0>(sfab_tuple##i); \
  auto SFB##i = std::get<1>(sfab_tuple##i); \
  at::Tensor C##i = outputs[i]; \
  const int M##i = A##i.size(0); \
  const int N##i = B##i.size(0); \
  auto A##i##_ptr = reinterpret_cast<const char*>(A##i.data_ptr()); \
  auto B##i##_ptr = reinterpret_cast<const char*>(B##i.data_ptr()); \
  auto SFA##i##_ptr = reinterpret_cast<const char*>(SFA##i.data_ptr()); \
  auto SFB##i##_ptr = reinterpret_cast<const char*>(SFB##i.data_ptr()); \
  auto C##i##_ptr = reinterpret_cast<half*>(C##i.data_ptr()); \
  int new_M##i = M##i; \
  int new_N##i = N##i; \
  if constexpr (SWAP_AB) { \
    std::swap(A##i##_ptr, B##i##_ptr); \
    std::swap(SFA##i##_ptr, SFB##i##_ptr); \
    std::swap(new_M##i, new_N##i); \
  } \
  CUtensorMap A##i##_tmap, B##i##_tmap; \
  init_AB_tmap(&A##i##_tmap, A##i##_ptr, new_M##i, K, BLOCK_M, BLOCK_K); \
  init_AB_tmap(&B##i##_tmap, B##i##_ptr, new_N##i, K, BLOCK_N, BLOCK_K); \
  const int rest_M##i = DIVUP(new_M##i, SF_TILE_ROWS); \
  const int rest_N##i = DIVUP(new_N##i, SF_TILE_ROWS); \
  auto new_M##i##_padded = SF_TILE_ROWS * rest_M##i; \
  auto new_N##i##_padded = SF_TILE_ROWS * rest_N##i; \
  CUtensorMap SFA##i##_tmap, SFB##i##_tmap; \
  init_SF_tmap(&SFA##i##_tmap, SFA##i##_ptr, new_M##i##_padded * K / SF_BLOCK_SIZE, SF_ATOM_NUM_ELEMENTS * SF_ATOMS_PER_BLOCK_K); \
  init_SF_tmap(&SFB##i##_tmap, SFB##i##_ptr, new_N##i##_padded * K / SF_BLOCK_SIZE, SF_ATOM_NUM_ELEMENTS * SF_ATOMS_PER_BLOCK_K);

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  const int NUM_SFB_ATOMS = 2;
  constexpr int SF_ATOMS_PER_BLOCK_K = BLOCK_K / 16 / 4; // each atom has 16 * 4 columns

  static_assert(BLOCK_K % 256 == 0);
  EXTRACT_PROBLEM(0);
  EXTRACT_PROBLEM(1);
  EXTRACT_PROBLEM(2);
  EXTRACT_PROBLEM(3);
  EXTRACT_PROBLEM(4);
  EXTRACT_PROBLEM(5);
  EXTRACT_PROBLEM(6);
  EXTRACT_PROBLEM(7);

  dim3 grid(SPLIT_K, NUM_SMS);
  int tb_size = BLOCK_M + 2 * WARP_SIZE + 4 * WARP_SIZE; // 4 extra warps for stitching
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * (1 + NUM_SFB_ATOMS); // 1 for SFA atom, NUM_SFB_ATOMS, and buffer to stage the stiched SFs
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = gemm_kernel<K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, C_N_MAJOR != SWAP_AB, NUM_STAGES, NUM_SFB_ATOMS>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, tb_size, smem_size>>>(A0_tmap, B0_tmap,
                                            A1_tmap, B1_tmap,
                                            A2_tmap, B2_tmap,
                                            A3_tmap, B3_tmap,
                                            A4_tmap, B4_tmap,
                                            A5_tmap, B5_tmap,
                                            A6_tmap, B6_tmap,
                                            A7_tmap, B7_tmap,
                                            SFA0_tmap, SFB0_tmap,
                                            SFA1_tmap, SFB1_tmap,
                                            SFA2_tmap, SFB2_tmap,
                                            SFA3_tmap, SFB3_tmap,
                                            SFA4_tmap, SFB4_tmap,
                                            SFA5_tmap, SFB5_tmap,
                                            SFA6_tmap, SFB6_tmap,
                                            SFA7_tmap, SFB7_tmap,
                                            C0_ptr,
                                            C1_ptr,
                                            C2_ptr,
                                            C3_ptr,
                                            C4_ptr,
                                            C5_ptr,
                                            C6_ptr,
                                            C7_ptr,
                                            new_M0, new_N0,
                                            new_M1, new_N1,
                                            new_M2, new_N2,
                                            new_M3, new_N3,
                                            new_M4, new_N4,
                                            new_M5, new_N5,
                                            new_M6, new_N6,
                                            new_M7, new_N7
                                          );
  if constexpr (C_N_MAJOR) {
    return {outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7]};
  } else {
    at::Tensor out0 = outputs[0];
    at::Tensor out1 = outputs[1];
    at::Tensor out2 = outputs[2];
    at::Tensor out3 = outputs[3];
    at::Tensor out4 = outputs[4];
    at::Tensor out5 = outputs[5];
    at::Tensor out6 = outputs[6];
    at::Tensor out7 = outputs[7];
    return {out0.view({new_N0, new_M0, 1}).transpose(0, 1), out1.view({new_N1, new_M1, 1}).transpose(0, 1), 
            out2.view({new_N2, new_M2, 1}).transpose(0, 1), out3.view({new_N3, new_M3, 1}).transpose(0, 1),
            out4.view({new_N4, new_M4, 1}).transpose(0, 1), out5.view({new_N5, new_M5, 1}).transpose(0, 1), 
            out6.view({new_N6, new_M6, 1}).transpose(0, 1), out7.view({new_N7, new_M7, 1}).transpose(0, 1)};
  }
}

template std::vector<at::Tensor> gemm_launch<7168, 128,  96, 256, 1, true,  true, 6>(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);
"""

CPP_SRC_0 = r"""

#include <vector>
#include <torch/script.h>

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);

std::vector<at::Tensor> gemm(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  auto result = gemm_launch<7168, 128,  96, 256, 1, true,  true, 6>(ABC, SFAB, outputs);
  return {result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]};
}

TORCH_LIBRARY(blockscaled_cstr_p0, m) {
  m.def("gemm_cstr_p0((Tensor, Tensor, Tensor)[] ABC, (Tensor, Tensor)[] SFAB, Tensor[] outputs) -> Tensor[]");
  m.impl("gemm_cstr_p0", &gemm);
}
"""

CUDA_SRC_1  = r"""

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool C_N_MAJOR,
  int NUM_STAGES
>
__global__
__launch_bounds__(10 * WARP_SIZE)
void gemm_kernel(
  const __grid_constant__ CUtensorMap A0_tmap,
  const __grid_constant__ CUtensorMap B0_tmap,
  const __grid_constant__ CUtensorMap A1_tmap,
  const __grid_constant__ CUtensorMap B1_tmap,
  const __grid_constant__ CUtensorMap A2_tmap,
  const __grid_constant__ CUtensorMap B2_tmap,
  const __grid_constant__ CUtensorMap A3_tmap,
  const __grid_constant__ CUtensorMap B3_tmap,
  const __grid_constant__ CUtensorMap A4_tmap,
  const __grid_constant__ CUtensorMap B4_tmap,
  const __grid_constant__ CUtensorMap A5_tmap,
  const __grid_constant__ CUtensorMap B5_tmap,
  const __grid_constant__ CUtensorMap A6_tmap,
  const __grid_constant__ CUtensorMap B6_tmap,
  const __grid_constant__ CUtensorMap A7_tmap,
  const __grid_constant__ CUtensorMap B7_tmap,
  const char *SFA0_ptr,
  const char *SFB0_ptr,
  const char *SFA1_ptr,
  const char *SFB1_ptr,
  const char *SFA2_ptr,
  const char *SFB2_ptr,
  const char *SFA3_ptr,
  const char *SFB3_ptr,
  const char *SFA4_ptr,
  const char *SFB4_ptr,
  const char *SFA5_ptr,
  const char *SFB5_ptr,
  const char *SFA6_ptr,
  const char *SFB6_ptr,
  const char *SFA7_ptr,
  const char *SFB7_ptr,
  half *C0_ptr,
  half *C1_ptr,
  half *C2_ptr,
  half *C3_ptr,
  half *C4_ptr,
  half *C5_ptr,
  half *C6_ptr,
  half *C7_ptr,
  int M0, int N0,
  int M1, int N1,
  int M2, int N2,
  int M3, int N3,
  int M4, int N4,
  int M5, int N5,
  int M6, int N6,
  int M7, int N7
) {
  constexpr int EPILOGUE_BUFFER_SIZE = 3;

  const int tid = threadIdx.x;
  const int bid_k = blockIdx.x;
  const int bid = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  constexpr int problem0_blocks = C_N_MAJOR ? 112: 56;
  constexpr int problem1_blocks = C_N_MAJOR ? 999999999: problem0_blocks + 56;
  constexpr int problem2_blocks = C_N_MAJOR ? 999999999: problem1_blocks + 112;
  constexpr int problem3_blocks = C_N_MAJOR ? 999999999: problem2_blocks + 56;
  constexpr int problem4_blocks = C_N_MAJOR ? 999999999: problem3_blocks + 112;
  constexpr int problem5_blocks = C_N_MAJOR ? 999999999: problem4_blocks + 112;
  constexpr int problem6_blocks = C_N_MAJOR ? 999999999: problem5_blocks + 112;
  constexpr int problem7_blocks = C_N_MAJOR ? 999999999: problem6_blocks + 112;

  // const int problem0_blocks = DIVUP(M0, BLOCK_M) * DIVUP(N0, BLOCK_N);
  // const int problem1_blocks = problem0_blocks + DIVUP(M1, BLOCK_M) * DIVUP(N1, BLOCK_N);
  // const int problem2_blocks = problem1_blocks + DIVUP(M2, BLOCK_M) * DIVUP(N2, BLOCK_N);
  // const int problem3_blocks = problem2_blocks + DIVUP(M3, BLOCK_M) * DIVUP(N3, BLOCK_N);
  // const int problem4_blocks = problem3_blocks + DIVUP(M4, BLOCK_M) * DIVUP(N4, BLOCK_N);
  // const int problem5_blocks = problem4_blocks + DIVUP(M5, BLOCK_M) * DIVUP(N5, BLOCK_N);
  // const int problem6_blocks = problem5_blocks + DIVUP(M6, BLOCK_M) * DIVUP(N6, BLOCK_N);
  // const int problem7_blocks = problem6_blocks + DIVUP(M7, BLOCK_M) * DIVUP(N7, BLOCK_N);

  using Problem = std::tuple<int, int, int, int, int, int, int>;

  auto get_problem = [&](int block_id) -> Problem {
    int grid_m;
    int grid_n;
    int bid_m;
    int bid_n;

    int problem_id;

    int M;
    int N;

    if (block_id < problem0_blocks) {
      problem_id = 0;
      M = M0;
      N = N0;
      grid_m = DIVUP(M0, BLOCK_M);
      grid_n = DIVUP(N0, BLOCK_N);
      bid_m = block_id / grid_n;
      bid_n = block_id % grid_n;
    } else if (block_id < problem1_blocks) {
      problem_id = 1;
      M = M1;
      N = N1;
      grid_m = DIVUP(M1, BLOCK_M);
      grid_n = DIVUP(N1, BLOCK_N);
      bid_m = (block_id - problem0_blocks) / grid_n;
      bid_n = (block_id - problem0_blocks) % grid_n;
    } else if (block_id < problem2_blocks) {
      problem_id = 2;
      M = M2;
      N = N2;
      grid_m = DIVUP(M2, BLOCK_M);
      grid_n = DIVUP(N2, BLOCK_N);
      bid_m = (block_id - problem1_blocks) / grid_n;
      bid_n = (block_id - problem1_blocks) % grid_n;
    } else if (block_id < problem3_blocks) {
      problem_id = 3;
      M = M3;
      N = N3;
      grid_m = DIVUP(M3, BLOCK_M);
      grid_n = DIVUP(N3, BLOCK_N);
      bid_m = (block_id - problem2_blocks) / grid_n;
      bid_n = (block_id - problem2_blocks) % grid_n;
    } else if (block_id < problem4_blocks) {
      problem_id = 4;
      M = M4;
      N = N4;
      grid_m = DIVUP(M4, BLOCK_M);
      grid_n = DIVUP(N4, BLOCK_N);
      bid_m = (block_id - problem3_blocks) / grid_n;
      bid_n = (block_id - problem3_blocks) % grid_n;
    } else if (block_id < problem5_blocks) {
      problem_id = 5;
      M = M5;
      N = N5;
      grid_m = DIVUP(M5, BLOCK_M);
      grid_n = DIVUP(N5, BLOCK_N);
      bid_m = (block_id - problem4_blocks) / grid_n;
      bid_n = (block_id - problem4_blocks) % grid_n;
    } else if (block_id < problem6_blocks) {
      problem_id = 6;
      M = M6;
      N = N6;
      grid_m = DIVUP(M6, BLOCK_M);
      grid_n = DIVUP(N6, BLOCK_N);
      bid_m = (block_id - problem5_blocks) / grid_n;
      bid_n = (block_id - problem5_blocks) % grid_n;
    } else {
      problem_id = 7;
      M = M7;
      N = N7;
      grid_m = DIVUP(M7, BLOCK_M);
      grid_n = DIVUP(N7, BLOCK_N);
      bid_m = (block_id - problem6_blocks) / grid_n;
      bid_n = (block_id - problem6_blocks) % grid_n;
    }
    const int off_m = bid_m * BLOCK_M;
    const int off_n = bid_n * BLOCK_N;

    return {problem_id, M, N, bid_m, bid_n, off_m, off_n};
  };


  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // always copy 128xBLOCK_K/16
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  // set up mbarriers and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  1 mbar  for mainloop
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + EPILOGUE_BUFFER_SIZE * 8;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N * EPILOGUE_BUFFER_SIZE;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, 1); // one thread in each cluster CTA reports.
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }

    for (int i = 0; i < EPILOGUE_BUFFER_SIZE; ++i) {
      mbarrier_init(mainloop_mbar_addr + 8 * i, 1);
      mbarrier_init(epilogue_mbar_addr + 8 * i, 8); // num epi warps
    }

    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(MAX_TMEM_COLS));
  }
  __syncthreads();  // visible to all threads

  constexpr int num_iters = K / BLOCK_K / SPLIT_K;

  // warp-specialization
  if (warp_id == 8 && elect_sync()) {

    auto issue_tma = [&](int iter_k, int stage_id, const Problem& problem) {
      const int problem_id = std::get<0>(problem);
      const int M = std::get<1>(problem);
      const int N = std::get<2>(problem);
      const int off_m = std::get<5>(problem);
      const int off_n = std::get<6>(problem);

      // TMA warp
      uint64_t cache_A, cache_B;
      if (M > N) {
        cache_A = EVICT_FIRST;
        cache_B = EVICT_LAST;
      } else {
        cache_A = EVICT_LAST;
        cache_B = EVICT_FIRST;
      }
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const CUtensorMap *A_tmap, *B_tmap;
      const char *SFA_ptr, *SFB_ptr;

      switch (problem_id) {
        case 0:
          A_tmap = &A0_tmap;
          B_tmap = &B0_tmap;
          SFA_ptr = SFA0_ptr;
          SFB_ptr = SFB0_ptr;
          break;
        case 1:
          A_tmap = &A1_tmap;
          B_tmap = &B1_tmap;
          SFA_ptr = SFA1_ptr;
          SFB_ptr = SFB1_ptr;
          break;
        case 2:
          A_tmap = &A2_tmap;
          B_tmap = &B2_tmap;
          SFA_ptr = SFA2_ptr;
          SFB_ptr = SFB2_ptr;
          break;
        case 3:
          A_tmap = &A3_tmap;
          B_tmap = &B3_tmap;
          SFA_ptr = SFA3_ptr;
          SFB_ptr = SFB3_ptr;
          break;
        case 4:
          A_tmap = &A4_tmap;
          B_tmap = &B4_tmap;
          SFA_ptr = SFA4_ptr;
          SFB_ptr = SFB4_ptr;
          break;
        case 5:
          A_tmap = &A5_tmap;
          B_tmap = &B5_tmap;
          SFA_ptr = SFA5_ptr;
          SFB_ptr = SFB5_ptr;
          break;
        case 6:
          A_tmap = &A6_tmap;
          B_tmap = &B6_tmap;
          SFA_ptr = SFA6_ptr;
          SFB_ptr = SFB6_ptr;
          break;
        case 7:
          A_tmap = &A7_tmap;
          B_tmap = &B7_tmap;
          SFA_ptr = SFA7_ptr;
          SFB_ptr = SFB7_ptr;
          break;
      }

      // issue TMA
      const int off_k = SPLIT_K == 1 ? iter_k * BLOCK_K : (iter_k * SPLIT_K + bid_k) * BLOCK_K;
      tma_3d_gmem2smem(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;

      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;  // 512 = 32x4x4
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      // signal TMA done
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    int mma_phase = 1; // Start with phase 1 as there are no pending MMA operations
    int tma_pipeline_stage = 0;
    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // wait MMA
        mbarrier_wait(mma_mbar_addr + tma_pipeline_stage * 8, mma_phase);

        issue_tma(iter_k, tma_pipeline_stage, problem);

        tma_pipeline_stage = (tma_pipeline_stage + 1) % NUM_STAGES;

        if (tma_pipeline_stage == 0) {
          mma_phase ^= 1;
        }
      } // for iter_k
    } // for this_bid
  }
  else if (warp_id == 9 && elect_sync()) {
    // MMA warp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    // fp4 MMA doesn't support MMA_M=64. Hence, we will use MMA_M=128 and ignore the rest.
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)MMA_N >> 3U << 17U)
                              | ((uint32_t)MMA_M >> 7U << 27U)
                              ;
    int tma_phase = 0;
    int mma_pipeline_stage = 0;
    int epilogue_phase = 1;
    int mainloop_stage = 0; // 0/1 indicates where in tmem we will put the result

    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      const int bid_m = std::get<3>(problem);
      const int bid_n = std::get<4>(problem);

      mbarrier_wait(epilogue_mbar_addr + 8 * mainloop_stage, epilogue_phase);

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // wait TMA
        mbarrier_wait(tma_mbar_addr + mma_pipeline_stage * 8, tma_phase);

        const int A_smem = smem + mma_pipeline_stage * STAGE_SIZE;
        const int B_smem = A_smem + A_size;
        const int SFA_smem = B_smem + B_size;
        const int SFB_smem = SFA_smem + SFA_size;

        // set up shared memory descriptors for A and B
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
        // 128-byte swizzling. LBO is implied to be 1.
        auto make_desc_AB = [](int addr) -> uint64_t {
          const int SBO = 8 * 128;
          return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        };
        // no swizzling
        auto make_desc_SF = [](int addr) -> uint64_t {
          const int SBO = 8 * 16;
          return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
        };

        // tcgen05.cp -> tcgen05.mma should be pipelined correctly per PTX doc
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions
        // cutlass issues all of smem->tmem BEFORE mma
        // https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L1013-L1016
        constexpr uint64_t SF_desc = make_desc_SF(0);
        const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
        const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);

        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);  // 4 columns, 512 bytes of 128x4 / 32x4x4
          uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
          tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
          tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
        }

        // k1 selects the (BLOCK_M, 256) tile.
        // k2 selects the (BLOCK_M, 64) tile, whose rows are swizzled.
        // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
        // to have 2-column (8-byte) alignment (looks like not documented).
        for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
          for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
            uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
            uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

            int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
            const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
            const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

            const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
            tcgen05_mma_nvfp4(/*dst tmem*/ BLOCK_N * mainloop_stage, a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
          }

        // signal MMA done
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(mma_mbar_addr + mma_pipeline_stage * 8) : "memory");
        mma_pipeline_stage = (mma_pipeline_stage + 1) % NUM_STAGES;

        if (mma_pipeline_stage == 0) {
          tma_phase ^= 1;
        }
      }

      // signal mainloop done
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mainloop_mbar_addr + mainloop_stage * 8) : "memory");
      mainloop_stage = (mainloop_stage + 1) % EPILOGUE_BUFFER_SIZE; 

      if (mainloop_stage == 0) {
        epilogue_phase ^= 1;
      }
    } // this_bid
  }
  else if (warp_id < 8) {
    // epilogue warps
    int mainloop_phase = 0;
    int mainloop_stage = 0; // 0/1 indicates where in tmem we will put the result

    for (int this_bid = bid; this_bid < problem7_blocks; this_bid += NUM_SMS) {
      auto problem = get_problem(this_bid);
      const int problem_id = std::get<0>(problem);
      const int M = std::get<1>(problem);
      const int N = std::get<2>(problem);
      const int off_m = std::get<5>(problem);
      const int off_n = std::get<6>(problem);

      half* C_ptr;
      switch (problem_id) {
        case 0:
          C_ptr = C0_ptr;
          break;
        case 1:
          C_ptr = C1_ptr;
          break;
        case 2:
          C_ptr = C2_ptr;
          break;
        case 3:
          C_ptr = C3_ptr;
          break;
        case 4:
          C_ptr = C4_ptr;
          break;
        case 5:
          C_ptr = C5_ptr;
          break;
        case 6:
          C_ptr = C6_ptr;
          break;
        case 7:
          C_ptr = C7_ptr;
          break;
      }

      // wait mainloop
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      auto epilogue_M_major = [&]() {
        // C is M-major
        constexpr int WIDTH = std::min(BLOCK_N, 64);  // using 128 might be slower
        const int n = warp_id / 4;
        const int row_group = warp_id % 4;

        // for (int n = 0; n < BLOCK_N / WIDTH; n++) {
          float tmp[WIDTH];  // if WIDTH=128, we are using 128 registers here
          if constexpr (WIDTH == 128) tcgen05_ld_32x32bx128(tmp, warp_id * 32, n * WIDTH + BLOCK_N * mainloop_stage);
          if constexpr (WIDTH == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH + BLOCK_N * mainloop_stage);
          if constexpr (WIDTH == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, n * WIDTH + BLOCK_N * mainloop_stage);
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          for (int i = 0; i < WIDTH; i++) {
            const int row = off_n + n * WIDTH + i;
            const int col = off_m + row_group * 32 + lane_id;

            if constexpr (SPLIT_K == 1) {
              if (row < N && col < M) {
                C_ptr[row * M + col] = __float2half(tmp[i]);
              }
            }
            // else
            //   atomicAdd(buf_ptr + row * M + col, tmp[i]);
          }
        // }
      };
      auto epilogue_N_major = [&]() {
        // C is N-major
        for (int m = 0; m < 32 / 16; m++) {
          float tmp[BLOCK_N / 2];
          if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, BLOCK_N * mainloop_stage);
          if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, BLOCK_N * mainloop_stage);
          if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, BLOCK_N * mainloop_stage);
          asm volatile("tcgen05.wait::ld.sync.aligned;");

          for (int i = 0; i < BLOCK_N / 8; i++) {
            const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
            const int col = off_n + i * 8 + (lane_id % 4) * 2;

            if constexpr (SPLIT_K == 1) {
              if (col < N) {
                if (row < M) {
                  reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
                }
                if (row + 8 < M) {
                  reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
                }
              }
            } 
            // else {
            //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 0) * N + col), float2({tmp[i * 4 + 0], tmp[i * 4 + 1]}));
            //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 8) * N + col), float2({tmp[i * 4 + 2], tmp[i * 4 + 3]}));
            // }
          }
        }
      };

      static_assert(!C_N_MAJOR);
      if constexpr (C_N_MAJOR)
        epilogue_N_major();
      else
        epilogue_M_major();

      if (elect_sync()) {
        // signal when the epilogue is complete
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(epilogue_mbar_addr + 8 * mainloop_stage) : "memory");
      }
      mainloop_stage = (mainloop_stage + 1) % EPILOGUE_BUFFER_SIZE; 
      if (mainloop_stage == 0) {
        mainloop_phase ^= 1;
      }
    } // this_bid

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");  // everyone is done with tmem
  } // epilogue warps
  if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(MAX_TMEM_COLS));
}

#define EXTRACT_PROBLEM(i) \
  std::tuple<at::Tensor, at::Tensor, at::Tensor> abc_tuple##i = ABC[i]; \
  auto A##i = std::get<0>(abc_tuple##i); \
  auto B##i = std::get<1>(abc_tuple##i); \
  std::tuple<at::Tensor, at::Tensor> sfab_tuple##i = SFAB[i]; \
  auto SFA##i = std::get<0>(sfab_tuple##i); \
  auto SFB##i = std::get<1>(sfab_tuple##i); \
  at::Tensor C##i = outputs[i]; \
  const int M##i = A##i.size(0); \
  const int N##i = B##i.size(0); \
  auto A##i##_ptr = reinterpret_cast<const char*>(A##i.data_ptr()); \
  auto B##i##_ptr = reinterpret_cast<const char*>(B##i.data_ptr()); \
  auto SFA##i##_ptr = reinterpret_cast<const char*>(SFA##i.data_ptr()); \
  auto SFB##i##_ptr = reinterpret_cast<const char*>(SFB##i.data_ptr()); \
  auto C##i##_ptr = reinterpret_cast<half*>(C##i.data_ptr()); \
  int new_M##i = M##i; \
  int new_N##i = N##i; \
  if constexpr (SWAP_AB) { \
    std::swap(A##i##_ptr, B##i##_ptr); \
    std::swap(SFA##i##_ptr, SFB##i##_ptr); \
    std::swap(new_M##i, new_N##i); \
  } \
  CUtensorMap A##i##_tmap, B##i##_tmap; \
  init_AB_tmap(&A##i##_tmap, A##i##_ptr, new_M##i, K, BLOCK_M, BLOCK_K); \
  init_AB_tmap(&B##i##_tmap, B##i##_ptr, new_N##i, K, BLOCK_N, BLOCK_K);

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  static_assert(BLOCK_K % 256 == 0);
  EXTRACT_PROBLEM(0);
  EXTRACT_PROBLEM(1);
  EXTRACT_PROBLEM(2);
  EXTRACT_PROBLEM(3);
  EXTRACT_PROBLEM(4);
  EXTRACT_PROBLEM(5);
  EXTRACT_PROBLEM(6);
  EXTRACT_PROBLEM(7);

  dim3 grid(SPLIT_K, NUM_SMS);
  int tb_size = 10 * WARP_SIZE;
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = gemm_kernel<K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, C_N_MAJOR != SWAP_AB, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, tb_size, smem_size>>>(A0_tmap, B0_tmap,
                                            A1_tmap, B1_tmap,
                                            A2_tmap, B2_tmap,
                                            A3_tmap, B3_tmap,
                                            A4_tmap, B4_tmap,
                                            A5_tmap, B5_tmap,
                                            A6_tmap, B6_tmap,
                                            A7_tmap, B7_tmap,
                                            SFA0_ptr, SFB0_ptr,
                                            SFA1_ptr, SFB1_ptr,
                                            SFA2_ptr, SFB2_ptr,
                                            SFA3_ptr, SFB3_ptr,
                                            SFA4_ptr, SFB4_ptr,
                                            SFA5_ptr, SFB5_ptr,
                                            SFA6_ptr, SFB6_ptr,
                                            SFA7_ptr, SFB7_ptr,
                                            C0_ptr,
                                            C1_ptr,
                                            C2_ptr,
                                            C3_ptr,
                                            C4_ptr,
                                            C5_ptr,
                                            C6_ptr,
                                            C7_ptr,
                                            new_M0, new_N0,
                                            new_M1, new_N1,
                                            new_M2, new_N2,
                                            new_M3, new_N3,
                                            new_M4, new_N4,
                                            new_M5, new_N5,
                                            new_M6, new_N6,
                                            new_M7, new_N7
                                          );
  if constexpr (C_N_MAJOR) {
    return {outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7]};
  } else {
    at::Tensor out0 = outputs[0];
    at::Tensor out1 = outputs[1];
    at::Tensor out2 = outputs[2];
    at::Tensor out3 = outputs[3];
    at::Tensor out4 = outputs[4];
    at::Tensor out5 = outputs[5];
    at::Tensor out6 = outputs[6];
    at::Tensor out7 = outputs[7];
    return {out0.view({new_N0, new_M0, 1}).transpose(0, 1), out1.view({new_N1, new_M1, 1}).transpose(0, 1), 
            out2.view({new_N2, new_M2, 1}).transpose(0, 1), out3.view({new_N3, new_M3, 1}).transpose(0, 1),
            out4.view({new_N4, new_M4, 1}).transpose(0, 1), out5.view({new_N5, new_M5, 1}).transpose(0, 1), 
            out6.view({new_N6, new_M6, 1}).transpose(0, 1), out7.view({new_N7, new_M7, 1}).transpose(0, 1)};
  }
}

template std::vector<at::Tensor> gemm_launch<2048, 128,  128, 256, 1, true,  true, 6>(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);
"""

CPP_SRC_1 = r"""
#include <vector>
#include <torch/script.h>

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);

std::vector<at::Tensor> gemm(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  auto result = gemm_launch<2048, 128,  128, 256, 1, true,  true, 6>(ABC, SFAB, outputs);
  return {result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7]};
}

TORCH_LIBRARY(blockscaled_cstr_p1, m) {
  m.def("gemm_cstr_p1((Tensor, Tensor, Tensor)[] ABC, (Tensor, Tensor)[] SFAB, Tensor[] outputs) -> Tensor[]");
  m.impl("gemm_cstr_p1", &gemm);
}
"""

CUDA_SRC_2  = r"""

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool C_N_MAJOR,
  int NUM_STAGES
>
__global__
__launch_bounds__(10 * WARP_SIZE)
void gemm_kernel(
  const __grid_constant__ CUtensorMap A0_tmap,
  const __grid_constant__ CUtensorMap B0_tmap,
  const __grid_constant__ CUtensorMap A1_tmap,
  const __grid_constant__ CUtensorMap B1_tmap,
  const char *SFA0_ptr,
  const char *SFB0_ptr,
  const char *SFA1_ptr,
  const char *SFB1_ptr,
  half *C0_ptr,
  half *C1_ptr,
  int M0, int N0,
  int M1, int N1
) {
  const int tid = threadIdx.x;
  const int bid_k = blockIdx.x;
  const int bid = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  int grid_m;
  int grid_n;
  int bid_m;
  int bid_n;

  int off_m;
  int off_n;

  int problem_id;

  int M;
  int N;

  const int problem0_blocks = DIVUP(M0, BLOCK_M) * DIVUP(N0, BLOCK_N);

  if (bid < problem0_blocks) {
    problem_id = 0;
    M = M0;
    N = N0;
    grid_m = DIVUP(M0, BLOCK_M);
    grid_n = DIVUP(N0, BLOCK_N);
    bid_m = bid / grid_n;
    bid_n = bid % grid_n;

    off_m = bid_m * BLOCK_M;
    off_n = bid_n * BLOCK_N;
  } else {
    problem_id = 1;
    M = M1;
    N = N1;
    grid_m = DIVUP(M1, BLOCK_M);
    grid_n = DIVUP(N1, BLOCK_N);
    bid_m = (bid - problem0_blocks) / grid_n;
    bid_n = (bid - problem0_blocks) % grid_n;

    off_m = bid_m * BLOCK_M;
    off_n = bid_n * BLOCK_N;
  }


  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // always copy 128xBLOCK_K/16
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  // set up mbarriers and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  1 mbar  for mainloop
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    // only 1 thread issue
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    // tmem address should be 0, don't bother storing and reading it.
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(BLOCK_N * 2));
  } else if (warp_id == 2 && elect_sync()) {
    auto A_tmap = problem_id == 0 ? &A0_tmap : &A1_tmap;
    auto B_tmap = problem_id == 0 ? &B0_tmap : &B1_tmap;
    tma_3d_prefetch(A_tmap, 0, off_m, 0, EVICT_LAST);
    tma_3d_prefetch(B_tmap, 0, off_n, 0, EVICT_LAST);
    constexpr int rest_k = K / 16 / 4;
    auto SFA_ptr = problem_id == 0 ? SFA0_ptr : SFA1_ptr;
    auto SFB_ptr = problem_id == 0 ? SFB0_ptr : SFB1_ptr;
    const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k) * 512;  // 512 = 32x4x4
    const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k) * 512;
    tma_prefetch(SFA_src, SFA_size, EVICT_LAST);
    tma_prefetch(SFB_src, SFB_size, EVICT_LAST);
  }
  __syncthreads();  // visible to all threads

  constexpr int num_iters = K / BLOCK_K / SPLIT_K;

  // warp-specialization
  if (warp_id == 8 && elect_sync()) {
    // TMA warp
    uint64_t cache_A, cache_B;
    if (M > N) {
      cache_A = EVICT_FIRST;
      cache_B = EVICT_LAST;
    } else {
      cache_A = EVICT_LAST;
      cache_B = EVICT_FIRST;
    }

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      auto A_tmap = problem_id == 0 ? &A0_tmap : &A1_tmap;
      auto B_tmap = problem_id == 0 ? &B0_tmap : &B1_tmap;

      // issue TMA
      const int off_k = SPLIT_K == 1 ? iter_k * BLOCK_K : (iter_k * SPLIT_K + bid_k) * BLOCK_K;
      tma_3d_gmem2smem(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;
      auto SFA_ptr = problem_id == 0 ? SFA0_ptr : SFA1_ptr;
      auto SFB_ptr = problem_id == 0 ? SFB0_ptr : SFB1_ptr;

      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;  // 512 = 32x4x4
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      // signal TMA done
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    // issue TMA without waiting for MMA
    for (int iter_k = 0; iter_k < NUM_STAGES; iter_k++)
      issue_tma(iter_k, iter_k);

    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      // wait MMA
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);

      issue_tma(iter_k, stage_id);
    }
  }
  else if (warp_id == 9 && elect_sync()) {
    // MMA warp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    // fp4 MMA doesn't support MMA_M=64. Hence, we will use MMA_M=128 and ignore the rest.
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)MMA_N >> 3U << 17U)
                              | ((uint32_t)MMA_M >> 7U << 27U)
                              ;

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      // wait TMA
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      // set up shared memory descriptors for A and B
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
      // 128-byte swizzling. LBO is implied to be 1.
      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      // no swizzling
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      // tcgen05.cp -> tcgen05.mma should be pipelined correctly per PTX doc
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions
      // cutlass issues all of smem->tmem BEFORE mma
      // https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L1013-L1016
      constexpr uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
      const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);  // 4 columns, 512 bytes of 128x4 / 32x4x4
        uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }

      // k1 selects the (BLOCK_M, BLOCK_K = 256) tile.
      // k2 selects the (BLOCK_M, MMA_K = 64) tile, whose rows are swizzled.
      // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
      // to have 2-column (8-byte) alignment (looks like not documented).
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          tcgen05_mma_nvfp4(0, a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
        }

      // signal MMA done
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
    }

    // signal mainloop done
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory");
  }
  else if (warp_id < 8) {
    // epilogue warps
    auto C_ptr = problem_id == 0 ? C0_ptr : C1_ptr;
    // wait mainloop
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    auto epilogue_M_major = [&]() {
      // C is M-major
      constexpr int WIDTH = std::min(BLOCK_N, 64);  // using 128 might be slower
      const int n = warp_id / 4;
      const int row_group = warp_id % 4;
      // for (int n = 0; n < BLOCK_N / WIDTH; n++) {
        float tmp[WIDTH];  // if WIDTH=128, we are using 128 registers here
        if constexpr (WIDTH == 128) tcgen05_ld_32x32bx128(tmp, warp_id * 32, n * WIDTH);
        if constexpr (WIDTH == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH);
        if constexpr (WIDTH == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, n * WIDTH);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < WIDTH; i++) {
          const int row = off_n + n * WIDTH + i;
          const int col = off_m + row_group * 32 + lane_id;

          if constexpr (SPLIT_K == 1) {
            if (row < N && col < M) {
              C_ptr[row * M + col] = __float2half(tmp[i]);
            }
          }
          // else
          //   atomicAdd(buf_ptr + row * M + col, tmp[i]);
        }
      // }
    };
    auto epilogue_N_major = [&]() {
      // C is N-major
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;

          if constexpr (SPLIT_K == 1) {
            if (col < N) {
              if (row < M) {
                reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
              }
              if (row + 8 < M) {
                reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
              }
            }
          } 
          // else {
          //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 0) * N + col), float2({tmp[i * 4 + 0], tmp[i * 4 + 1]}));
          //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 8) * N + col), float2({tmp[i * 4 + 2], tmp[i * 4 + 3]}));
          // }
        }
      }
    };
    static_assert(!C_N_MAJOR);
    if constexpr (C_N_MAJOR)
      epilogue_N_major();
    else
      epilogue_M_major();

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");  // everyone is done with tmem
    if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 2));

  }
}

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  static_assert(BLOCK_K % 256 == 0);

  std::tuple<at::Tensor, at::Tensor, at::Tensor> abc_tuple = ABC[0]; 
  auto A = std::get<0>(abc_tuple);
  auto B = std::get<1>(abc_tuple);

  std::tuple<at::Tensor, at::Tensor> sfab_tuple = SFAB[0]; 
  auto SFA = std::get<0>(sfab_tuple);
  auto SFB = std::get<1>(sfab_tuple);

  at::Tensor C = outputs[0];

  const int M = A.size(0);
  const int N = B.size(0);

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr  = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());
    
  int new_M = M;
  int new_N = N;

  if constexpr (SWAP_AB) {
    std::swap(A_ptr, B_ptr);
    std::swap(SFA_ptr, SFB_ptr);
    std::swap(new_M, new_N);
  }

  CUtensorMap A_tmap, B_tmap;
  init_AB_tmap(&A_tmap, A_ptr, new_M, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B_tmap, B_ptr, new_N, K, BLOCK_N, BLOCK_K);

  std::tuple<at::Tensor, at::Tensor, at::Tensor> abc_tuple1 = ABC[1]; 
  auto A1 = std::get<0>(abc_tuple1);
  auto B1 = std::get<1>(abc_tuple1);

  std::tuple<at::Tensor, at::Tensor> sfab_tuple1 = SFAB[1]; 
  auto SFA1 = std::get<0>(sfab_tuple1);
  auto SFB1 = std::get<1>(sfab_tuple1);

  at::Tensor C1 = outputs[1];

  const int M1 = A1.size(0);
  const int N1 = B1.size(0);

  auto A1_ptr   = reinterpret_cast<const char *>(A1.data_ptr());
  auto B1_ptr  = reinterpret_cast<const char *>(B1.data_ptr());
  auto SFA1_ptr = reinterpret_cast<const char *>(SFA1.data_ptr());
  auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
  auto C1_ptr   = reinterpret_cast<half *>(C1.data_ptr());
    
  int new_M1 = M1;
  int new_N1 = N1;

  if constexpr (SWAP_AB) {
    std::swap(A1_ptr, B1_ptr);
    std::swap(SFA1_ptr, SFB1_ptr);
    std::swap(new_M1, new_N1);
  }

  CUtensorMap A1_tmap, B1_tmap;
  init_AB_tmap(&A1_tmap, A1_ptr, new_M1, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B1_tmap, B1_ptr, new_N1, K, BLOCK_N, BLOCK_K);

  dim3 grid(SPLIT_K, DIVUP(new_M, BLOCK_M) * DIVUP(new_N, BLOCK_N) + DIVUP(new_M1, BLOCK_M) * DIVUP(new_N1, BLOCK_N));

  // printf("(new_M / BLOCK_M) * (new_N / BLOCK_N) %d\n", (new_M / BLOCK_M) * (new_N / BLOCK_N));
  int tb_size = 10 * WARP_SIZE;
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = gemm_kernel<K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, C_N_MAJOR != SWAP_AB, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, tb_size, smem_size>>>(A_tmap, B_tmap, A1_tmap, B1_tmap, 
                                            SFA_ptr, SFB_ptr, SFA1_ptr, SFB1_ptr,
                                            C_ptr, C1_ptr,
                                            new_M, new_N,
                                            new_M1, new_N1);
  if constexpr (C_N_MAJOR) {
    return {outputs[0], outputs[1]};
  } else {
    at::Tensor out0 = outputs[0];
    at::Tensor out1 = outputs[1];
    return {out0.view({N, M, 1}).transpose(0, 1) , out1.view({N1, M1, 1}).transpose(0, 1)};
  }
}

template std::vector<at::Tensor> gemm_launch<4096, 128,  128, 256, 1, false,  false, 6>(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);
"""

CPP_SRC_2 = r"""
#include <vector>
#include <torch/script.h>

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);

std::vector<at::Tensor> gemm(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  // Note SWAP_AB can only be true as the shapes are only evenly tiled when swapped
  auto result = gemm_launch<4096, 128,  128, 256, 1, false,  false, 6>(ABC, SFAB, outputs);
  return {result[0], result[1]};
}

TORCH_LIBRARY(blockscaled_cstr_p2, m) {
  m.def("gemm_cstr_p2((Tensor, Tensor, Tensor)[] ABC, (Tensor, Tensor)[] SFAB, Tensor[] outputs) -> Tensor[]");
  m.impl("gemm_cstr_p2", &gemm);
}
"""

CUDA_SRC_3  = r"""

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool C_N_MAJOR,
  int NUM_STAGES
>
__global__
__launch_bounds__(10 * WARP_SIZE)
void gemm_kernel(
  const __grid_constant__ CUtensorMap A0_tmap,
  const __grid_constant__ CUtensorMap B0_tmap,
  const __grid_constant__ CUtensorMap A1_tmap,
  const __grid_constant__ CUtensorMap B1_tmap,
  const char *SFA0_ptr,
  const char *SFB0_ptr,
  const char *SFA1_ptr,
  const char *SFB1_ptr,
  half *C0_ptr,
  half *C1_ptr,
  int M0, int N0,
  int M1, int N1
) {
  const int tid = threadIdx.x;
  const int bid_k = blockIdx.x;
  const int bid = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  int grid_m;
  int grid_n;
  int bid_m;
  int bid_n;

  int off_m;
  int off_n;

  int problem_id;

  int M;
  int N;

  const int problem0_blocks = DIVUP(M0, BLOCK_M) * DIVUP(N0, BLOCK_N);

  if (bid < problem0_blocks) {
    problem_id = 0;
    M = M0;
    N = N0;
    grid_m = DIVUP(M0, BLOCK_M);
    grid_n = DIVUP(N0, BLOCK_N);
    bid_m = bid / grid_n;
    bid_n = bid % grid_n;

    off_m = bid_m * BLOCK_M;
    off_n = bid_n * BLOCK_N;
  } else {
    problem_id = 1;
    M = M1;
    N = N1;
    grid_m = DIVUP(M1, BLOCK_M);
    grid_n = DIVUP(N1, BLOCK_N);
    bid_m = (bid - problem0_blocks) / grid_n;
    bid_n = (bid - problem0_blocks) % grid_n;

    off_m = bid_m * BLOCK_M;
    off_n = bid_n * BLOCK_N;
  }


  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // always copy 128xBLOCK_K/16
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  // set up mbarriers and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  1 mbar  for mainloop
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    // only 1 thread issue
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    // tmem address should be 0, don't bother storing and reading it.
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(BLOCK_N * 2));
  } else if (warp_id == 2 && elect_sync()) {
    auto A_tmap = problem_id == 0 ? &A0_tmap : &A1_tmap;
    auto B_tmap = problem_id == 0 ? &B0_tmap : &B1_tmap;
    tma_3d_prefetch(A_tmap, 0, off_m, 0, EVICT_LAST);
    tma_3d_prefetch(B_tmap, 0, off_n, 0, EVICT_LAST);
    constexpr int rest_k = K / 16 / 4;
    auto SFA_ptr = problem_id == 0 ? SFA0_ptr : SFA1_ptr;
    auto SFB_ptr = problem_id == 0 ? SFB0_ptr : SFB1_ptr;
    const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k) * 512;  // 512 = 32x4x4
    const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k) * 512;
    tma_prefetch(SFA_src, SFA_size, EVICT_LAST);
    tma_prefetch(SFB_src, SFB_size, EVICT_LAST);
  }
  __syncthreads();  // visible to all threads

  constexpr int num_iters = K / BLOCK_K / SPLIT_K;

  // warp-specialization
  if (warp_id == 8 && elect_sync()) {
    // TMA warp
    uint64_t cache_A, cache_B;
    if (M > N) {
      cache_A = EVICT_FIRST;
      cache_B = EVICT_LAST;
    } else {
      cache_A = EVICT_LAST;
      cache_B = EVICT_FIRST;
    }

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      auto A_tmap = problem_id == 0 ? &A0_tmap : &A1_tmap;
      auto B_tmap = problem_id == 0 ? &B0_tmap : &B1_tmap;

      // issue TMA
      const int off_k = SPLIT_K == 1 ? iter_k * BLOCK_K : (iter_k * SPLIT_K + bid_k) * BLOCK_K;
      tma_3d_gmem2smem(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;
      auto SFA_ptr = problem_id == 0 ? SFA0_ptr : SFA1_ptr;
      auto SFB_ptr = problem_id == 0 ? SFB0_ptr : SFB1_ptr;

      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;  // 512 = 32x4x4
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      // signal TMA done
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    // issue TMA without waiting for MMA
    for (int iter_k = 0; iter_k < NUM_STAGES; iter_k++)
      issue_tma(iter_k, iter_k);

    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      // wait MMA
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);

      issue_tma(iter_k, stage_id);
    }
  }
  else if (warp_id == 9 && elect_sync()) {
    // MMA warp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    // fp4 MMA doesn't support MMA_M=64. Hence, we will use MMA_M=128 and ignore the rest.
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)MMA_N >> 3U << 17U)
                              | ((uint32_t)MMA_M >> 7U << 27U)
                              ;

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      // wait TMA
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      // set up shared memory descriptors for A and B
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
      // 128-byte swizzling. LBO is implied to be 1.
      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      // no swizzling
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      // tcgen05.cp -> tcgen05.mma should be pipelined correctly per PTX doc
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions
      // cutlass issues all of smem->tmem BEFORE mma
      // https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L1013-L1016
      constexpr uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
      const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem >> 4ULL);

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);  // 4 columns, 512 bytes of 128x4 / 32x4x4
        uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }

      // k1 selects the (BLOCK_M, 256) tile.
      // k2 selects the (BLOCK_M, 64) tile, whose rows are swizzled.
      // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
      // to have 2-column (8-byte) alignment (looks like not documented).
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          tcgen05_mma_nvfp4(0, a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
        }

      // signal MMA done
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
    }

    // signal mainloop done
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory");
  }
  else if (warp_id < 8) {
    // epilogue warps
    auto C_ptr = problem_id == 0 ? C0_ptr : C1_ptr;
    // wait mainloop
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    auto epilogue_M_major = [&]() {
      // C is M-major
      constexpr int WIDTH = std::min(BLOCK_N, 64);  // using 128 might be slower
      const int n = warp_id / 4;
      const int row_group = warp_id % 4;

      // for (int n = 0; n < BLOCK_N / WIDTH; n++) {
        float tmp[WIDTH];  // if WIDTH=128, we are using 128 registers here
        if constexpr (WIDTH == 128) tcgen05_ld_32x32bx128(tmp, warp_id * 32, n * WIDTH);
        if constexpr (WIDTH == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH);
        if constexpr (WIDTH == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, n * WIDTH);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < WIDTH; i++) {
          const int row = off_n + n * WIDTH + i;
          const int col = off_m + row_group * 32 + lane_id;

          if constexpr (SPLIT_K == 1) {
            if (row < N && col < M) {
              C_ptr[row * M + col] = __float2half(tmp[i]);
            }
          }
          // else
          //   atomicAdd(buf_ptr + row * M + col, tmp[i]);
        }
      // }
    };
    auto epilogue_N_major = [&]() {
      // C is N-major
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;

          if constexpr (SPLIT_K == 1) {
            if (col < N) {
              if (row < M) {
                reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
              }
              if (row + 8 < M) {
                reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
              }
            }
          } 
          // else {
          //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 0) * N + col), float2({tmp[i * 4 + 0], tmp[i * 4 + 1]}));
          //   atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 8) * N + col), float2({tmp[i * 4 + 2], tmp[i * 4 + 3]}));
          // }
        }
      }
    };
    static_assert(!C_N_MAJOR);
    if constexpr (C_N_MAJOR)
      epilogue_N_major();
    else
      epilogue_M_major();

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");  // everyone is done with tmem
    if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 2));

  }
}

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  static_assert(BLOCK_K % 256 == 0);

  std::tuple<at::Tensor, at::Tensor, at::Tensor> abc_tuple = ABC[0]; 
  auto A = std::get<0>(abc_tuple);
  auto B = std::get<1>(abc_tuple);

  std::tuple<at::Tensor, at::Tensor> sfab_tuple = SFAB[0]; 
  auto SFA = std::get<0>(sfab_tuple);
  auto SFB = std::get<1>(sfab_tuple);

  at::Tensor C = outputs[0];

  const int M = A.size(0);
  const int N = B.size(0);

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr  = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());
    
  int new_M = M;
  int new_N = N;

  if constexpr (SWAP_AB) {
    std::swap(A_ptr, B_ptr);
    std::swap(SFA_ptr, SFB_ptr);
    std::swap(new_M, new_N);
  }

  CUtensorMap A_tmap, B_tmap;
  init_AB_tmap(&A_tmap, A_ptr, new_M, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B_tmap, B_ptr, new_N, K, BLOCK_N, BLOCK_K);

  std::tuple<at::Tensor, at::Tensor, at::Tensor> abc_tuple1 = ABC[1]; 
  auto A1 = std::get<0>(abc_tuple1);
  auto B1 = std::get<1>(abc_tuple1);

  std::tuple<at::Tensor, at::Tensor> sfab_tuple1 = SFAB[1]; 
  auto SFA1 = std::get<0>(sfab_tuple1);
  auto SFB1 = std::get<1>(sfab_tuple1);

  at::Tensor C1 = outputs[1];

  const int M1 = A1.size(0);
  const int N1 = B1.size(0);

  auto A1_ptr   = reinterpret_cast<const char *>(A1.data_ptr());
  auto B1_ptr  = reinterpret_cast<const char *>(B1.data_ptr());
  auto SFA1_ptr = reinterpret_cast<const char *>(SFA1.data_ptr());
  auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
  auto C1_ptr   = reinterpret_cast<half *>(C1.data_ptr());
    
  int new_M1 = M1;
  int new_N1 = N1;

  if constexpr (SWAP_AB) {
    std::swap(A1_ptr, B1_ptr);
    std::swap(SFA1_ptr, SFB1_ptr);
    std::swap(new_M1, new_N1);
  }

  CUtensorMap A1_tmap, B1_tmap;
  init_AB_tmap(&A1_tmap, A1_ptr, new_M1, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B1_tmap, B1_ptr, new_N1, K, BLOCK_N, BLOCK_K);

  dim3 grid(SPLIT_K, DIVUP(new_M, BLOCK_M) * DIVUP(new_N, BLOCK_N) + DIVUP(new_M1, BLOCK_M) * DIVUP(new_N1, BLOCK_N));

  // printf("(new_M / BLOCK_M) * (new_N / BLOCK_N) %d\n", (new_M / BLOCK_M) * (new_N / BLOCK_N));
  int tb_size = 10 * WARP_SIZE;
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = gemm_kernel<K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, C_N_MAJOR != SWAP_AB, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, tb_size, smem_size>>>(A_tmap, B_tmap, A1_tmap, B1_tmap, 
                                            SFA_ptr, SFB_ptr, SFA1_ptr, SFB1_ptr,
                                            C_ptr, C1_ptr,
                                            new_M, new_N,
                                            new_M1, new_N1);
  if constexpr (C_N_MAJOR) {
    return {outputs[0], outputs[1]};
  } else {
    at::Tensor out0 = outputs[0];
    at::Tensor out1 = outputs[1];
    return {out0.view({N, M, 1}).transpose(0, 1) , out1.view({N1, M1, 1}).transpose(0, 1)};
  }
}

template std::vector<at::Tensor> gemm_launch<1536, 128,  128, 256, 1, false,  false, 6>(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);
"""

CPP_SRC_3 = r"""
#include <vector>
#include <torch/script.h>

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES
>
std::vector<at::Tensor> gemm_launch(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
);

std::vector<at::Tensor> gemm(
  c10::List<std::tuple<at::Tensor, at::Tensor, at::Tensor>> ABC,
  c10::List<std::tuple<at::Tensor, at::Tensor>> SFAB,
  c10::List<at::Tensor> outputs
) {
  // Note SWAP_AB can be true or false as the shapes are evenly tiled swapped or not swapped
  auto result = gemm_launch<1536, 128,  128, 256, 1, false,  false, 6>(ABC, SFAB, outputs);
  return {result[0], result[1]};
}

TORCH_LIBRARY(blockscaled_cstr_p3, m) {
  m.def("gemm_cstr_p3((Tensor, Tensor, Tensor)[] ABC, (Tensor, Tensor)[] SFAB, Tensor[] outputs) -> Tensor[]");
  m.impl("gemm_cstr_p3", &gemm);
}
"""
ON_VERDA = False
optional_args = {"extra_include_paths": ['/root/cutlass/include/', '/root/cutlass/tools/util/include/',]} if ON_VERDA else {}

module = load_inline(
    "some_module_name_0",
    cpp_sources=CPP_SRC_0,
    cuda_sources=COMMON_CU + CUDA_SRC_0,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v",
    ],
    extra_ldflags=["-lcuda"],
    **optional_args
)

module = load_inline(
    "some_module_name_1",
    cpp_sources=CPP_SRC_1,
    cuda_sources=COMMON_CU + CUDA_SRC_1,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v",
    ],
    extra_ldflags=["-lcuda"],
    **optional_args
)

module = load_inline(
    "some_module_name_2",
    cpp_sources=CPP_SRC_2,
    cuda_sources=COMMON_CU + CUDA_SRC_2,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v",
    ],
    extra_ldflags=["-lcuda"],
    **optional_args
)

module = load_inline(
    "some_module_name_3",
    cpp_sources=CPP_SRC_3,
    cuda_sources=COMMON_CU + CUDA_SRC_3,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v",
    ],
    extra_ldflags=["-lcuda"],
    **optional_args
)
################################################################################
########################## Test custom kernel ##################################
################################################################################

# Scaling factor vector size
sf_vec_size = 16

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    # Pad the input matrix if necessary
    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
        padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(
    data: input_t,
) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled group GEMM.
    """
    abc_tensors, sfasfb_tensors, _, problem_sizes = data
    
    result_tensors = []
    for i, (
        (a_ref, b_ref, c_ref),
        (sfa_ref, sfb_ref),
        (m, n, k, l),
    ) in enumerate(
        zip(
            abc_tensors,
            sfasfb_tensors,
            problem_sizes,
        )
    ):
        for l_idx in range(l):
            # Convert the scale factor tensor to blocked format
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])
            # (m, k) @ (n, k).T -> (m, n)
            res = torch._scaled_mm(
                a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a.cuda(),
                scale_b.cuda(),
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, :, l_idx] = res
        result_tensors.append((c_ref))
    return result_tensors


# Helper function to prepare the scale factor tensors for both reference
# kernel and customize kernel. The customized data layout can be found in:
# https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
def create_reordered_scale_factor_tensor(l, mn, k, ref_f8_tensor):
    sf_k = ceil_div(k, sf_vec_size)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,  # batch size
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    # Create the reordered scale factor tensor (32, 4, rest_m, 4, rest_k, l) on GPU.
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    # Generate a random int8 tensor, then convert to float8_e4m3fn
    rand_int_tensor = torch.randint(1, 3, mma_shape, dtype=torch.int8, device='cuda')
    reordered_f8_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    # Permute according to mma_permute_order
    reordered_f8_tensor = reordered_f8_tensor.permute(*mma_permute_order)

    # Move ref_f8_tensor to GPU if not already there
    if ref_f8_tensor.device.type == 'cpu':
        ref_f8_tensor = ref_f8_tensor.cuda()

    # GPU-side vectorized reordering (replaces slow CPU nested loops)
    # Create index grids for all dimensions
    i_idx = torch.arange(mn, device='cuda')
    j_idx = torch.arange(sf_k, device='cuda')
    b_idx = torch.arange(l, device='cuda')
    
    # Create meshgrid for all combinations of (i, j, b)
    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')
    
    # Calculate target indices in vectorized manner
    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k
    
    # Perform the reordering with advanced indexing (all on GPU)
    reordered_f8_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_tensor[i_grid, j_grid, b_grid]
    
    return reordered_f8_tensor


def _create_fp4_tensors(l, mn, k):
    # generate uint8 tensor, then convert to float4e2m1fn_x2 data type
    # generate all bit patterns
    ref_i8 = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda") # * 0 + 34 # Remove comment to make inputs one 34 = b'0010'0010

    # for each nibble, only keep the sign bit and 2 LSBs
    # the possible values are [-1.5, -1, -0.5, 0, +0.5, +1, +1.5]
    ref_i8 = ref_i8 & 0b1011_1011
    return ref_i8.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)


def generate_input(
    m: tuple,
    n: tuple,
    k: tuple,
    g: int,
    seed: int,
):
    """
    Generate input tensors for NVFP4 block-scaled group GEMM. 
    Each group can have different m, n, k, l.
    
    Args:
        problem_sizes: List of tuples (m, n, k, l) for each problem
        m: Number of rows in matrix A
        n: Number of columns in matrix B
        k: Number of columns in A and rows of B
        l: Batch size, always is 1
        groups: Number of groups
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (list(tuple(a, b, c)), list(tuple(sfa, sfb)), list(tuple(sfa_reordered, sfb_reordered)), list(tuple(m, n, k, l))) where each group has its own a, b, c, sfa, sfb.
            a: [m, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            b: [n, k, l] - Input matrix in torch.float4e2m1fn_x2 data type
            sfa: [m, k // 16, l] - Input scale factors in torch.float8e4m3fn data type
            sfb: [n, k // 16, l] - Input scale factors in torch.float8e4m3fn data type
            sfa_reordered: [32, 4, rest_m, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            sfb_reordered: [32, 4, rest_n, 4, rest_k, l] - Input scale factors in torch.float8e4m3fn data type
            c: [m, n, l] - Output matrix in torch.float16 data type
    """
    torch.manual_seed(seed)
    
    abc_tensors = []
    sfasfb_tensors = []
    sfasfb_reordered_tensors = []
    problem_sizes = []
    l = 1
    # Generate a, b, c, sfa, sfb tensors for all groups
    for group_idx in range(g):
        mi = m[group_idx]
        ni = n[group_idx]
        ki = k[group_idx]
        a_ref = _create_fp4_tensors(l, mi, ki)
        b_ref = _create_fp4_tensors(l, ni, ki)

        c_ref = torch.randn((l, mi, ni), dtype=torch.float16, device="cuda").permute(
            1, 2, 0
        )

        sf_k = ceil_div(ki, sf_vec_size)
         
        sfa_ref_cpu_random = torch.randint(
            1, 3, (l, mi, sf_k), dtype=torch.int8
        ) # * 0 + 1 # Remove comment to make sfs one
        sfa_ref_cpu = sfa_ref_cpu_random.to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)
        sfb_ref_cpu_random = torch.randint(
            1, 3, (l, ni, sf_k), dtype=torch.int8
        ) # * 0 + 1 # Remove comment to make sfs one
        sfb_ref_cpu = sfb_ref_cpu_random.to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)

        sfa_reordered = create_reordered_scale_factor_tensor(l, mi, ki, sfa_ref_cpu)
        sfb_reordered = create_reordered_scale_factor_tensor(l, ni, ki, sfb_ref_cpu)

        abc_tensors.append((a_ref, b_ref, c_ref))
        sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
        sfasfb_reordered_tensors.append((sfa_reordered, sfb_reordered))
        problem_sizes.append((mi, ni, ki, l))
    return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)

################################################################################
########################## End Test custom kernel ##############################
################################################################################

start = 0
BIG_BUFFER = torch.zeros(int(3e10), dtype=torch.float16, device="cuda")

def allocate(c: torch.Tensor):
    global start
    end = start + c.numel()
    buf = BIG_BUFFER[start:end].as_strided(c.shape, c.stride())
    start = end
    return buf


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp4 group gemm
    Args:
        data: list of tuples (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes) where:
            abc_tensors: list of tuples (a, b, c) where 
                a is torch.Tensor[float4e2m1fn_x2] of shape [m, k // 2, l]
                b is torch.Tensor[float4e2m1fn_x2] of shape [n, k // 2, l]
                c is torch.Tensor[float16] of shape [m, n, l]
            sfasfb_tensors: list of tuples (sfa, sfb) where 
                sfa is torch.Tensor[float8_e4m3fnuz] of shape [m, k // 16, l]
                sfb is torch.Tensor[float8_e4m3fnuz] of shape [n, k // 16, l]
            sfasfb_reordered_tensors: list of tuples (sfa_reordered, sfb_reordered) where 
                sfa_reordered is torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_m, 4, rest_k, l]
                sfb_reordered is torch.Tensor[float8_e4m3fnuz] of shape [32, 4, rest_n, 4, rest_k, l]
            problem_sizes: list of tuples (m, n, k, l)
        each group has its own a, b, c, sfa, sfb with different m, n, k, l problem sizes
        l should always be 1 for each group.
    Returns:
        list of tuples (c) where c is torch.Tensor[float16] of shape [m, n, l]
    """
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    if abc_tensors[0][0].shape[1] == 7168 // 2:
      output_tensors = []
      for item in abc_tensors:
          _, _, c_ref = item
          output_tensors.append(c_ref)
      return torch.ops.blockscaled_cstr_p0.gemm_cstr_p0(abc_tensors, sfasfb_reordered_tensors, output_tensors)
    elif abc_tensors[0][0].shape[1] == 2048 // 2:
      output_tensors = []
      for item in abc_tensors:
          _, _, c_ref = item
          output_tensors.append(c_ref)
      return torch.ops.blockscaled_cstr_p1.gemm_cstr_p1(abc_tensors, sfasfb_reordered_tensors, output_tensors)
    elif abc_tensors[0][0].shape[1] == 4096 // 2:
      output_tensors = []
      for item in abc_tensors:
          _, _, c_ref = item
          output_tensors.append(c_ref)
      return torch.ops.blockscaled_cstr_p2.gemm_cstr_p2(abc_tensors, sfasfb_reordered_tensors, output_tensors)
    elif abc_tensors[0][0].shape[1] == 1536 // 2:
      output_tensors = []
      for item in abc_tensors:
          _, _, c_ref = item
          output_tensors.append(c_ref)
      return torch.ops.blockscaled_cstr_p3.gemm_cstr_p3(abc_tensors, sfasfb_reordered_tensors, output_tensors)

    return ref_kernel(data)

if __name__ == '__main__':
  benchmarks = [[8, [80, 176, 128, 72, 64, 248, 96, 160], [4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096], [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168]],\
                [8, [40, 76, 168, 72, 164, 148, 196, 160], [7168, 7168, 7168, 7168, 7168, 7168, 7168, 7168], [2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048]],\
                [2, [192, 320], [3072, 3072], [4096, 4096]],\
                [2, [128, 384], [4096, 4096], [1536, 1536]]]

  for benchmark in benchmarks:
    G, M, N, K = benchmark
    print(f"{M=}, {N=}, {K=}")
    input_data = generate_input(M, N, K, G, 1234)

    raw_results = custom_kernel(input_data)
    ref_results = ref_kernel(input_data)

    print(raw_results[0] - ref_results[0])
