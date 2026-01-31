// @chunk name=device_header
#include <cudaTypedefs.h>
#include <cuda_fp16.h>

#include <torch/library.h>
#include <ATen/core/Tensor.h>

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;  // 32 bytes

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
// @chunk name=kernel_v4
template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int SPLIT_K,
  bool C_N_MAJOR,
  int NUM_STAGES
>
// @kernel name=kernel_v4
__global__
__launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void kernel_v4(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  const char *SFA_ptr,
  const char *SFB_ptr,
  half *C_ptr,
  float *buf_ptr,
  int M, int N
) {
  const int tid = threadIdx.x;
  const int bid_k = blockIdx.x;
  const int bid = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

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
  // @barrier name=tma_mbar scope=cta count=NUM_STAGES
  // @barrier name=mma_mbar scope=cta count=NUM_STAGES
  // @barrier name=mainloop_mbar scope=cta count=1

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);
  // @buffer name=tmem0 space=tmem cols=BLOCK_N*2

  if (warp_id == 0 && elect_sync()) {
    // only 1 thread issue
    // @op mbarrier_init bar=tma_mbar count=1 scope=cta
    // @op mbarrier_init bar=mma_mbar count=1 scope=cta
    // @op mbarrier_init bar=mainloop_mbar count=1 scope=cta
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    // tmem address should be 0, don't bother storing and reading it.
    // @op tcgen05_alloc tmem=tmem0 cols=BLOCK_N*2 cta_group=1 scope=one_warp
    tcgen05_alloc(smem, BLOCK_N * 2);
  }
  __syncthreads();  // visible to all threads

  constexpr int num_iters = K / BLOCK_K / SPLIT_K;

  // warp-specialization
  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
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

      // issue TMA
      const int off_k = SPLIT_K == 1 ? iter_k * BLOCK_K : (iter_k * SPLIT_K + bid_k) * BLOCK_K;
      // @op tma_3d_gmem2smem bar=tma_mbar tmap=A_tmap
      tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      // @op tma_3d_gmem2smem bar=tma_mbar tmap=B_tmap
      tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;
      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;  // 512 = 32x4x4
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      // @op tma_gmem2smem bar=tma_mbar size=SFA_size dst_align=16 src_align=16
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      // @op tma_gmem2smem bar=tma_mbar size=SFB_size dst_align=16 src_align=16
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      // signal TMA done
      // @op mbarrier_arrive_expect_tx bar=tma_mbar size=STAGE_SIZE
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    // issue TMA without waiting for MMA
    for (int iter_k = 0; iter_k < NUM_STAGES; iter_k++)
      issue_tma(iter_k, iter_k);

    // @loop var=iter_k iters=num_iters start=NUM_STAGES
    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      // wait MMA
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      // @op mbarrier_wait bar=mma_mbar phase=mma_phase
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);

      issue_tma(iter_k, stage_id);
    }
    // @endloop
  }
  else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
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

    // @loop var=iter_k iters=num_iters
    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      // wait TMA
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      // @op mbarrier_wait bar=tma_mbar phase=tma_phase
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

      // @loop var=k iters=BLOCK_K/MMA_K
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);  // 4 columns, 512 bytes of 128x4 / 32x4x4
        uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
        // @op tcgen05_cp tmem=tmem0 cta_group=1 issue=one_thread
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        // @op tcgen05_cp tmem=tmem0 cta_group=1 issue=one_thread
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }
      // @endloop

      // k1 selects the (BLOCK_M, 256) tile.
      // k2 selects the (BLOCK_M, 64) tile, whose rows are swizzled.
      // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
      // to have 2-column (8-byte) alignment (looks like not documented).
      // @loop var=k1 iters=BLOCK_K/256
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
        // @loop var=k2 iters=256/MMA_K
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          // @op tcgen05_mma tmem=tmem0 cta_group=1 issue=one_thread
          tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
        }
        // @endloop
      }
      // @endloop

      // signal MMA done
      // @op tcgen05_commit bar=mma_mbar cta_group=1
      tcgen05_commit(mma_mbar_addr + stage_id * 8);
    }
    // @endloop

    // signal mainloop done
    // @op tcgen05_commit bar=mainloop_mbar cta_group=1
    tcgen05_commit(mainloop_mbar_addr);
  }
  else if (tid < BLOCK_M) {
    // epilogue warps

    // wait mainloop
    // @op mbarrier_wait bar=mainloop_mbar phase=0
    mbarrier_wait(mainloop_mbar_addr, 0);
    // @op tcgen05_fence_after_thread_sync
    tcgen05_fence_after_thread_sync();

    auto epilogue_M_major = [&]() {
      // C is M-major
      constexpr int WIDTH = std::min(BLOCK_N, 64);  // using 128 might be slower

      // @loop var=n iters=BLOCK_N/WIDTH
      for (int n = 0; n < BLOCK_N / WIDTH; n++) {
        float tmp[WIDTH];  // if WIDTH=128, we are using 128 registers here
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=WIDTH==128
        if constexpr (WIDTH == 128) tcgen05_ld_32x32bx128(tmp, warp_id * 32, n * WIDTH);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=WIDTH==64
        if constexpr (WIDTH == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=WIDTH==32
        if constexpr (WIDTH == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, n * WIDTH);
        // @op tcgen05_wait_ld
        tcgen05_wait_ld();

        for (int i = 0; i < WIDTH; i++) {
          const int row = off_n + n * WIDTH + i;
          const int col = off_m + tid;

          if constexpr (SPLIT_K == 1)
            C_ptr[row * M + col] = __float2half(tmp[i]);
          else
            atomicAdd(buf_ptr + row * M + col, tmp[i]);
        }
      }
      // @endloop
    };
    auto epilogue_N_major = [&]() {
      // C is N-major
      // @loop var=m iters=32/16
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==128
        if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==64
        if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==32
        if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_wait_ld
        tcgen05_wait_ld();

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;

          if constexpr (SPLIT_K == 1) {
            reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
            reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
          } else {
            atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 0) * N + col), float2({tmp[i * 4 + 0], tmp[i * 4 + 1]}));
            atomicAdd(reinterpret_cast<float2 *>(buf_ptr + (row + 8) * N + col), float2({tmp[i * 4 + 2], tmp[i * 4 + 3]}));
          }
        }
      }
      // @endloop
    };

    if constexpr (C_N_MAJOR)
      epilogue_N_major();
    else
      epilogue_M_major();

    // @op ptx_bar_sync bar_id=1 count=BLOCK_M
    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");  // everyone is done with tmem
    if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
      // @op tcgen05_dealloc tmem=tmem0 cols=BLOCK_N*2 cta_group=1 scope=one_warp
      tcgen05_dealloc(0, BLOCK_N * 2);

  }
}

// @endkernel
// @chunk name=kernel_v3b
template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  bool C_N_MAJOR,
  int NUM_STAGES,
  bool DO_PROFILE
>
// @kernel name=kernel_v3b
__global__
__launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void kernel_v3b(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  const char *SFA_ptr,
  const char *SFB_ptr,
  half *C_ptr,
  int M, int N,
  int64_t *profiler_ptr,
  int num_entries
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  Profiler profiler;
  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.init(num_entries, profiler_ptr, bid * NUM_WARPS + warp_id);
    profiler.start(ProfilerTag::Setup);
  }

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
    // @op mbarrier_init bar=tma_mbar count=1 scope=cta
    // @op mbarrier_init bar=mma_mbar count=1 scope=cta
    // @op mbarrier_init bar=mainloop_mbar count=1 scope=cta
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    // tmem address should be 0, don't bother storing and reading it.
    // @op tcgen05_alloc tmem=tmem0 cols=BLOCK_N*2 cta_group=1 scope=one_warp
    tcgen05_alloc(smem, BLOCK_N * 2);
  }
  __syncthreads();  // visible to all threads
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

  // TODO: make K constexpr as well
  const int num_iters = K / BLOCK_K;

  // warp-specialization
  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
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
      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueTMA);
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      // issue TMA
      const int off_k = iter_k * BLOCK_K;
      // @op tma_3d_gmem2smem bar=tma_mbar tmap=A_tmap
      tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      // @op tma_3d_gmem2smem bar=tma_mbar tmap=B_tmap
      tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;
      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;  // 512 = 32x4x4
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      // @op tma_gmem2smem bar=tma_mbar size=SFA_size dst_align=16 src_align=16
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      // @op tma_gmem2smem bar=tma_mbar size=SFB_size dst_align=16 src_align=16
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      // signal TMA done
      // @op mbarrier_arrive_expect_tx bar=tma_mbar size=STAGE_SIZE
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
      if constexpr (DO_PROFILE) profiler.stop();
    };

    // issue TMA without waiting for MMA
    for (int iter_k = 0; iter_k < NUM_STAGES; iter_k++)
      issue_tma(iter_k, iter_k);

    // @loop var=iter_k iters=num_iters start=NUM_STAGES
    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      // wait MMA
      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitMMA);
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      // @op mbarrier_wait bar=mma_mbar phase=mma_phase
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      if constexpr (DO_PROFILE) profiler.stop();

      issue_tma(iter_k, stage_id);
    }
    // @endloop
  }
  else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    // MMA warp
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    // fp4 MMA doesn't support MMA_M=64. Hence, we will use MMA_M=128 and ignore the rest.
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                              | ((uint32_t)128 >> 7U << 27U)  // MMA_M
                              ;

    // @loop var=iter_k iters=num_iters
    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      // wait TMA
      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitTMA);
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      // @op mbarrier_wait bar=tma_mbar phase=tma_phase
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);
      if constexpr (DO_PROFILE) profiler.stop();

      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueMMA);
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

      // @loop var=k iters=BLOCK_K/MMA_K
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);  // 4 columns, 512 bytes of 128x4 / 32x4x4
        uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
        // @op tcgen05_cp tmem=tmem0 cta_group=1 issue=one_thread
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        // @op tcgen05_cp tmem=tmem0 cta_group=1 issue=one_thread
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }
      // @endloop

      // k1 selects the (BLOCK_M, 256) tile.
      // k2 selects the (BLOCK_M, 64) tile, whose rows are swizzled.
      // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
      // to have 2-column (8-byte) alignment (looks like not documented).
      // @loop var=k1 iters=BLOCK_K/256
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
        // @loop var=k2 iters=256/MMA_K
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          // @op tcgen05_mma tmem=tmem0 cta_group=1 issue=one_thread
          tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
        }
        // @endloop
      }
      // @endloop

      // signal MMA done
      // @op tcgen05_commit bar=mma_mbar cta_group=1
      tcgen05_commit(mma_mbar_addr + stage_id * 8);
      if constexpr (DO_PROFILE) profiler.stop();
    }
    // @endloop

    // signal mainloop done
    // @op tcgen05_commit bar=mainloop_mbar cta_group=1
    tcgen05_commit(mainloop_mbar_addr);
  }
  else if (tid < BLOCK_M) {
    // epilogue warps

    // wait mainloop
    if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::WaitMainloop);
    // @op mbarrier_wait bar=mainloop_mbar phase=0
    mbarrier_wait(mainloop_mbar_addr, 0);
    // @op tcgen05_fence_after_thread_sync
    tcgen05_fence_after_thread_sync();
    if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

    auto epilogue_M_major = [&]() {
      // C is M-major
      constexpr int WIDTH = std::min(BLOCK_N, 64);  // using 128 might be slower

      for (int n = 0; n < BLOCK_N / WIDTH; n++) {
        float tmp[WIDTH];  // if WIDTH=128, we are using 128 registers here
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=WIDTH==128
        if constexpr (WIDTH == 128) tcgen05_ld_32x32bx128(tmp, warp_id * 32, n * WIDTH);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=WIDTH==64
        if constexpr (WIDTH == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * WIDTH);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=WIDTH==32
        if constexpr (WIDTH == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, n * WIDTH);
        // @op tcgen05_wait_ld
        tcgen05_wait_ld();

        for (int i = 0; i < WIDTH; i++)
          C_ptr[(off_n + n * WIDTH + i) * M + (off_m + tid)] = __float2half(tmp[i]);
      }
    };
    auto epilogue_N_major = [&]() {
      // C is N-major
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==128
        if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==64
        if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==32
        if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_wait_ld
        tcgen05_wait_ld();

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;

          reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
          reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
        }
      }
    };

    if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::Epilogue);
    if constexpr (C_N_MAJOR)
      epilogue_N_major();
    else
      epilogue_M_major();

    // @op ptx_bar_sync bar_id=1 count=BLOCK_M
    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");  // everyone is done with tmem
    if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
      // @op tcgen05_dealloc tmem=tmem0 cols=BLOCK_N*2 cta_group=1 scope=one_warp
      tcgen05_dealloc(0, BLOCK_N * 2);

    if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();
  }

  if constexpr (DO_PROFILE) if (elect_sync()) profiler.flush();
}

// @endkernel
