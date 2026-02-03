// experimental device snippet (placeholder)
__device__ __forceinline__ int exp_add(int a, int b) { return a + b; }
__device__ __forceinline__ int exp_sub(int a, int b) { return a - b; }


// Grouped GEMM kernel for NVFP4 block-scaled matmul
// Optimized for small M values with TMA + tcgen05 MMA pipeline

// Problem descriptor for grouped GEMM
struct GroupedProblem {
  int M, N, K;              // problem dimensions
  int64_t A_offset;         // byte offset into packed A buffer
  int64_t B_offset;         // byte offset into packed B buffer
  int64_t SFA_offset;       // byte offset into packed scale factor A buffer
  int64_t SFB_offset;       // byte offset into packed scale factor B buffer
  int64_t C_offset;         // element offset into packed C buffer
  int tiles_m;              // number of tiles in M dimension
  int tiles_n;              // number of tiles in N dimension
  int tile_start;           // global tile index where this group starts
};

// @buffer name=tmem0 space=tmem cols=BLOCK_N*2 dtype=f32
// @buffer name=A_smem space=smem dtype=fp4 major=K swizzle=128B element_bits=4
// @buffer name=B_smem space=smem dtype=fp4 major=K swizzle=128B element_bits=4
// @buffer name=SFA_smem space=smem dtype=fp4 major=K swizzle=none element_bits=4
// @buffer name=SFB_smem space=smem dtype=fp4 major=K swizzle=none element_bits=4

template <
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  bool C_N_MAJOR,
  int NUM_STAGES
>
// @kernel name=grouped_gemm_kernel arch=sm_100a warp_size=WARP_SIZE num_warps=NUM_WARPS cluster_ctas=1
__global__
__launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void grouped_gemm_kernel(
  const char *A_packed,
  const char *B_packed,
  const char *SFA_packed,
  const char *SFB_packed,
  half *C_packed,
  const GroupedProblem *problems,
  int num_groups,
  int total_tiles
) {
  const int tid = threadIdx.x;
  const int global_tile_idx = blockIdx.x;
  
  if (global_tile_idx >= total_tiles) return;
  
  // Binary search to find which group this tile belongs to
  int group_idx = 0;
  for (int g = 0; g < num_groups; g++) {
    if (global_tile_idx >= problems[g].tile_start) {
      group_idx = g;
    }
  }
  
  const GroupedProblem &prob = problems[group_idx];
  const int local_tile_idx = global_tile_idx - prob.tile_start;
  const int bid_m = local_tile_idx / prob.tiles_n;
  const int bid_n = local_tile_idx % prob.tiles_n;
  
  const int M = prob.M;
  const int N = prob.N;
  const int K = prob.K;
  
  const char *A_ptr = A_packed + prob.A_offset;
  const char *B_ptr = B_packed + prob.B_offset;
  const char *SFA_ptr = SFA_packed + prob.SFA_offset;
  const char *SFB_ptr = SFB_packed + prob.SFB_offset;
  half *C_ptr = C_packed + prob.C_offset;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  // @barrier name=tma_mbar scope=cta count=NUM_STAGES
  // @barrier name=mma_mbar scope=cta count=NUM_STAGES
  // @barrier name=mainloop_mbar scope=cta count=1

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);


  // Initialize barriers and allocate tmem
  if (warp_id == 0 && elect_sync()) {
    // @op mbarrier_init bar=tma_mbar count=1 scope=cta
    // @op warp_id=0 lane_id=elect
    // @op mbarrier_init bar=mma_mbar count=1 scope=cta
    // @op warp_id=0 lane_id=elect
    // @op mbarrier_init bar=mainloop_mbar count=1 scope=cta
    // @op warp_id=0 lane_id=elect
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 1) {
    // @op tcgen05_alloc tmem=tmem0 cols=BLOCK_N*2 cta_group=1 scope=one_warp
    // @op warp_id=1
    tcgen05_alloc(smem, BLOCK_N * 2);
  }
  __syncthreads();

  const int num_iters = K / BLOCK_K;
  
  // Warp-specialization
  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    // TMA warp
    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto issue_tma_bulk = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem_base = smem + stage_id * STAGE_SIZE;
      const int B_smem_base = A_smem_base + A_size;
      const int SFA_smem_base = B_smem_base + B_size;
      const int SFB_smem_base = SFA_smem_base + SFA_size;

      const int off_k = iter_k * BLOCK_K;
      
      // Calculate source pointers for A and B
      // A layout: [M, K/2] packed FP4
      const char *A_src = A_ptr + (off_m * (K / 2) + off_k / 2);
      const char *B_src = B_ptr + (off_n * (K / 2) + off_k / 2);
      
      // Issue TMA bulk copies for A and B tiles
      // A tile: BLOCK_M x BLOCK_K (packed as BLOCK_M x BLOCK_K/2 bytes)
      // @op tma_gmem2smem bar=tma_mbar size=A_size dst_align=128 src_align=16
      // @op warp_id=NUM_WARPS-2 lane_id=elect
      tma_gmem2smem(A_smem_base, A_src, A_size, mbar_addr, cache_A);
      // @op tma_gmem2smem bar=tma_mbar size=B_size dst_align=128 src_align=16
      // @op warp_id=NUM_WARPS-2 lane_id=elect
      tma_gmem2smem(B_smem_base, B_src, B_size, mbar_addr, cache_B);

      // Scale factors layout: [M/128, K/64, 32, 4, 4]
      const int rest_k = K / 16 / 4;
      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * 512;
      // @op tma_gmem2smem bar=tma_mbar size=SFA_size dst_align=16 src_align=16
      // @op warp_id=NUM_WARPS-2 lane_id=elect
      tma_gmem2smem(SFA_smem_base, SFA_src, SFA_size, mbar_addr, cache_A);
      // @op tma_gmem2smem bar=tma_mbar size=SFB_size dst_align=16 src_align=16
      // @op warp_id=NUM_WARPS-2 lane_id=elect
      tma_gmem2smem(SFB_smem_base, SFB_src, SFB_size, mbar_addr, cache_B);

      // @op mbarrier_arrive_expect_tx bar=tma_mbar size=STAGE_SIZE
      // @op warp_id=NUM_WARPS-2 lane_id=elect
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");
    };

    // Issue initial TMA loads
    for (int iter_k = 0; iter_k < NUM_STAGES; iter_k++)
      issue_tma_bulk(iter_k, iter_k);

    // @loop var=iter_k iters=num_iters start=NUM_STAGES
    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      // @op mbarrier_wait bar=mma_mbar phase=mma_phase
      // @op warp_id=NUM_WARPS-2 lane_id=elect
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      
      issue_tma_bulk(iter_k, stage_id);
    }
    // @endloop
  }


  else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    // MMA warp
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                              | ((uint32_t)128 >> 7U << 27U)       // MMA_M=128
                              ;

    // @loop var=iter_k iters=num_iters
    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      // @op mbarrier_wait bar=tma_mbar phase=tma_phase
      // @op warp_id=NUM_WARPS-1 lane_id=elect
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      const int A_smem_base = smem + stage_id * STAGE_SIZE;
      const int B_smem_base = A_smem_base + A_size;
      const int SFA_smem_base = B_smem_base + B_size;
      const int SFB_smem_base = SFA_smem_base + SFA_size;

      // @desc name=AB_desc major=K swizzle=128B sbo=8*128 lbo=1
      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      // @desc name=SF_desc major=K swizzle=none sbo=8*16 lbo=1
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      constexpr uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem_base >> 4ULL);
      const uint64_t SFB_desc = SF_desc + ((uint64_t)SFB_smem_base >> 4ULL);

      // @loop var=k iters=BLOCK_K/MMA_K
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);
        uint64_t sfb_desc = SFB_desc + (uint64_t)k * (512ULL >> 4ULL);
        // @op tcgen05_cp tmem=tmem0 cta_group=1 issue=one_thread
        // @op shape=32x128b tile=warpx4 warp_id=NUM_WARPS-1 lane_id=elect desc=SF_desc smem_buf=SFA_smem
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        // @op tcgen05_cp tmem=tmem0 cta_group=1 issue=one_thread
        // @op shape=32x128b tile=warpx4 warp_id=NUM_WARPS-1 lane_id=elect desc=SF_desc smem_buf=SFB_smem
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }
      // @endloop

      // @loop var=k1 iters=BLOCK_K/256
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
        // @loop var=k2 iters=256/MMA_K
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem_base + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem_base + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          // @op tcgen05_mma tmem=tmem0 cta_group=1 issue=one_thread
          // @op shape=mxf4nvf4.block16 warp_id=NUM_WARPS-1 lane_id=elect desc_a=AB_desc desc_b=AB_desc
          tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
        }
        // @endloop
      }
      // @endloop

      // @op tcgen05_commit bar=mma_mbar cta_group=1
      // @op warp_id=NUM_WARPS-1 lane_id=elect
      tcgen05_commit(mma_mbar_addr + stage_id * 8);
    }
    // @endloop

    // @op tcgen05_commit bar=mainloop_mbar cta_group=1
    // @op warp_id=NUM_WARPS-1 lane_id=elect
    tcgen05_commit(mainloop_mbar_addr);
  }


  else if (tid < BLOCK_M) {
    // Epilogue warps
    
    // @op mbarrier_wait bar=mainloop_mbar phase=0
    // @op
    mbarrier_wait(mainloop_mbar_addr, 0);
    // @op tcgen05_fence_after_thread_sync
    // @op
    tcgen05_fence_after_thread_sync();

    auto epilogue_N_major = [&]() {
      // C is N-major (row-major when C has shape [M, N])
      // @loop var=m iters=32/16
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==128
        // @op shape=16x256b num=16 warp_id=warp_id lane_id=lane_id
        if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==64
        // @op shape=16x256b num=8 warp_id=warp_id lane_id=lane_id
        if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_ld tmem=tmem0 cta_group=1 when=BLOCK_N==32
        // @op shape=16x256b num=4 warp_id=warp_id lane_id=lane_id
        if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        // @op tcgen05_wait_ld
        // @op
        tcgen05_wait_ld();

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;
          
          if (row < M && col + 1 < N) {
            reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
            reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
          }
        }
      }
      // @endloop
    };

    if constexpr (C_N_MAJOR)
      epilogue_N_major();

    // @op ptx_bar_sync bar_id=1 count=BLOCK_M
    // @op
    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
    if (warp_id == 0)
      // @op tcgen05_dealloc tmem=tmem0 cols=BLOCK_N*2 cta_group=1 scope=one_warp
      // @op warp_id=0
      tcgen05_dealloc(0, BLOCK_N * 2);
  }
}

// @endkernel
