// @chunk name=host_helpers
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
  uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank-1] = {global_width / 2, 128};  // in bytes
  uint32_t boxDim[rank]          = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank]  = {1, 1, 1};

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
// @chunk name=gemm_launch_v4
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
at::Tensor gemm_launch_v4(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
        at::Tensor& buf
) {
  static_assert(BLOCK_K % 256 == 0);

  const int M = A.size(0);
  const int N = B.size(0);

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr   = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());
  auto buf_ptr = buf.data_ptr<float>();

  int new_M = M;
  int new_N = N;
  if constexpr (SWAP_AB) {
    std::swap(A_ptr, B_ptr);
    std::swap(SFA_ptr, SFB_ptr);
    std::swap(new_M, new_N);
  }

  CUtensorMap A_tmap, B_tmap;
  // @op cute_tmap name=A_tmap rank=3
  // @op
  init_AB_tmap(&A_tmap, A_ptr, new_M, K, BLOCK_M, BLOCK_K);
  // @op cute_tmap name=B_tmap rank=3
  // @op
  init_AB_tmap(&B_tmap, B_ptr, new_N, K, BLOCK_N, BLOCK_K);

  dim3 grid(SPLIT_K, (new_M / BLOCK_M) * (new_N / BLOCK_N));
  int tb_size = BLOCK_M + 2 * WARP_SIZE;
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = kernel_v4<K, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, C_N_MAJOR != SWAP_AB, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, tb_size, smem_size>>>(A_tmap, B_tmap, SFA_ptr, SFB_ptr, C_ptr, buf_ptr, new_M, new_N);

  if constexpr (SPLIT_K == 1)
    return C_N_MAJOR ? C : C.view({N, M, 1}).transpose(0, 1);
  else
    return C_N_MAJOR ? buf : buf.view({N, M, 1}).transpose(0, 1);
}
// @chunk name=gemm_v4
at::Tensor gemm_v4(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
        at::Tensor& buf
) {
  const int K = A.size(1) * 2;

#define LAUNCH(K_, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, SWAP_AB, C_N_MAJOR, NUM_STAGES) \
  else if (K == K_) C = gemm_launch_v4<K_, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, SWAP_AB, C_N_MAJOR, NUM_STAGES>(A, B, SFA, SFB, C, buf);

  if (false) {}
  LAUNCH(16384, 128, 128, 256, 2, true,  true, 6)
  LAUNCH( 7168, 128,  64, 256, 2, true,  true, 8)
  LAUNCH( 2048, 128,  64, 256, 1, true, false, 8)
  // the rest
  LAUNCH( 256, 128, 64, 256, 1, true, false, 6)
  LAUNCH( 512, 128, 64, 256, 1, true, false, 6)
  LAUNCH(1536, 128, 64, 256, 1, true, false, 6)
  LAUNCH(2304, 128, 64, 256, 1, true, false, 6)

#undef LAUNCH

  return C;
}
// @chunk name=register_v4
TORCH_LIBRARY(my_module_v4, m) {
  m.def("gemm(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, Tensor(b!) buf) -> Tensor");
  m.impl("gemm", &gemm_v4);
}
// @chunk name=gemm_launch_v3b
template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  bool SWAP_AB,
  bool C_N_MAJOR,
  int NUM_STAGES,
  bool DO_PROFILE
>
at::Tensor gemm_launch_v3b(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
  int64_t *profiler_ptr,
  int num_entries
) {
  static_assert(BLOCK_K % 256 == 0);

  const int M = A.size(0);
  const int N = B.size(0);

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr   = reinterpret_cast<const char *>(B.data_ptr());
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
  // @op cute_tmap name=A_tmap rank=3
  // @op
  init_AB_tmap(&A_tmap, A_ptr, new_M, K, BLOCK_M, BLOCK_K);
  // @op cute_tmap name=B_tmap rank=3
  // @op
  init_AB_tmap(&B_tmap, B_ptr, new_N, K, BLOCK_N, BLOCK_K);

  int grid = (new_M / BLOCK_M) * (new_N / BLOCK_N);
  int tb_size = BLOCK_M + 2 * WARP_SIZE;
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = kernel_v3b<K, BLOCK_M, BLOCK_N, BLOCK_K, C_N_MAJOR != SWAP_AB, NUM_STAGES, DO_PROFILE>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, tb_size, smem_size>>>(A_tmap, B_tmap, SFA_ptr, SFB_ptr, C_ptr, new_M, new_N, profiler_ptr, num_entries);

  return C_N_MAJOR ? C : C.view({N, M, 1}).transpose(0, 1);
}
// @chunk name=gemm_v3b
at::Tensor gemm_v3b(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C
) {
  const int K = A.size(1) * 2;
  constexpr bool DO_PROFILE = false;
  int64_t *profiler_ptr = nullptr;
  int num_entries = 0;

#define LAUNCH(K_, BLOCK_M, BLOCK_N, BLOCK_K, SWAP_AB, C_N_MAJOR, NUM_STAGES) \
  else if (K == K_) C = gemm_launch_v3b<K_, BLOCK_M, BLOCK_N, BLOCK_K, SWAP_AB, C_N_MAJOR, NUM_STAGES, DO_PROFILE>(A, B, SFA, SFB, C, profiler_ptr, num_entries);

  if (false) {}
  LAUNCH(16384, 128, 64, 256, true, false, 8)
  LAUNCH( 7168,  64, 64, 512, true, false, 5)
  LAUNCH( 2048, 128, 64, 256, true, false, 8)
  // the rest
  LAUNCH( 256, 128, 64, 256, true, false, 6)
  LAUNCH( 512, 128, 64, 256, true, false, 6)
  LAUNCH(1536, 128, 64, 256, true, false, 6)
  LAUNCH(2304, 128, 64, 256, true, false, 6)

  return C;
}
// @chunk name=register_v3b
TORCH_LIBRARY(my_module_v3b, m) {
  m.def("gemm(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C) -> Tensor");
  m.impl("gemm", &gemm_v3b);
}
