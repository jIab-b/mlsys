#!POPCORN leaderboard nvfp4_group_gemm

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object cuda_nvfp4_group_gemm(
  py::handle abc_tensors,
  py::handle sfasfb_tensors,
  py::handle problem_sizes
);
"""

CUDA_HEADERS = [
    r"""
    #include <cstdio>
    #include <cuda.h>

    ////////////////////////////////////////////////////////////////////////////////
    // HELPER FUNCTION TO CHECK FOR ERRORS
    ////////////////////////////////////////////////////////////////////////////////
    void cuda_check(CUresult code, const char *file, int line) {
      if (code != CUDA_SUCCESS) {
        char const *str;
        cuGetErrorString(code, &str);
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, str);
        exit(1);
      }
    }

    void cuda_check(cudaError_t code, const char *file, int line) {
      if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%n: %s\n", file, line,
                cudaGetErrorString(code));
        exit(1);
      }
    }

    // Macro for convenient CUDA error checking
    #define CUDA_CHECK(x)                                                          \
      do {                                                                         \
        cuda_check((x), __FILE__, __LINE__);                                       \
      } while (0)

    ////////////////////////////////////////////////////////////////////////////////
    // ASYNC PROXY FENCE
    ////////////////////////////////////////////////////////////////////////////////

    __device__ static __forceinline__ void async_proxy_fence() {
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // MBARRIER FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////

    __device__ static __forceinline__ void init_barrier(uint64_t *bar,
                                                        int arrival_count) {
      uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
                  "r"(arrival_count)
                  : "memory");
    }

    __device__ static __forceinline__ void fence_barrier_init() {
      asm volatile("fence.mbarrier_init.release.cluster;\n");
    }

    __device__ static __forceinline__ void arrive(uint64_t *bar, uint32_t count) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0],  %1;\n"
                  :
                  : "r"(mbar_ptr), "r"(count)
                  : "memory");
    }

    __device__ static __forceinline__ void wait(uint64_t *bar, int phaseParity) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile("{\n"
                  ".reg .pred                P1;\n"
                  "LAB_WAIT:\n"
                  "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                  "@P1                       bra.uni DONE;\n"
                  "bra.uni                   LAB_WAIT;\n"
                  "DONE:\n"
                  "}\n" ::"r"(mbar_ptr),
                  "r"(phaseParity));
    }

    __device__ static __forceinline__ void wait_relaxed(uint64_t *bar,
                                                        int phaseParity) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "{\n"
          ".reg .pred                P1;\n"
          "LAB_WAIT:\n"
          "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 P1, [%0], %1;\n"
          "@P1                       bra.uni DONE;\n"
          "bra.uni                   LAB_WAIT;\n"
          "DONE:\n"
          "}\n" ::"r"(mbar_ptr),
          "r"(phaseParity));
    }

    __device__ static __forceinline__ void expect_bytes_and_arrive(uint64_t *bar,
                                                                  uint32_t bytes) {
      uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm("mbarrier.arrive.expect_tx.release.cta.shared.b64 _, [%0], %1;\n "
          :
          : "r"(bar_ptr), "r"(bytes)
          : "memory");
    }

    __device__ static __forceinline__ void
    expect_bytes_and_arrive_cluster(uint64_t *bar, uint32_t bytes) {
      uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], "
          "%1;\n "
          :
          : "r"(bar_ptr), "r"(bytes)
          : "memory");
    }

    __device__ static __forceinline__ void
    expect_bytes_and_arrive_cluster_raw(uint32_t bar_ptr, uint32_t bytes) {
      asm("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], "
          "%1;\n "
          :
          : "r"(bar_ptr), "r"(bytes)
          : "memory");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // TMA GROUP OPERATIONS
    ////////////////////////////////////////////////////////////////////////////////

    __device__ static __forceinline__ void tma_commit_group() {
      asm volatile("cp.async.bulk.commit_group;");
    }

    template <int N>
    __device__ static __forceinline__ void tma_wait_until_pending() {
      asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
    }

    __device__ static __forceinline__ void
    prefetch_tensormap(const CUtensorMap *tensor_map) {
      asm volatile("prefetch.tensormap [%0];\n" ::"l"(tensor_map) : "memory");
    }

    __device__ static __forceinline__ void
    tensormap_acquire(const CUtensorMap *tensor_map) {
      asm volatile("fence.proxy.tensormap::generic.acquire.cta [%0], 128;\n" ::"l"(
                      tensor_map)
                  : "memory");
    }

    __device__ static __forceinline__ void tensormap_release() {
      asm volatile("fence.proxy.tensormap::generic.release.cta;\n");
    }

    __device__ static __forceinline__ void
    tensormap_cp_and_fence(CUtensorMap *dst, const CUtensorMap *src) {
      uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
      asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic."
                  "release.cta.sync.aligned [%0], [%1], 128;\n" ::"l"(dst),
                  "r"(smem_ptr));
    }

    __device__ static __forceinline__ void
    tensormap_replace_global_address(CUtensorMap *tensor_map,
                                    const void *new_addr) {
      uint32_t tensormap_ptr =
          static_cast<uint32_t>(__cvta_generic_to_shared(tensor_map));
      asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 "
                  "[%0], %1;\n" ::"r"(tensormap_ptr),
                  "l"(new_addr));
    }

    template <int Ord>
    __device__ static __forceinline__ void
    tensormap_replace_global_dim(CUtensorMap *tensor_map, uint32_t new_dim) {
      uint32_t tensormap_ptr =
          static_cast<uint32_t>(__cvta_generic_to_shared(tensor_map));
      asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], "
                  "%2, %1;\n" ::"r"(tensormap_ptr),
                  "r"(new_dim), "n"(Ord));
    }

    template <int Ord>
    __device__ static __forceinline__ void
    tensormap_replace_global_stride(CUtensorMap *tensor_map, uint64_t new_stride) {
      uint32_t tensormap_ptr =
          static_cast<uint32_t>(__cvta_generic_to_shared(tensor_map));
      asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 "
                  "[%0], %2, %1;\n" ::"r"(tensormap_ptr),
                  "l"(new_stride), "n"(Ord));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // TMA CACHE POLICIES
    ////////////////////////////////////////////////////////////////////////////////

    enum class CachePolicy : uint64_t {
      EVICT_NORMAL = 0x1000000000000000,
      EVICT_FIRST = 0x12F0000000000000,
      EVICT_LAST = 0x14F0000000000000,
    };

    ////////////////////////////////////////////////////////////////////////////////
    // GLOBAL -> SHARED
    ////////////////////////////////////////////////////////////////////////////////

    __device__ static __forceinline__ void
    cp_async_bulk_global_to_shared(void *smem_dest, const void *src, int size,
                                  uint64_t *bar, CachePolicy cache_policy) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes."
          "L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(src), "r"(size), "r"(mbar_ptr), "l"(cache_policy)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
        void *smem_dest, const CUtensorMap *tensor_map, int c0, uint64_t *bar,
        CachePolicy cache_policy) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%5.L2::cache_hint "
          "[%0], [%1, {%2}], [%3], %4;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(mbar_ptr), "l"(cache_policy),
            "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void
    cp_async_bulk_tensor_1d_global_to_shared_raw(void *smem_dest,
                                                const CUtensorMap *tensor_map,
                                                int c0, uint32_t mbar_ptr,
                                                CachePolicy cache_policy) {
      asm volatile(
          "cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%5.L2::cache_hint "
          "[%0], [%1, {%2}], [%3], %4;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(mbar_ptr), "l"(cache_policy),
            "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
        void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1,
        uint64_t *bar, CachePolicy cache_policy) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%6.L2::cache_hint "
          "[%0], [%1, {%2, %3}], [%4], %5;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(c1), "r"(mbar_ptr), "l"(cache_policy),
            "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void
    cp_async_bulk_tensor_2d_global_to_shared_raw(void *smem_dest,
                                                const CUtensorMap *tensor_map,
                                                int c0, int c1, uint32_t mbar_ptr,
                                                CachePolicy cache_policy) {
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%6.L2::cache_hint "
          "[%0], [%1, {%2, %3}], [%4], %5;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(c1), "r"(mbar_ptr), "l"(cache_policy),
            "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void cp_async_bulk_tensor_3d_global_to_shared(
        void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
        uint64_t *bar, CachePolicy cache_policy) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%7.L2::cache_hint "
          "[%0], [%1, {%2, %3, %4}], [%5], %6;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_ptr),
            "l"(cache_policy), "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void
    cp_async_bulk_tensor_3d_global_to_shared_raw(void *smem_dest,
                                                const CUtensorMap *tensor_map,
                                                int c0, int c1, int c2,
                                                uint32_t mbar_ptr,
                                                CachePolicy cache_policy) {
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%7.L2::cache_hint "
          "[%0], [%1, {%2, %3, %4}], [%5], %6;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_ptr),
            "l"(cache_policy), "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
        void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
        int c3, uint64_t *bar, CachePolicy cache_policy) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%8.L2::cache_hint "
          "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_ptr),
            "l"(cache_policy), "n"(CTA_GROUP)
          : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ static __forceinline__ void cp_async_bulk_tensor_5d_global_to_shared(
        void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
        int c3, int c4, uint64_t *bar, CachePolicy cache_policy) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile(
          "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_"
          "tx::bytes.cta_group::%9.L2::cache_hint "
          "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
          :
          : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
            "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
            "r"(mbar_ptr), "l"(cache_policy), "n"(CTA_GROUP)
          : "memory");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // SHARED -> GLOBAL
    ////////////////////////////////////////////////////////////////////////////////

    __device__ static __forceinline__ void
    cp_async_bulk_tensor_1d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                            const void *src,
                                            CachePolicy cache_policy) {
      asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.L2::"
                  "cache_hint "
                  "[%0, {%1}], [%2], %3;\n"
                  :
                  : "l"(tensor_map), "r"(c0),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))),
                    "l"(cache_policy)
                  : "memory");
    }

    __device__ static __forceinline__ void
    cp_async_bulk_tensor_2d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                            int c1, const void *src,
                                            CachePolicy cache_policy) {
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::"
                  "cache_hint "
                  "[%0, {%1, %2}], [%3], %4;\n"
                  :
                  : "l"(tensor_map), "r"(c0), "r"(c1),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))),
                    "l"(cache_policy)
                  : "memory");
    }

    __device__ static __forceinline__ void
    cp_async_bulk_tensor_3d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                            int c1, int c2, const void *src,
                                            CachePolicy cache_policy) {
      asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::"
                  "cache_hint "
                  "[%0, {%1, %2, %3}], [%4], %5;\n"
                  :
                  : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))),
                    "l"(cache_policy)
                  : "memory");
    }

    __device__ static __forceinline__ void cp_async_bulk_tensor_4d_shared_to_global(
        const CUtensorMap *tensor_map, int c0, int c1, int c2, int c3,
        const void *src, CachePolicy cache_policy) {
      asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.L2::"
                  "cache_hint "
                  "[%0, {%1, %2, %3, %4}], [%5], %6;\n"
                  :
                  : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))),
                    "l"(cache_policy)
                  : "memory");
    }

    __device__ static __forceinline__ void cp_async_bulk_tensor_5d_shared_to_global(
        const CUtensorMap *tensor_map, int c0, int c1, int c2, int c3, int c4,
        const void *src, CachePolicy cache_policy) {
      asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.L2::"
                  "cache_hint "
                  "[%0, {%1, %2, %3, %4, %5}], [%6], %7;\n"
                  :
                  : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
                    "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))),
                    "l"(cache_policy)
                  : "memory");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PREFETCH
    ////////////////////////////////////////////////////////////////////////////////

    __device__ static __forceinline__ void
    cp_async_bulk_prefetch(const void *src, int size, CachePolicy cache_policy) {
      asm volatile(
          "cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(
              src),
          "r"(size), "l"(cache_policy)
          : "memory");
    }
    """,
    r"""
    #include <cuda_fp16.h>

    ////////////////////////////////////////////////////////////////////////////////
    // WARP GROUP REGISTER ALLOCATION
    ////////////////////////////////////////////////////////////////////////////////

    template <uint32_t RegCount> __device__ void warpgroup_reg_alloc() {
      asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
    }

    template <uint32_t RegCount> __device__ void warpgroup_reg_dealloc() {
      asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // tcgen05 COMMIT GROUP FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////

    template <int CTA_GROUP = 1> __device__ void tcgen05_commit(uint64_t *bar) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile("tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::"
                  "cluster.b64 [%0];\n"
                  :
                  : "r"(mbar_ptr), "n"(CTA_GROUP)
                  : "memory");
    }

    template <int CTA_GROUP = 1>
    __device__ void tcgen05_commit_mcast(uint64_t *bar, uint16_t mask) {
      uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
      asm volatile("tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::"
                  "cluster.multicast::cluster.b64 [%0], %1;\n"
                  :
                  : "r"(mbar_ptr), "h"(mask), "n"(CTA_GROUP)
                  : "memory");
    }

    __device__ void tcgen05_fence_before() {
      asm volatile("tcgen05.fence::before_thread_sync;\n");
    }

    __device__ void tcgen05_fence_after() {
      asm volatile("tcgen05.fence::after_thread_sync;\n");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // TMEM FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////

    template <int CTA_GROUP = 1>
    __device__ void tcgen05_alloc(uint32_t *dst, uint32_t cols) {
      uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
      asm volatile(
          "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;\n"
          :
          : "r"(smem_ptr), "r"(cols), "n"(CTA_GROUP));
    }

    template <int CTA_GROUP = 1>
    __device__ void tcgen05_dealloc(uint32_t tmem, uint32_t cols) {
      asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;\n"
                  :
                  : "r"(tmem), "r"(cols), "n"(CTA_GROUP));
    }

    struct SHAPE {
      static constexpr char _32x32b[] = ".32x32b";
      static constexpr char _16x64b[] = ".16x64b";
      static constexpr char _16x128b[] = ".16x128b";
      static constexpr char _16x256b[] = ".16x256b";
    };

    template <int NUM_REGS, const char *SHAPE, int NUM>
    __device__ void tcgen05_ld(uint32_t tmem, float *d) {
      if constexpr (NUM_REGS == 1)
        asm volatile("tcgen05.ld.sync.aligned%3.x%4.b32 {%0}, [%1];"
                    : "=f"(d[0])
                    : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 2)
        asm volatile("tcgen05.ld.sync.aligned%3.x%4.b32 {%0, %1}, [%2];"
                    : "=f"(d[0]), "=f"(d[1])
                    : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 4)
        asm volatile("tcgen05.ld.sync.aligned%5.x%6.b32 "
                    "{%0, %1, %2, %3}, [%4];"
                    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                    : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 8)
        asm volatile("tcgen05.ld.sync.aligned%9.x%10.b32 "
                    "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7}, [%8];"
                    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
                      "=f"(d[5]), "=f"(d[6]), "=f"(d[7])
                    : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 16)
        asm volatile("tcgen05.ld.sync.aligned%17.x%18.b32 "
                    "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
                      "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
                      "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
                      "=f"(d[14]), "=f"(d[15])
                    : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 32)
        asm volatile("tcgen05.ld.sync.aligned%33.x%34.b32 "
                    "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                    "  %8,  %9, %10, %11, %12, %13, %14, %15, "
                    " %16, %17, %18, %19, %20, %21, %22, %23, "
                    " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
                      "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
                      "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
                      "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]),
                      "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]),
                      "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
                      "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
                      "=f"(d[30]), "=f"(d[31])
                    : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 64)
        asm volatile(
            "tcgen05.ld.sync.aligned%65.x%66.b32 "
            "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
            "  %8,  %9, %10, %11, %12, %13, %14, %15, "
            " %16, %17, %18, %19, %20, %21, %22, %23, "
            " %24, %25, %26, %27, %28, %29, %30, %31, "
            " %32, %33, %34, %35, %36, %37, %38, %39, "
            " %40, %41, %42, %43, %44, %45, %46, %47, "
            " %48, %49, %50, %51, %52, %53, %54, %55, "
            " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
              "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
              "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]),
              "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
              "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]),
              "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
              "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]),
              "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
              "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]),
              "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]),
              "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]),
              "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
              "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
            : "r"(tmem), "C"(SHAPE), "n"(NUM));
      if constexpr (NUM_REGS == 128)
        asm volatile(
            "tcgen05.ld.sync.aligned%129.x%130.b32 "
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
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
              "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
              "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]),
              "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]),
              "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]),
              "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
              "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]),
              "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]),
              "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]),
              "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]),
              "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]),
              "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]),
              "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]),
              "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]),
              "=f"(d[70]), "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]),
              "=f"(d[75]), "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]),
              "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]),
              "=f"(d[85]), "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]),
              "=f"(d[90]), "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]),
              "=f"(d[95]), "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]),
              "=f"(d[100]), "=f"(d[101]), "=f"(d[102]), "=f"(d[103]), "=f"(d[104]),
              "=f"(d[105]), "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]),
              "=f"(d[110]), "=f"(d[111]), "=f"(d[112]), "=f"(d[113]), "=f"(d[114]),
              "=f"(d[115]), "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]),
              "=f"(d[120]), "=f"(d[121]), "=f"(d[122]), "=f"(d[123]), "=f"(d[124]),
              "=f"(d[125]), "=f"(d[126]), "=f"(d[127])
            : "r"(tmem), "C"(SHAPE), "n"(NUM));
    }

    template <int NUM> __device__ void tcgen05_ld_32x32b(uint32_t tmem, float *d) {
      tcgen05_ld<NUM, SHAPE::_32x32b, NUM>(tmem, d);
    }

    template <int NUM> __device__ void tcgen05_ld_16x64b(uint32_t tmem, float *d) {
      tcgen05_ld<NUM, SHAPE::_16x64b, NUM>(tmem, d);
    }

    template <int NUM> __device__ void tcgen05_ld_16x128b(uint32_t tmem, float *d) {
      tcgen05_ld<NUM * 2, SHAPE::_16x128b, NUM>(tmem, d);
    }

    template <int NUM> __device__ void tcgen05_ld_16x256b(uint32_t tmem, float *d) {
      tcgen05_ld<NUM * 4, SHAPE::_16x256b, NUM>(tmem, d);
    }

    __device__ void tcgen05_wait_ld() {
      asm volatile("tcgen05.wait::ld.sync.aligned;");
    }

    __device__ void tcgen05_wait_st() {
      asm volatile("tcgen05.wait::st.sync.aligned;");
    }

    template <int CTA_GROUP = 1>
    __device__ void tcgen05_cp_32x128b_x4(uint64_t desc, uint32_t tmem) {
      asm volatile("tcgen05.cp.cta_group::%2.32x128b.warpx4 [%0], %1;\n"
                  :
                  : "r"(tmem), "l"(desc), "n"(CTA_GROUP));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // SHARED MEMORY DESCRIPTORS
    ////////////////////////////////////////////////////////////////////////////////

    enum wgmmaSwizzle {
      NO_SWIZZLE,
      SWIZZLE_128B_ATOM_32B,
      SWIZZLE_128B,
      SWIZZLE_64B,
      SWIZZLE_32B,
    };

    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
      return (x & 0x3FFFFULL) >> 0x4;
    }

    template <wgmmaSwizzle Swizzle>
    __device__ uint64_t make_smem_desc(char *ptr, uint64_t lbo, uint64_t sbo) {
      uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
      uint64_t desc = 0x0000000000000000;
      desc |= matrix_descriptor_encode(addr);
      desc |= matrix_descriptor_encode(lbo) << 16;
      desc |= matrix_descriptor_encode(sbo) << 32;
      desc |= 1llu << 46;

      uint64_t swizzle_val;
      if constexpr (Swizzle == NO_SWIZZLE) {
        swizzle_val = 0llu;
      } else if constexpr (Swizzle == SWIZZLE_128B_ATOM_32B) {
        swizzle_val = 1llu;
      } else if constexpr (Swizzle == SWIZZLE_128B) {
        swizzle_val = 2llu;
      } else if constexpr (Swizzle == SWIZZLE_64B) {
        swizzle_val = 4llu;
      } else if constexpr (Swizzle == SWIZZLE_32B) {
        swizzle_val = 6llu;
      } else {
        static_assert(true, "Invalid wgmmaSwizzle value");
      }

      desc |= swizzle_val << 61;
      return desc;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // INSTRUCTION DESCRIPTORS
    ////////////////////////////////////////////////////////////////////////////////

    template <int M, int N, int K, int NegateA, int NegateB>
    __device__ constexpr uint32_t make_inst_desc() {
      uint32_t desc = 0x00000000;
      desc |= 1lu << 7;
      desc |= 1lu << 10;
      desc |= uint32_t(NegateA) << 13;
      desc |= uint32_t(NegateB) << 14;
      desc |= ((N >> 3) & 0x3F) << 17;
      desc |= ((M >> 7) & 0x03) << 27;

      if constexpr (K == 64 || K == 128) {
        desc |= 0lu << 31;
      } else if constexpr (K == 96) {
        desc |= 1lu << 31;
      } else {
        static_assert(true, "Invalid K value");
      }

      return desc;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // tcgen05 Intrinsic Calls
    ////////////////////////////////////////////////////////////////////////////////

    struct CollectorUsage {
      static constexpr char FILL[] = "fill";
      static constexpr char USE[] = "use";
      static constexpr char LASTUSE[] = "lastuse";
      static constexpr char DISCARD[] = "discard";
    };

    template <int CTA_GROUP = 1, int EnableD, uint32_t InstDesc>
    __device__ void tcgen05_mma(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_d,
                                uint32_t tmem_sfa, uint32_t tmem_sfb) {
      asm volatile(
          "{\n"
          "tcgen05.mma.cta_group::%7.kind::mxf4nvf4.block_scale.scale_vec::"
          "4X [%0], %1, %2, %3, [%4], [%5], %6;\n"
          "}\n"
          :
          : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(InstDesc), "r"(tmem_sfa),
            "r"(tmem_sfb), "n"(int32_t(EnableD)), "n"(CTA_GROUP));
    }
    """,
]

CUDA_SRC = r"""
#define WARP_SIZE 32
#define WARPGROUP_SIZE 128
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define ALIGN_UP(x, a) (((x) + ((a)-1)) & ~((a)-1))

#define UMMA_K 64
#define SF_VEC_SIZE 16

#define TMA_WARP 4
#define MMA_WARP 5

struct Group {
  int M, N, K;
  char *A_ptr;
  char *B_ptr;
  char *SFA_ptr;
  char *SFB_ptr;
  half *C_ptr;

  // kernel parameters
  bool swap;
  bool transpose;
};

template <int32_t G, int32_t BLOCK_M, int32_t BLOCK_N, int32_t NUM_SMS>
struct Scheduler {
  const uint32_t *grouped_layout;
  int sm_id;

  int current_iter = 0;
  int current_group_idx = 0;
  uint32_t last_psum, current_psum;
  uint32_t problem_m, problem_n, problem_k;
  bool problem_swap, problem_transpose;

  __device__ __forceinline__ explicit Scheduler(const uint32_t *grouped_layout)
      : grouped_layout(grouped_layout), sm_id(blockIdx.x) {
    last_psum = 0;
    problem_m = grouped_layout[G + 0];
    problem_n = grouped_layout[G + 1];
    problem_k = grouped_layout[G + 2];

    problem_swap = grouped_layout[0] & (1u << 31);
    problem_transpose = grouped_layout[0] & (1u << 30);
    current_psum = grouped_layout[0] & ((1u << 30) - 1);
  }

  __device__ __forceinline__ bool next(int32_t &m_block_idx,
                                       int32_t &n_block_idx) {
    const int32_t next_block_idx = current_iter * NUM_SMS + sm_id;

    while (true) {
      if (next_block_idx < current_psum)
        break;

      if (++current_group_idx == G)
        return false;

      last_psum = current_psum;
      problem_m = grouped_layout[G + 3 * current_group_idx + 0];
      problem_n = grouped_layout[G + 3 * current_group_idx + 1];
      problem_k = grouped_layout[G + 3 * current_group_idx + 2];
      problem_swap = grouped_layout[current_group_idx] & (1u << 31);
      problem_transpose = grouped_layout[current_group_idx] & (1u << 30);
      current_psum = grouped_layout[current_group_idx] & ((1u << 30) - 1);
    }

    const int32_t num_n_blocks = CEIL_DIV(problem_n, BLOCK_N);
    m_block_idx = (next_block_idx - last_psum) / num_n_blocks;
    n_block_idx = (next_block_idx - last_psum) % num_n_blocks;

    current_iter++;
    return true;
  }
};

__device__ __forceinline__ void sync_wg(int wg_id) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(wg_id + 1) : "memory");
}

__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t"
               "}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

template <typename T> __device__ __forceinline__ T make_warp_uniform(T x) {
  return __shfl_sync(0xFFFF'FFFF, x, 0);
}

__device__ __forceinline__ float reinterpret_as_float(half2 h) {
  return *reinterpret_cast<float *>(&h);
}

__device__ __forceinline__ void st_v8(float *addr, float d[8]) {
  asm volatile(
      "st.global.wt.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n" ::"l"(
          addr),
      "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "f"(d[4]), "f"(d[5]),
      "f"(d[6]), "f"(d[7]));
}

template <int32_t SHAPE_K, int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t PIPE_DEPTH, int32_t PIPE_DEPTH_EPI, int32_t NUM_SMS>
__global__ void __launch_bounds__(WARPGROUP_SIZE + 2 * WARP_SIZE, 1)
    nvfp4_group_gemm_g8_cutlass(
        __grid_constant__ const std::array<int32_t, 8> psums,
        __grid_constant__ const std::array<int32_t, 8> M_arr,
        __grid_constant__ const std::array<int32_t, 8> N_arr,
        __grid_constant__ const std::array<CUtensorMap, 8> A_maps,
        __grid_constant__ const std::array<CUtensorMap, 8> B_maps,
        __grid_constant__ const std::array<const char *__restrict__, 8>
            SFA_ptrs,
        __grid_constant__ const std::array<const char *__restrict__, 8>
            SFB_ptrs,
        __grid_constant__ const std::array<half *__restrict__, 8> C_ptrs,
        bool is_swap, bool is_transpose) {
  extern __shared__ __align__(1024) char shmem[];
  __shared__ __align__(8) uint64_t bars[2 * PIPE_DEPTH + 2 * PIPE_DEPTH_EPI];
  __shared__ uint32_t tmem_base[1];

  uint64_t *full_bar = bars;
  uint64_t *empty_bar = bars + PIPE_DEPTH;
  uint64_t *tmem_full_bar = bars + 2 * PIPE_DEPTH;
  uint64_t *tmem_empty_bar = bars + 2 * PIPE_DEPTH + PIPE_DEPTH_EPI;

  constexpr int32_t TILE_A_BYTES = BLOCK_K * BLOCK_M / 2;
  constexpr int32_t TILE_B_BYTES = BLOCK_K * BLOCK_N / 2;
  constexpr int32_t TILE_SFA_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t TILE_SFB_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t STAGE_BYTES =
      TILE_A_BYTES + TILE_B_BYTES + TILE_SFA_BYTES + TILE_SFB_BYTES;

  char *pipe_start = shmem;

  constexpr int32_t K = SHAPE_K;
  constexpr int32_t TMEM_WIDTH_SFA = 16;
  constexpr int32_t TMEM_WIDTH_SFB = 16;

  const int32_t cta_id = blockIdx.x;
  const int32_t warp_id = make_warp_uniform(threadIdx.x / WARP_SIZE);

  int32_t block_idx = cta_id;
  int32_t group_idx = 0;

  int32_t M, N;
  int32_t m_block_idx, n_block_idx;
  int32_t last_psum = 0;

  auto next_block = [&]() -> bool {
    while (true) {
      int32_t current_psum = psums[group_idx];
      if (block_idx < current_psum)
        break;
      last_psum = current_psum;
      if (++group_idx == 8)
        return false;
    }

    M = M_arr[group_idx];
    N = N_arr[group_idx];

    const int32_t num_m_blocks = CEIL_DIV(M, BLOCK_M);
    const int32_t num_n_blocks = CEIL_DIV(N, BLOCK_N);
    m_block_idx = (block_idx - last_psum) / num_n_blocks;
    n_block_idx = (block_idx - last_psum) % num_n_blocks;

    block_idx += NUM_SMS;
    return true;
  };

  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < PIPE_DEPTH; i++) {
      init_barrier(&full_bar[i], 1);
      init_barrier(&empty_bar[i], 1);
    }
    for (int32_t i = 0; i < PIPE_DEPTH_EPI; i++) {
      init_barrier(&tmem_full_bar[i], 1);
      init_barrier(&tmem_empty_bar[i], WARPGROUP_SIZE);
    }
    fence_barrier_init();
  }
  __syncthreads();

  int32_t pipe_idx = 0;
  int32_t phase = 0;
  auto advance_pipeline = [&]() {
    pipe_idx = (pipe_idx == PIPE_DEPTH - 1) ? 0 : pipe_idx + 1;
    phase ^= (pipe_idx == 0);
  };

  int32_t epi_pipe_idx = 0;
  int32_t epi_phase = 0;
  auto advance_epi_pipeline = [&]() {
    epi_pipe_idx = (epi_pipe_idx == PIPE_DEPTH_EPI - 1) ? 0 : epi_pipe_idx + 1;
    epi_phase ^= (epi_pipe_idx == 0);
  };

  if (warp_id == TMA_WARP) {
    if (elect_one_sync()) {
      while (next_block()) {
        const CUtensorMap *a_map = &A_maps[group_idx];
        const CUtensorMap *b_map = &B_maps[group_idx];
        const char *SFA_ptr = SFA_ptrs[group_idx];
        const char *SFB_ptr = SFB_ptrs[group_idx];

        CachePolicy cache_policy_a = CachePolicy::EVICT_FIRST;
        CachePolicy cache_policy_b = CachePolicy::EVICT_FIRST;

#pragma unroll 1
        for (int32_t k = 0; k < (K / BLOCK_K); k++) {
          wait_relaxed(&empty_bar[pipe_idx], phase ^ 1);
          expect_bytes_and_arrive(&full_bar[pipe_idx], STAGE_BYTES);

          char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
          char *b_shr = a_shr + TILE_A_BYTES;
          cp_async_bulk_tensor_2d_global_to_shared(
              a_shr, a_map, k * BLOCK_K, m_block_idx * BLOCK_M,
              &full_bar[pipe_idx], cache_policy_a);
          cp_async_bulk_tensor_2d_global_to_shared(
              b_shr, b_map, k * BLOCK_K, n_block_idx * BLOCK_N,
              &full_bar[pipe_idx], cache_policy_b);

          char *sfa_shr = b_shr + TILE_B_BYTES;
          char *sfb_shr = sfa_shr + TILE_SFA_BYTES;
          cp_async_bulk_global_to_shared(
              sfa_shr,
              SFA_ptr +
                  ((m_block_idx * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) +
                   k * 4) *
                      512,
              TILE_SFA_BYTES, &full_bar[pipe_idx], cache_policy_a);
          cp_async_bulk_global_to_shared(
              sfb_shr,
              SFB_ptr +
                  ((n_block_idx * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                   k * 4) *
                      512,
              TILE_SFB_BYTES, &full_bar[pipe_idx], cache_policy_b);

          advance_pipeline();
        }
      }
    }
  } else if (warp_id == MMA_WARP) {
    tcgen05_alloc(tmem_base, 512);

    constexpr uint32_t tmem_sfa = PIPE_DEPTH_EPI * BLOCK_N;
    constexpr uint32_t tmem_sfb = tmem_sfa + TMEM_WIDTH_SFA;

    constexpr uint32_t inst_desc =
        make_inst_desc<BLOCK_M, BLOCK_N, UMMA_K, 0, 0>();

    if (elect_one_sync()) {
      while (next_block()) {
        const uint32_t tmem_d = epi_pipe_idx * BLOCK_N;
        const uint32_t sfa_offset = (BLOCK_M == 64) ? (m_block_idx % 2) * 2 : 0;
        const uint32_t sfb_offset = (BLOCK_N == 64) ? (n_block_idx % 2) * 2 : 0;

        wait(&tmem_empty_bar[epi_pipe_idx], epi_phase ^ 1);
        for (int32_t k = 0; k < (K / BLOCK_K); k++) {
          char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
          char *sfa_shr = a_shr + TILE_A_BYTES + TILE_B_BYTES;

          const uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_shr, 1, 1024);
          const uint64_t desc_b = desc_a + (TILE_A_BYTES >> 4);
          const uint64_t desc_sfa =
              make_smem_desc<NO_SWIZZLE>(sfa_shr, 128, 128);
          const uint64_t desc_sfb = desc_sfa + (TILE_SFA_BYTES >> 4);

          wait_relaxed(&full_bar[pipe_idx], phase);

#pragma unroll
          for (int32_t j = 0; j < 4; j++) {
            tcgen05_cp_32x128b_x4(desc_sfa + j * 32, tmem_sfa + j * 4);
            tcgen05_cp_32x128b_x4(desc_sfb + j * 32, tmem_sfb + j * 4);
          }
#pragma unroll
          for (int32_t j = 0; j < 4; j++) {
            if (j == 0 && k == 0) {
              tcgen05_mma<1, 0, inst_desc>(
                  desc_a + j * 2, desc_b + j * 2, tmem_d,
                  tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                  tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            } else {
              tcgen05_mma<1, 1, inst_desc>(
                  desc_a + j * 2, desc_b + j * 2, tmem_d,
                  tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                  tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            }
          }
          tcgen05_commit(&empty_bar[pipe_idx]);
          advance_pipeline();
        }
        tcgen05_commit(&tmem_full_bar[epi_pipe_idx]);
        advance_epi_pipeline();
      }
    }
  } else {
    warpgroup_reg_alloc<256>();

    while (next_block()) {
      half *C_ptr = C_ptrs[group_idx];

      constexpr int32_t UNROLL_FACTOR = 8;
      float acc[BLOCK_N / UNROLL_FACTOR];

      uint32_t tmem_d = epi_pipe_idx * BLOCK_N;
      wait_relaxed(&tmem_full_bar[epi_pipe_idx], epi_phase);
      tcgen05_fence_after();

      auto epilogue_n_16x256 = [&]() {
        const int32_t lane_id = threadIdx.x % WARP_SIZE;
        half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

        if (is_swap) {
          for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
            for (int32_t r1 = 0; r1 < 2; r1++) {
              tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
                  tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                      r1 * (16 << 16),
                  acc);
              tcgen05_wait_ld();

              for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
                const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
                const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                    (lane_id % 4) * 2 + 2 * i;

                if (n_block_idx * BLOCK_N + col < N) {
                  half2 vec1 =
                      __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
                  __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]),
                         vec1);
                  half2 vec2 =
                      __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
                  __stwt(reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                         vec2);
                }
              }
            }
          }
        } else {
          if (M % 16 == 0) {
            for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
              for (int32_t r1 = 0; r1 < 2; r1++) {
                tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
                    tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                        r1 * (16 << 16),
                    acc);
                tcgen05_wait_ld();

                for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
                  const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
                  const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                      (lane_id % 4) * 2 + 2 * i;

                  if (m_block_idx * BLOCK_M + row < M) {
                    half2 vec1 =
                        __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
                    __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]),
                           vec1);
                    half2 vec2 =
                        __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
                    __stwt(
                        reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                        vec2);
                  }
                }
              }
            }
          } else {
            for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
              for (int32_t r1 = 0; r1 < 2; r1++) {
                tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
                    tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                        r1 * (16 << 16),
                    acc);
                tcgen05_wait_ld();

                for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
                  const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
                  const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                      (lane_id % 4) * 2 + 2 * i;

                  if (m_block_idx * BLOCK_M + row < M) {
                    half2 vec1 =
                        __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
                    __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]),
                           vec1);
                  }
                  if (m_block_idx * BLOCK_M + row + 8 < M) {
                    half2 vec2 =
                        __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
                    __stwt(
                        reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                        vec2);
                  }
                }
              }
            }
          }
        }
      };

      auto epilogue_n_32x32 = [&]() {
        half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

        if (is_swap) {
          if (N % 16 == 0) {
            for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
              tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
                  tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
              tcgen05_wait_ld();

              for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 16) {
                const int32_t row = threadIdx.x;
                const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;

                if (n_block_idx * BLOCK_N + col < N) {
                  float acc_packed[8];
                  for (int32_t j = 0; j < 8; j++) {
                    acc_packed[j] = reinterpret_as_float(__float22half2_rn(
                        make_float2(acc[i + j * 2], acc[i + j * 2 + 1])));
                  }
                  st_v8(reinterpret_cast<float *>(&C_off[row * N + col]),
                        acc_packed);
                }
              }
            }
          } else { // fallback to 16x256
            const int32_t lane_id = threadIdx.x % WARP_SIZE;
            for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
              for (int32_t r1 = 0; r1 < 2; r1++) {
                tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
                    tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                        r1 * (16 << 16),
                    acc);
                tcgen05_wait_ld();

                for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
                  const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
                  const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                      (lane_id % 4) * 2 + 2 * i;

                  if (n_block_idx * BLOCK_N + col < N) {
                    half2 vec1 =
                        __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
                    __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]),
                           vec1);
                    half2 vec2 =
                        __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
                    __stwt(
                        reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                        vec2);
                  }
                }
              }
            }
          }
        } else {
          if (M % 16 == 0) {
            for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
              tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
                  tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
              tcgen05_wait_ld();

              for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 16) {
                const int32_t row = threadIdx.x;
                const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;

                if (m_block_idx * BLOCK_M + row < M) {
                  float acc_packed[8];
                  for (int32_t j = 0; j < 8; j++) {
                    acc_packed[j] = reinterpret_as_float(__float22half2_rn(
                        make_float2(acc[i + j * 2], acc[i + j * 2 + 1])));
                  }
                  st_v8(reinterpret_cast<float *>(&C_off[row * N + col]),
                        acc_packed);
                }
              }
            }
          } else { // fallback to 16x256
            const int32_t lane_id = threadIdx.x % WARP_SIZE;
            for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
              for (int32_t r1 = 0; r1 < 2; r1++) {
                tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
                    tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                        r1 * (16 << 16),
                    acc);
                tcgen05_wait_ld();

                for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
                  const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
                  const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                      (lane_id % 4) * 2 + 2 * i;

                  if (m_block_idx * BLOCK_M + row < M) {
                    half2 vec1 =
                        __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
                    __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]),
                           vec1);
                  }
                  if (m_block_idx * BLOCK_M + row + 8 < M) {
                    half2 vec2 =
                        __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
                    __stwt(
                        reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                        vec2);
                  }
                }
              }
            }
          }
        }
      };

      auto epilogue_t_32x32 = [&]() {
        half *C_off = C_ptr + n_block_idx * BLOCK_N * M + m_block_idx * BLOCK_M;

        if (is_swap) {
          for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
            tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
                tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
            tcgen05_wait_ld();

            for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i++) {
              const int32_t row = threadIdx.x;
              const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;

              if (n_block_idx * BLOCK_N + col < N) {
                __stwt(&C_off[col * M + row], __float2half_rn(acc[i]));
              }
            }
          }
        } else {
          for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
            tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
                tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
            tcgen05_wait_ld();

            for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i++) {
              const int32_t row = threadIdx.x;
              const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;

              if (m_block_idx * BLOCK_M + row < M) {
                __stwt(&C_off[col * M + row], __float2half_rn(acc[i]));
              }
            }
          }
        }
      };

      is_transpose ? epilogue_t_32x32() : epilogue_n_32x32();
      arrive(&tmem_empty_bar[epi_pipe_idx], 1);
      advance_epi_pipeline();
    }
    sync_wg(0);
    if (warp_id == 0) {
      tcgen05_dealloc(0, 512);
    }
  }
}

template <int32_t SHAPE_N, int32_t SHAPE_K, int32_t BLOCK_M, int32_t BLOCK_N,
          int32_t BLOCK_K, int32_t PIPE_DEPTH, bool SWAPPED, bool TRANSPOSE,
          bool SKIP_CHECK>
__global__ void __launch_bounds__(WARPGROUP_SIZE + 2 * WARP_SIZE, 1)
    nvfp4_group_gemm_g2_cutlass(int psum, int M1, int M2, int N1, int N2,
                                __grid_constant__ const CUtensorMap a1_map,
                                __grid_constant__ const CUtensorMap a2_map,
                                __grid_constant__ const CUtensorMap b1_map,
                                __grid_constant__ const CUtensorMap b2_map,
                                const char *__restrict__ SFA1_ptr,
                                const char *__restrict__ SFA2_ptr,
                                const char *__restrict__ SFB1_ptr,
                                const char *__restrict__ SFB2_ptr,
                                half *__restrict__ C1_ptr,
                                half *__restrict__ C2_ptr) {
  extern __shared__ __align__(1024) char shmem[];
  __shared__ __align__(8) uint64_t bars[2 * PIPE_DEPTH + 1];
  __shared__ uint32_t tmem_base[1];

  uint64_t *full_bar = bars;
  uint64_t *empty_bar = bars + PIPE_DEPTH;
  uint64_t *mainloop_bar = bars + 2 * PIPE_DEPTH;

  constexpr int32_t TILE_A_BYTES = BLOCK_K * BLOCK_M / 2;
  constexpr int32_t TILE_B_BYTES = BLOCK_K * BLOCK_N / 2;
  constexpr int32_t TILE_SFA_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t TILE_SFB_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t STAGE_BYTES =
      TILE_A_BYTES + TILE_B_BYTES + TILE_SFA_BYTES + TILE_SFB_BYTES;

  char *pipe_start = shmem;

  constexpr int32_t K = SHAPE_K;
  constexpr int32_t TMEM_WIDTH_SFA = 16;
  constexpr int32_t TMEM_WIDTH_SFB = 16;

  const int32_t cta_id = blockIdx.x;
  const int32_t warp_id = make_warp_uniform(threadIdx.x / WARP_SIZE);

  if constexpr (SWAPPED) {
    M1 = SHAPE_N;
    M2 = SHAPE_N;
  } else {
    N1 = SHAPE_N;
    N2 = SHAPE_N;
  }

  const bool is_first_group = cta_id < psum;

  const CUtensorMap *a_map = is_first_group ? &a1_map : &a2_map;
  const CUtensorMap *b_map = is_first_group ? &b1_map : &b2_map;
  const char *SFA_ptr = is_first_group ? SFA1_ptr : SFA2_ptr;
  const char *SFB_ptr = is_first_group ? SFB1_ptr : SFB2_ptr;
  half *C_ptr = is_first_group ? C1_ptr : C2_ptr;

  const int32_t num_rows = CEIL_DIV(is_first_group ? M1 : M2, BLOCK_M);
  const int32_t num_cols = CEIL_DIV(is_first_group ? N1 : N2, BLOCK_N);
  const int32_t bid = is_first_group ? cta_id : cta_id - psum;

  const int32_t m_block_idx = bid / num_cols;
  const int32_t n_block_idx = bid % num_cols;

  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < 2 * PIPE_DEPTH + 1; i++) {
      init_barrier(&bars[i], 1);
    }
    fence_barrier_init();
  }
  __syncthreads();

  if (warp_id == TMA_WARP) {
    if (elect_one_sync()) {
      CachePolicy cache_policy_a = CachePolicy::EVICT_FIRST;
      CachePolicy cache_policy_b = CachePolicy::EVICT_FIRST;

      auto issue_tma = [&](int32_t k, int32_t pipe_idx) {
        expect_bytes_and_arrive(&full_bar[pipe_idx], STAGE_BYTES);

        char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
        char *b_shr = a_shr + TILE_A_BYTES;
        if constexpr (SHAPE_K == 4096) {
          cp_async_bulk_tensor_2d_global_to_shared(
              a_shr, a_map, k * BLOCK_K, m_block_idx * BLOCK_M,
              &full_bar[pipe_idx], cache_policy_a);
          cp_async_bulk_tensor_2d_global_to_shared(
              b_shr, b_map, k * BLOCK_K, n_block_idx * BLOCK_N,
              &full_bar[pipe_idx], cache_policy_b);
        } else {
          cp_async_bulk_tensor_2d_global_to_shared(
              b_shr, b_map, k * BLOCK_K, n_block_idx * BLOCK_N,
              &full_bar[pipe_idx], cache_policy_b);
          cp_async_bulk_tensor_2d_global_to_shared(
              a_shr, a_map, k * BLOCK_K, m_block_idx * BLOCK_M,
              &full_bar[pipe_idx], cache_policy_a);
        }

        char *sfa_shr = b_shr + TILE_B_BYTES;
        char *sfb_shr = sfa_shr + TILE_SFA_BYTES;
        if constexpr (SHAPE_K == 4096) {
          cp_async_bulk_global_to_shared(
              sfa_shr,
              SFA_ptr + ((m_block_idx * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                            512,
              TILE_SFA_BYTES, &full_bar[pipe_idx], cache_policy_a);
          cp_async_bulk_global_to_shared(
              sfb_shr,
              SFB_ptr + ((n_block_idx * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                            512,
              TILE_SFB_BYTES, &full_bar[pipe_idx], cache_policy_b);
        } else {
          cp_async_bulk_global_to_shared(
              sfb_shr,
              SFB_ptr + ((n_block_idx * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                            512,
              TILE_SFB_BYTES, &full_bar[pipe_idx], cache_policy_b);
          cp_async_bulk_global_to_shared(
              sfa_shr,
              SFA_ptr + ((m_block_idx * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                            512,
              TILE_SFA_BYTES, &full_bar[pipe_idx], cache_policy_a);
        }
      };

      for (int32_t k = 0; k < PIPE_DEPTH; k++) {
        issue_tma(k, k);
      }
      for (int32_t k = PIPE_DEPTH; k < K / BLOCK_K; k++) {
        const int32_t pipe_idx = k % PIPE_DEPTH;
        const int32_t phase = (k / PIPE_DEPTH) % 2;
        wait_relaxed(&empty_bar[pipe_idx], phase ^ 1);
        issue_tma(k, pipe_idx);
      }
    }
  } else if (warp_id == MMA_WARP) {
    tcgen05_alloc(tmem_base, 512);

    constexpr uint32_t tmem_d = 0;
    constexpr uint32_t tmem_sfa = BLOCK_N;
    constexpr uint32_t tmem_sfb = tmem_sfa + TMEM_WIDTH_SFA;

    constexpr uint32_t inst_desc =
        make_inst_desc<BLOCK_M, BLOCK_N, UMMA_K, 0, 0>();

    if (elect_one_sync()) {
      const uint32_t sfa_offset = (BLOCK_M == 64) ? (m_block_idx % 2) * 2 : 0;
      const uint32_t sfb_offset = (BLOCK_N == 64) ? (n_block_idx % 2) * 2 : 0;

      for (int32_t k = 0; k < K / BLOCK_K; k++) {
        const int32_t pipe_idx = k % PIPE_DEPTH;
        const int32_t phase = (k / PIPE_DEPTH) % 2;

        char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
        char *sfa_shr = a_shr + TILE_A_BYTES + TILE_B_BYTES;

        const uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_shr, 1, 1024);
        const uint64_t desc_b = desc_a + (TILE_A_BYTES >> 4);
        const uint64_t desc_sfa = make_smem_desc<NO_SWIZZLE>(sfa_shr, 128, 128);
        const uint64_t desc_sfb = desc_sfa + (TILE_SFA_BYTES >> 4);

        wait_relaxed(&full_bar[pipe_idx], phase);

#pragma unroll
        for (int32_t j = 0; j < 4; j++) {
          tcgen05_cp_32x128b_x4(desc_sfa + j * 32, tmem_sfa + j * 4);
          tcgen05_cp_32x128b_x4(desc_sfb + j * 32, tmem_sfb + j * 4);
        }
#pragma unroll
        for (int32_t j = 0; j < 4; j++) {
          if (j == 0 && k == 0) {
            tcgen05_mma<1, 0, inst_desc>(
                desc_a + j * 2, desc_b + j * 2, tmem_d,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          } else {
            tcgen05_mma<1, 1, inst_desc>(
                desc_a + j * 2, desc_b + j * 2, tmem_d,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          }
        }
        if (k < (K / BLOCK_K) - PIPE_DEPTH) {
          tcgen05_commit(&empty_bar[pipe_idx]);
        }
      }
      tcgen05_commit(&mainloop_bar[0]);
    }
  } else {
    warpgroup_reg_alloc<256>();

    constexpr int32_t UNROLL_FACTOR = 8;
    float acc[BLOCK_N / UNROLL_FACTOR];
    uint32_t tmem_d = 0;

    const int32_t M = is_first_group ? M1 : M2;
    const int32_t N = is_first_group ? N1 : N2;

    wait_relaxed(&mainloop_bar[0], 0);
    tcgen05_fence_after();

    auto epilogue_n_16x256 = [&]() {
      const int32_t lane_id = threadIdx.x % WARP_SIZE;
      half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

      for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
        for (int32_t r1 = 0; r1 < 2; r1++) {
          tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
              tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) + r1 * (16 << 16),
              acc);
          tcgen05_wait_ld();

          for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
            const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
            const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                (lane_id % 4) * 2 + 2 * i;
            const bool is_valid =
                SKIP_CHECK || (SWAPPED ? (n_block_idx * BLOCK_N + col < N)
                                       : (m_block_idx * BLOCK_M + row < M));

            if (is_valid) {
              half2 vec1 =
                  __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
              __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]), vec1);
              half2 vec2 =
                  __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
              __stwt(reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                     vec2);
            }
          }
        }
      }
    };

    auto epilogue_n_32x32 = [&]() {
      half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

      for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
        tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
            tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
        tcgen05_wait_ld();

        for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 16) {
          const int32_t row = threadIdx.x;
          const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;
          const bool is_valid =
              SKIP_CHECK || (SWAPPED ? (n_block_idx * BLOCK_N + col < N)
                                     : (m_block_idx * BLOCK_M + row < M));

          if (is_valid) {
            float acc_packed[8];
            for (int32_t j = 0; j < 8; j++) {
              acc_packed[j] = reinterpret_as_float(__float22half2_rn(
                  make_float2(acc[i + j * 2], acc[i + j * 2 + 1])));
            }
            st_v8(reinterpret_cast<float *>(&C_off[row * N + col]), acc_packed);
          }
        }
      }
    };

    auto epilogue_t_32x32 = [&]() {
      half *C_off = C_ptr + n_block_idx * BLOCK_N * M + m_block_idx * BLOCK_M;

      for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
        tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
            tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
        tcgen05_wait_ld();

        for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i++) {
          const int32_t row = threadIdx.x;
          const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;
          const bool is_valid =
              SKIP_CHECK || (SWAPPED ? (n_block_idx * BLOCK_N + col < N)
                                     : (m_block_idx * BLOCK_M + row < M));

          if (is_valid) {
            __stwt(&C_off[col * M + row], __float2half_rn(acc[i]));
          }
        }
      }
    };

    TRANSPOSE ? epilogue_t_32x32() : epilogue_n_32x32();
    sync_wg(0);
    if (warp_id == 0) {
      tcgen05_dealloc(0, 512);
    }
  }
}

template <int32_t SHAPE_N, int32_t SHAPE_K, int32_t BLOCK_M, int32_t BLOCK_N,
          int32_t BLOCK_K, int32_t PIPE_DEPTH, bool SWAPPED, bool TRANSPOSE,
          bool SKIP_CHECK>
__global__ void __cluster_dims__(2, 1, 1)
    __launch_bounds__(WARPGROUP_SIZE + 2 * WARP_SIZE, 1)
        nvfp4_group_gemm_g2_2sm_cutlass(
        int psum, int M1, int M2, int N1, int N2,
        __grid_constant__ const CUtensorMap a1_map,
        __grid_constant__ const CUtensorMap a2_map,
        __grid_constant__ const CUtensorMap b1_map,
        __grid_constant__ const CUtensorMap b2_map,
        __grid_constant__ const CUtensorMap sfa1_map,
        __grid_constant__ const CUtensorMap sfa2_map,
        __grid_constant__ const CUtensorMap sfb1_map,
        __grid_constant__ const CUtensorMap sfb2_map, half *__restrict__ C1_ptr,
        half *__restrict__ C2_ptr) {
  extern __shared__ __align__(1024) char shmem[];
  __shared__ __align__(8) uint64_t bars[2 * PIPE_DEPTH + 1];
  __shared__ uint32_t tmem_base[1];

  uint64_t *full_bar = bars;
  uint64_t *empty_bar = bars + PIPE_DEPTH;
  uint64_t *mainloop_bar = bars + 2 * PIPE_DEPTH;

  constexpr int32_t TILE_A_BYTES = BLOCK_K * BLOCK_M / 2;
  constexpr int32_t TILE_B_BYTES = BLOCK_K * (BLOCK_N / 2) / 2;
  constexpr int32_t TILE_SFA_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t TILE_SFB_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t STAGE_BYTES =
      TILE_A_BYTES + TILE_B_BYTES + TILE_SFA_BYTES + TILE_SFB_BYTES;

  char *pipe_start = shmem;

  constexpr int32_t K = SHAPE_K;
  constexpr int32_t TMEM_WIDTH_SFA = 16;
  constexpr int32_t TMEM_WIDTH_SFB = 16;

  const int32_t cta_id = blockIdx.x;
  const int32_t cta_rank = cta_id % 2;
  const int32_t warp_id = make_warp_uniform(threadIdx.x / WARP_SIZE);

  constexpr uint16_t cta_mask = 3;

  if constexpr (SWAPPED) {
    M1 = SHAPE_N;
    M2 = SHAPE_N;
  } else {
    N1 = SHAPE_N;
    N2 = SHAPE_N;
  }

  const bool is_first_group = cta_id < psum;

  const CUtensorMap *a_map = is_first_group ? &a1_map : &a2_map;
  const CUtensorMap *b_map = is_first_group ? &b1_map : &b2_map;
  const CUtensorMap *sfa_map = is_first_group ? &sfa1_map : &sfa2_map;
  const CUtensorMap *sfb_map = is_first_group ? &sfb1_map : &sfb2_map;
  half *C_ptr = is_first_group ? C1_ptr : C2_ptr;

  const int32_t num_rows = CEIL_DIV(is_first_group ? M1 : M2, BLOCK_M);
  const int32_t num_cols = CEIL_DIV(is_first_group ? N1 : N2, BLOCK_N);
  const int32_t bid = is_first_group ? cta_id : cta_id - psum;

  const int32_t m_block_idx = bid / (num_cols * 2) * 2 + (bid % 2);
  const int32_t n_block_idx = (bid / 2) % num_cols;

  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < PIPE_DEPTH; i++) {
      init_barrier(&full_bar[i], 2);
      init_barrier(&empty_bar[i], 1);
    }
    init_barrier(&mainloop_bar[0], 1);
    fence_barrier_init();
  }

  asm volatile("barrier.cluster.arrive.relaxed.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");

  if (warp_id == TMA_WARP) {
    if (elect_one_sync()) {
      CachePolicy cache_policy_a = CachePolicy::EVICT_FIRST;
      CachePolicy cache_policy_b = CachePolicy::EVICT_FIRST;

      auto issue_tma = [&](int32_t k, int32_t pipe_idx) {
        const uint32_t mbar_addr =
            static_cast<uint32_t>(
                __cvta_generic_to_shared(&full_bar[pipe_idx])) &
            0xFEFFFFFF;
        expect_bytes_and_arrive_cluster_raw(mbar_addr, STAGE_BYTES);

        char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
        char *b_shr = a_shr + TILE_A_BYTES;
        cp_async_bulk_tensor_3d_global_to_shared_raw<2>(
            b_shr, b_map, 0, n_block_idx * BLOCK_N + cta_rank * (BLOCK_N / 2),
            k, mbar_addr, cache_policy_b);
        cp_async_bulk_tensor_3d_global_to_shared_raw<2>(
            a_shr, a_map, 0, m_block_idx * BLOCK_M, k, mbar_addr,
            cache_policy_a);
      
        char *sfa_shr = b_shr + TILE_B_BYTES;
        char *sfb_shr = sfa_shr + TILE_SFA_BYTES;
        cp_async_bulk_tensor_1d_global_to_shared_raw<2>(
            sfb_shr, sfb_map,
            ((n_block_idx * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) + k * 4) *
                512 / 8,
            mbar_addr, cache_policy_b);
        cp_async_bulk_tensor_1d_global_to_shared_raw<2>(
            sfa_shr, sfa_map,
            ((m_block_idx * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) + k * 4) *
                512 / 8,
            mbar_addr, cache_policy_a);
      };

      for (int32_t k = 0; k < PIPE_DEPTH; k++) {
        issue_tma(k, k);
      }
      for (int32_t k = PIPE_DEPTH; k < K / BLOCK_K; k++) {
        const int32_t pipe_idx = k % PIPE_DEPTH;
        const int32_t phase = (k / PIPE_DEPTH) % 2;
        wait(&empty_bar[pipe_idx], phase ^ 1);
        issue_tma(k, pipe_idx);
      }
    }
  } else if (warp_id == MMA_WARP) {
    tcgen05_alloc<2>(tmem_base, 512);

    constexpr uint32_t tmem_d = 0;
    constexpr uint32_t tmem_sfa = BLOCK_N;
    constexpr uint32_t tmem_sfb = tmem_sfa + TMEM_WIDTH_SFA;

    constexpr uint32_t inst_desc =
        make_inst_desc<2 * BLOCK_M, BLOCK_N, UMMA_K, 0, 0>();

    if (cta_rank == 0 && elect_one_sync()) {
      const uint32_t sfa_offset = (BLOCK_M == 64) ? (m_block_idx % 2) * 2 : 0;
      const uint32_t sfb_offset = (BLOCK_N == 64) ? (n_block_idx % 2) * 2 : 0;

      for (int32_t k = 0; k < K / BLOCK_K; k++) {
        const int32_t pipe_idx = k % PIPE_DEPTH;
        const int32_t phase = (k / PIPE_DEPTH) % 2;

        char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
        char *sfa_shr = a_shr + TILE_A_BYTES + TILE_B_BYTES;

        const uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_shr, 1, 1024);
        const uint64_t desc_b = desc_a + (TILE_A_BYTES >> 4);
        const uint64_t desc_sfa = make_smem_desc<NO_SWIZZLE>(sfa_shr, 128, 128);
        const uint64_t desc_sfb = desc_sfa + (TILE_SFA_BYTES >> 4);

        wait(&full_bar[pipe_idx], phase);

#pragma unroll
        for (int32_t j = 0; j < 4; j++) {
          tcgen05_cp_32x128b_x4<2>(desc_sfa + j * 32, tmem_sfa + j * 4);
          tcgen05_cp_32x128b_x4<2>(desc_sfb + j * 32, tmem_sfb + j * 4);
        }
#pragma unroll
        for (int32_t j = 0; j < 4; j++) {
          if (j == 0 && k == 0) {
            tcgen05_mma<2, 0, inst_desc>(
                desc_a + j * 2, desc_b + j * 2, tmem_d,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          } else {
            tcgen05_mma<2, 1, inst_desc>(
                desc_a + j * 2, desc_b + j * 2, tmem_d,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          }
        }
        if (k < (K / BLOCK_K) - PIPE_DEPTH) {
          tcgen05_commit_mcast<2>(&empty_bar[pipe_idx], cta_mask);
        }
      }
      tcgen05_commit_mcast<2>(&mainloop_bar[0], cta_mask);
    }
  } else {
    warpgroup_reg_alloc<256>();

    constexpr int32_t UNROLL_FACTOR = 8;
    float acc[BLOCK_N / UNROLL_FACTOR];
    uint32_t tmem_d = 0;

    const int32_t M = is_first_group ? M1 : M2;
    const int32_t N = is_first_group ? N1 : N2;

    wait(&mainloop_bar[0], 0);
    tcgen05_fence_after();

    auto epilogue_n_16x256 = [&]() {
      const int32_t lane_id = threadIdx.x % WARP_SIZE;
      half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

      for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
        for (int32_t r1 = 0; r1 < 2; r1++) {
          tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
              tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) + r1 * (16 << 16) + cta_rank * (BLOCK_M << 16),
              acc);
          tcgen05_wait_ld();

          for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
            const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
            const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                (lane_id % 4) * 2 + 2 * i;
            const bool is_valid =
                SKIP_CHECK || (SWAPPED ? (n_block_idx * BLOCK_N + col < N)
                                       : (m_block_idx * BLOCK_M + row < M));

            if (is_valid) {
              half2 vec1 =
                  __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
              __stwt(reinterpret_cast<half2 *>(&C_off[row * N + col]), vec1);
              half2 vec2 =
                  __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
              __stwt(reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col]),
                     vec2);
            }
          }
        }
      }
    };

    auto epilogue_n_32x32 = [&]() {
      half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

      for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
        tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
            tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
        tcgen05_wait_ld();

        for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 16) {
          const int32_t row = threadIdx.x;
          const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;
          const bool is_valid =
              SKIP_CHECK || (SWAPPED ? (n_block_idx * BLOCK_N + col < N)
                                     : (m_block_idx * BLOCK_M + row < M));

          if (is_valid) {
            float acc_packed[8];
            for (int32_t j = 0; j < 8; j++) {
              acc_packed[j] = reinterpret_as_float(__float22half2_rn(
                  make_float2(acc[i + j * 2], acc[i + j * 2 + 1])));
            }
            st_v8(reinterpret_cast<float *>(&C_off[row * N + col]), acc_packed);
          }
        }
      }
    };

    auto epilogue_t_32x32 = [&]() {
      half *C_off = C_ptr + n_block_idx * BLOCK_N * M + m_block_idx * BLOCK_M;

      for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
        tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
            tmem_d + r * (BLOCK_N / UNROLL_FACTOR) + cta_rank * (BLOCK_M << 16), acc);
        tcgen05_wait_ld();

        for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i++) {
          const int32_t row = threadIdx.x;
          const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;
          const bool is_valid =
              SKIP_CHECK || (SWAPPED ? (n_block_idx * BLOCK_N + col < N)
                                     : (m_block_idx * BLOCK_M + row < M));

          if (is_valid) {
            __stwt(&C_off[col * M + row], __float2half_rn(acc[i]));
          }
        }
      }
    };

    TRANSPOSE ? epilogue_t_32x32() : epilogue_n_32x32();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    if (warp_id == 0) {
      tcgen05_dealloc<2>(0, 512);
    }
  }
}

template <int32_t G, int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t PIPE_DEPTH, int32_t PIPE_DEPTH_EPI, int32_t NUM_SMS>
__global__ void __launch_bounds__(WARPGROUP_SIZE + 2 * WARP_SIZE, 1)
    nvfp4_group_gemm_persistent(
        __grid_constant__ const CUtensorMap initial_a_map,
        __grid_constant__ const CUtensorMap initial_b_map,
        __grid_constant__ const std::array<char *, G> A_ptrs,
        __grid_constant__ const std::array<char *, G> B_ptrs,
        __grid_constant__ const std::array<char *, G> SFA_ptrs,
        __grid_constant__ const std::array<char *, G> SFB_ptrs,
        __grid_constant__ const std::array<half *, G> C_ptrs,
        __grid_constant__ const std::array<uint32_t, 4 * G> grouped_layout,
        CUtensorMap *tensormap_buffer) {
  extern __shared__ __align__(1024) char shmem[];
  __shared__ __align__(8) uint64_t bars[2 * PIPE_DEPTH + 2 * PIPE_DEPTH_EPI];
  __shared__ __align__(128) CUtensorMap smem_tensormaps[2];
  __shared__ uint32_t tmem_base[1];

  uint64_t *full_bar = bars;
  uint64_t *empty_bar = bars + PIPE_DEPTH;
  uint64_t *tmem_full_bar = bars + 2 * PIPE_DEPTH;
  uint64_t *tmem_empty_bar = bars + 2 * PIPE_DEPTH + PIPE_DEPTH_EPI;

  constexpr int32_t TILE_A_BYTES = BLOCK_K * BLOCK_M / 2;
  constexpr int32_t TILE_B_BYTES = BLOCK_K * BLOCK_N / 2;
  constexpr int32_t TILE_SFA_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t TILE_SFB_BYTES = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t STAGE_BYTES =
      TILE_A_BYTES + TILE_B_BYTES + TILE_SFA_BYTES + TILE_SFB_BYTES;

  char *pipe_start = shmem;

  constexpr int32_t TMEM_WIDTH_SFA = 16;
  constexpr int32_t TMEM_WIDTH_SFB = 16;

  const int32_t cta_id = make_warp_uniform(blockIdx.x);
  const int32_t warp_id = make_warp_uniform(threadIdx.x / WARP_SIZE);
  const int32_t lane_id = threadIdx.x % WARP_SIZE;

  if (warp_id == 1 && elect_one_sync()) {
    prefetch_tensormap(&initial_a_map);
    prefetch_tensormap(&initial_b_map);
  }
  __syncwarp();

  const CUtensorMap *current_a_map = &initial_a_map;
  const CUtensorMap *current_b_map = &initial_b_map;

  CUtensorMap *gmem_a_map = tensormap_buffer + cta_id * 2;
  CUtensorMap *gmem_b_map = tensormap_buffer + cta_id * 2 + 1;

  CUtensorMap *smem_a_map = smem_tensormaps;
  CUtensorMap *smem_b_map = smem_tensormaps + 1;

  Scheduler<G, BLOCK_M, BLOCK_N, NUM_SMS> scheduler(grouped_layout.data());
  int32_t group_idx = 0;
  int32_t m_block_idx, n_block_idx;
  int32_t M, N, K;
  bool is_swap, is_transpose;

  if (warp_id == TMA_WARP && elect_one_sync()) {
    for (int32_t i = 0; i < PIPE_DEPTH; i++) {
      init_barrier(&full_bar[i], 1);
      init_barrier(&empty_bar[i], 1);
    }
    for (int32_t i = 0; i < PIPE_DEPTH_EPI; i++) {
      init_barrier(&tmem_full_bar[i], 1);
      init_barrier(&tmem_empty_bar[i], WARPGROUP_SIZE);
    }
    fence_barrier_init();
  } else if (warp_id == MMA_WARP && elect_one_sync()) {
    *smem_a_map = initial_a_map;
    *smem_b_map = initial_b_map;
  } else if (warp_id == 0) {
    tcgen05_alloc(tmem_base, 512);
  }
  __syncthreads();

  int32_t pipe_idx = 0;
  int32_t phase = 0;
  auto advance_pipeline = [&]() {
    pipe_idx = (pipe_idx == PIPE_DEPTH - 1) ? 0 : pipe_idx + 1;
    phase ^= (pipe_idx == 0);
  };

  int32_t epi_pipe_idx = 0;
  int32_t epi_phase = 0;
  auto advance_epi_pipeline = [&]() {
    epi_pipe_idx = (epi_pipe_idx == PIPE_DEPTH_EPI - 1) ? 0 : epi_pipe_idx + 1;
    epi_phase ^= (epi_pipe_idx == 0);
  };

  if (warp_id < 4) {
    warpgroup_reg_alloc<256>();

    half *C_ptr;
    while (scheduler.next(m_block_idx, n_block_idx)) {
      M = scheduler.problem_m;
      N = scheduler.problem_n;
      is_swap = scheduler.problem_swap;
      is_transpose = scheduler.problem_transpose;
      C_ptr = C_ptrs[scheduler.current_group_idx];

      constexpr int32_t UNROLL_FACTOR = 4;
      float acc[BLOCK_N / UNROLL_FACTOR];

      uint32_t tmem_d = epi_pipe_idx * BLOCK_N;
      wait(&tmem_full_bar[epi_pipe_idx], epi_phase);

      auto epilogue_n = [&]() {
        half *C_off = C_ptr + m_block_idx * BLOCK_M * N + n_block_idx * BLOCK_N;

        for (int32_t r0 = 0; r0 < UNROLL_FACTOR / 2; r0++) {
          for (int32_t r1 = 0; r1 < 2; r1++) {
            tcgen05_ld_16x256b<BLOCK_N / UNROLL_FACTOR / 4>(
                tmem_d + r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) + r1 * (16 << 16),
                acc);
            tcgen05_wait_ld();

            for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i += 4) {
              const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
              const int32_t col = r0 * (BLOCK_N / (UNROLL_FACTOR / 2)) +
                                  (lane_id % 4) * 2 + 2 * i;

              if (m_block_idx * BLOCK_M + row < M &&
                  n_block_idx * BLOCK_N + col < N) {
                half2 vec1 =
                    __float22half2_rn(make_float2(acc[i + 0], acc[i + 1]));
                reinterpret_cast<half2 *>(&C_off[row * N + col])[0] = vec1;
              }
              if (m_block_idx * BLOCK_M + row + 8 < M &&
                  n_block_idx * BLOCK_N + col < N) {
                half2 vec2 =
                    __float22half2_rn(make_float2(acc[i + 2], acc[i + 3]));
                reinterpret_cast<half2 *>(&C_off[(row + 8) * N + col])[0] =
                    vec2;
              }
            }
          }
        }
      };

      auto epilogue_t = [&]() {
        half *C_off = C_ptr + n_block_idx * BLOCK_N * M + m_block_idx * BLOCK_M;

        for (int32_t r = 0; r < UNROLL_FACTOR; r++) {
          tcgen05_ld_32x32b<BLOCK_N / UNROLL_FACTOR>(
              tmem_d + r * (BLOCK_N / UNROLL_FACTOR), acc);
          tcgen05_wait_ld();

          for (int32_t i = 0; i < BLOCK_N / UNROLL_FACTOR; i++) {
            const int32_t row = threadIdx.x;
            const int32_t col = r * (BLOCK_N / UNROLL_FACTOR) + i;
            if (m_block_idx * BLOCK_M + row < M &&
                n_block_idx * BLOCK_N + col < N) {
              C_off[col * M + row] = __float2half_rn(acc[i]);
            }
          }
        }
      };

      is_transpose ? epilogue_t() : epilogue_n();

      arrive(&tmem_empty_bar[epi_pipe_idx], 1);
      advance_epi_pipeline();
    }
    sync_wg(0);
    if (warp_id == 0) {
      tcgen05_dealloc(0, 512);
    }
  } else {
    char *SFA_ptr, *SFB_ptr;
    if (warp_id == TMA_WARP) {
      while (scheduler.next(m_block_idx, n_block_idx)) {
        M = scheduler.problem_m;
        N = scheduler.problem_n;
        K = scheduler.problem_k;

        if (group_idx != scheduler.current_group_idx) {
          group_idx = scheduler.current_group_idx;

          if (elect_one_sync()) {
            tensormap_replace_global_address(smem_a_map, A_ptrs[group_idx]);
            tensormap_replace_global_dim<0>(smem_a_map, K);
            tensormap_replace_global_dim<1>(smem_a_map, M);
            tensormap_replace_global_stride<0>(smem_a_map, K / 2);

            tensormap_replace_global_address(smem_b_map, B_ptrs[group_idx]);
            tensormap_replace_global_dim<0>(smem_b_map, K);
            tensormap_replace_global_dim<1>(smem_b_map, N);
            tensormap_replace_global_stride<0>(smem_b_map, K / 2);

            tma_wait_until_pending<0>();
          }
          __syncwarp();

          tensormap_cp_and_fence(gmem_a_map, smem_a_map);
          tensormap_cp_and_fence(gmem_b_map, smem_b_map);

          tensormap_acquire(gmem_a_map);
          tensormap_acquire(gmem_b_map);

          current_a_map = gmem_a_map;
          current_b_map = gmem_b_map;
        }

        SFA_ptr = SFA_ptrs[group_idx];
        SFB_ptr = SFB_ptrs[group_idx];

        CachePolicy cache_policy_a = CachePolicy::EVICT_NORMAL;
        CachePolicy cache_policy_b = CachePolicy::EVICT_NORMAL;

        if (elect_one_sync()) {
          for (int32_t k = 0; k < (K / BLOCK_K); k++) {
            wait(&empty_bar[pipe_idx], phase ^ 1);
            expect_bytes_and_arrive(&full_bar[pipe_idx], STAGE_BYTES);

            char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
            char *b_shr = a_shr + TILE_A_BYTES;
            char *sfa_shr = b_shr + TILE_B_BYTES;
            char *sfb_shr = sfa_shr + TILE_SFA_BYTES;

            cp_async_bulk_tensor_2d_global_to_shared(
                a_shr, current_a_map, k * BLOCK_K, m_block_idx * BLOCK_M,
                &full_bar[pipe_idx], cache_policy_a);
            cp_async_bulk_tensor_2d_global_to_shared(
                b_shr, current_b_map, k * BLOCK_K, n_block_idx * BLOCK_N,
                &full_bar[pipe_idx], cache_policy_b);

            cp_async_bulk_global_to_shared(
                sfa_shr,
                SFA_ptr +
                    ((m_block_idx * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) +
                     k * 4) *
                        512,
                TILE_SFA_BYTES, &full_bar[pipe_idx], cache_policy_a);
            cp_async_bulk_global_to_shared(
                sfb_shr,
                SFB_ptr +
                    ((n_block_idx * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                     k * 4) *
                        512,
                TILE_SFB_BYTES, &full_bar[pipe_idx], cache_policy_b);

            advance_pipeline();
          }
        }
      }
    } else if (warp_id == MMA_WARP && elect_one_sync()) {
      constexpr uint32_t tmem_sfa = PIPE_DEPTH_EPI * BLOCK_N;
      constexpr uint32_t tmem_sfb = tmem_sfa + TMEM_WIDTH_SFA;

      constexpr uint32_t inst_desc =
          make_inst_desc<BLOCK_M, BLOCK_N, UMMA_K, 0, 0>();

      while (scheduler.next(m_block_idx, n_block_idx)) {
        K = scheduler.problem_k;

        const uint32_t tmem_d = epi_pipe_idx * BLOCK_N;
        const uint32_t sfa_offset = (BLOCK_M == 64) ? (m_block_idx % 2) * 2 : 0;
        const uint32_t sfb_offset = (BLOCK_N == 64) ? (n_block_idx % 2) * 2 : 0;

        wait(&tmem_empty_bar[epi_pipe_idx], epi_phase ^ 1);
        for (int32_t k = 0; k < (K / BLOCK_K); k++) {
          char *a_shr = pipe_start + pipe_idx * STAGE_BYTES;
          char *sfa_shr = a_shr + TILE_A_BYTES + TILE_B_BYTES;

          const uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_shr, 1, 1024);
          const uint64_t desc_b = desc_a + (TILE_A_BYTES >> 4);
          const uint64_t desc_sfa =
              make_smem_desc<NO_SWIZZLE>(sfa_shr, 128, 128);
          const uint64_t desc_sfb = desc_sfa + (TILE_SFA_BYTES >> 4);

          wait(&full_bar[pipe_idx], phase);

          for (int32_t j = 0; j < 4; j++) {
            tcgen05_cp_32x128b_x4(desc_sfa + j * 32, tmem_sfa + j * 4);
            tcgen05_cp_32x128b_x4(desc_sfb + j * 32, tmem_sfb + j * 4);
          }
          for (int32_t j = 0; j < 4; j++) {
            if (j == 0 && k == 0) {
              tcgen05_mma<1, 0, inst_desc>(
                  desc_a + j * 2, desc_b + j * 2, tmem_d,
                  tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                  tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            } else {
              tcgen05_mma<1, 1, inst_desc>(
                  desc_a + j * 2, desc_b + j * 2, tmem_d,
                  tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                  tmem_sfb + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            }
          }
          tcgen05_commit(&empty_bar[pipe_idx]);
          advance_pipeline();
        }
        tcgen05_commit(&tmem_full_bar[epi_pipe_idx]);
        advance_epi_pipeline();
      }
    }
  }
}

template <int32_t SHAPE_K, int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t PIPE_DEPTH, int32_t PIPE_DEPTH_EPI, int32_t NUM_SMS>
void launch_nvfp4_group_gemm_g8(const Group *groups) {
  constexpr int32_t BLOCK_DIM = WARPGROUP_SIZE + 2 * WARP_SIZE;

  std::array<int32_t, 8> psums;
  std::array<int32_t, 8> M_arr;
  std::array<int32_t, 8> N_arr;
  std::array<CUtensorMap, 8> A_maps;
  std::array<CUtensorMap, 8> B_maps;
  std::array<const char *__restrict__, 8> SFA_ptrs;
  std::array<const char *__restrict__, 8> SFB_ptrs;
  std::array<half *__restrict__, 8> C_ptrs;

  int psum = 0;
  for (int g = 0; g < 8; g++) {
    M_arr[g] = groups[g].M;
    N_arr[g] = groups[g].N;

    const int32_t num_m_blocks = CEIL_DIV(M_arr[g], BLOCK_M);
    const int32_t num_n_blocks = CEIL_DIV(N_arr[g], BLOCK_N);
    psum += num_m_blocks * num_n_blocks;
    psums[g] = psum;

    const cuuint64_t globalDimA[2] = {SHAPE_K, groups[g].M};
    const cuuint64_t globalStridesA[1] = {SHAPE_K / 2};
    const cuuint32_t boxDimA[2] = {BLOCK_K, BLOCK_M};
    const cuuint32_t elementStridesA[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &A_maps[g], CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, groups[g].A_ptr,
        globalDimA, globalStridesA, boxDimA, elementStridesA,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    const cuuint64_t globalDimB[2] = {SHAPE_K, groups[g].N};
    const cuuint64_t globalStridesB[1] = {SHAPE_K / 2};
    const cuuint32_t boxDimB[2] = {BLOCK_K, BLOCK_N};
    const cuuint32_t elementStridesB[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &B_maps[g], CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, groups[g].B_ptr,
        globalDimB, globalStridesB, boxDimB, elementStridesB,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    SFA_ptrs[g] = groups[g].SFA_ptr;
    SFB_ptrs[g] = groups[g].SFB_ptr;
    C_ptrs[g] = groups[g].C_ptr;
  }

  constexpr int32_t shmem_size_a = PIPE_DEPTH * (BLOCK_K * BLOCK_M) / 2;
  constexpr int32_t shmem_size_b = PIPE_DEPTH * (BLOCK_K * BLOCK_N) / 2;
  constexpr int32_t shmem_size_sfa = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size_sfb = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size =
      shmem_size_a + shmem_size_b + shmem_size_sfa + shmem_size_sfb;
  static_assert(shmem_size <= 227 * 1024, "Shared memory size exceeds 227 KB");
  cudaFuncSetAttribute(
      nvfp4_group_gemm_g8_cutlass<SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
                                  PIPE_DEPTH, PIPE_DEPTH_EPI, NUM_SMS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

  nvfp4_group_gemm_g8_cutlass<SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH,
                              PIPE_DEPTH_EPI, NUM_SMS>
      <<<NUM_SMS, BLOCK_DIM, shmem_size>>>(
          psums, M_arr, N_arr, A_maps, B_maps, SFA_ptrs, SFB_ptrs, C_ptrs,
          groups[0].swap, groups[0].transpose != groups[0].swap);
}

template <int32_t SHAPE_N, int32_t SHAPE_K, int32_t BLOCK_M, int32_t BLOCK_N,
          int32_t BLOCK_K, int32_t PIPE_DEPTH>
void launch_nvfp4_group_gemm_g2(const Group *groups) {
  constexpr int32_t BLOCK_DIM = WARPGROUP_SIZE + 2 * WARP_SIZE;

  const int32_t M1 = groups[0].M, M2 = groups[1].M;
  const int32_t N1 = groups[0].N, N2 = groups[1].N;

  char *A1_ptr = groups[0].A_ptr, *A2_ptr = groups[1].A_ptr;
  char *B1_ptr = groups[0].B_ptr, *B2_ptr = groups[1].B_ptr;
  char *SFA1_ptr = groups[0].SFA_ptr, *SFA2_ptr = groups[1].SFA_ptr;
  char *SFB1_ptr = groups[0].SFB_ptr, *SFB2_ptr = groups[1].SFB_ptr;
  half *C1_ptr = groups[0].C_ptr, *C2_ptr = groups[1].C_ptr;

  CUtensorMap a1_map, a2_map;
  const cuuint64_t globalDimA1[2] = {SHAPE_K, M1};
  const cuuint64_t globalStridesA1[1] = {SHAPE_K / 2};
  const cuuint64_t globalDimA2[2] = {SHAPE_K, M2};
  const cuuint64_t globalStridesA2[1] = {SHAPE_K / 2};
  const cuuint32_t boxDimA[2] = {BLOCK_K, BLOCK_M};
  const cuuint32_t elementStridesA[2] = {1, 1};
  cuTensorMapEncodeTiled(
      &a1_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, A1_ptr, globalDimA1,
      globalStridesA1, boxDimA, elementStridesA, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cuTensorMapEncodeTiled(
      &a2_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, A2_ptr, globalDimA2,
      globalStridesA2, boxDimA, elementStridesA, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CUtensorMap b1_map, b2_map;
  const cuuint64_t globalDimB1[2] = {SHAPE_K, N1};
  const cuuint64_t globalStridesB1[1] = {SHAPE_K / 2};
  const cuuint64_t globalDimB2[2] = {SHAPE_K, N2};
  const cuuint64_t globalStridesB2[1] = {SHAPE_K / 2};
  const cuuint32_t boxDimB[2] = {BLOCK_K, BLOCK_N};
  const cuuint32_t elementStridesB[2] = {1, 1};
  cuTensorMapEncodeTiled(
      &b1_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, B1_ptr, globalDimB1,
      globalStridesB1, boxDimB, elementStridesB, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cuTensorMapEncodeTiled(
      &b2_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, B2_ptr, globalDimB2,
      globalStridesB2, boxDimB, elementStridesB, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  const bool swapped = groups[0].swap;
  const bool transpose = groups[0].transpose != groups[0].swap;
  const bool skip_check = swapped ? (N1 % BLOCK_N == 0) && (N2 % BLOCK_N == 0)
                                  : (M1 % BLOCK_N == 0) && (M2 % BLOCK_N == 0);

  const int32_t psum1 = CEIL_DIV(M1, BLOCK_M) * CEIL_DIV(N1, BLOCK_N);
  const int32_t psum2 = psum1 + CEIL_DIV(M2, BLOCK_M) * CEIL_DIV(N2, BLOCK_N);

  constexpr int32_t shmem_size_a = PIPE_DEPTH * (BLOCK_K * BLOCK_M) / 2;
  constexpr int32_t shmem_size_b = PIPE_DEPTH * (BLOCK_K * BLOCK_N) / 2;
  constexpr int32_t shmem_size_sfa = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size_sfb = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size =
      shmem_size_a + shmem_size_b + shmem_size_sfa + shmem_size_sfb;
  static_assert(shmem_size <= 227 * 1024, "Shared memory size exceeds 227 KB");

#define LAUNCH(SWAPPED, TRANSPOSE, SKIP_CHECK)                                 \
  else if ((swapped == (SWAPPED)) && (transpose == (TRANSPOSE)) &&             \
           (skip_check == (SKIP_CHECK))) {                                     \
    cudaFuncSetAttribute(                                                      \
        nvfp4_group_gemm_g2_cutlass<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,        \
                                    BLOCK_K, PIPE_DEPTH, (SWAPPED),            \
                                    (TRANSPOSE), (SKIP_CHECK)>,                \
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);              \
    nvfp4_group_gemm_g2_cutlass<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,   \
                                PIPE_DEPTH, (SWAPPED), (TRANSPOSE),            \
                                (SKIP_CHECK)>                                  \
        <<<psum2, BLOCK_DIM, shmem_size>>>(                                    \
            psum1, M1, M2, N1, N2, a1_map, a2_map, b1_map, b2_map, SFA1_ptr,   \
            SFA2_ptr, SFB1_ptr, SFB2_ptr, C1_ptr, C2_ptr);                     \
  }

  if (false) {
  }
  LAUNCH(false, false, true)
  LAUNCH(true, false, false)
  /*
  LAUNCH(true, true, true)
  LAUNCH(true, true, false)
  LAUNCH(true, false, true)
  LAUNCH(false, true, true)
  LAUNCH(false, true, false)
  LAUNCH(false, false, false)
  */

#undef LAUNCH
}

template <int32_t SHAPE_N, int32_t SHAPE_K, int32_t BLOCK_M, int32_t BLOCK_N,
          int32_t BLOCK_K, int32_t PIPE_DEPTH>
void launch_nvfp4_group_gemm_g2_2sm(const Group *groups) {
  constexpr int32_t BLOCK_DIM = WARPGROUP_SIZE + 2 * WARP_SIZE;

  const int32_t M1 = groups[0].M, M2 = groups[1].M;
  const int32_t N1 = groups[0].N, N2 = groups[1].N;

  char *A1_ptr = groups[0].A_ptr, *A2_ptr = groups[1].A_ptr;
  char *B1_ptr = groups[0].B_ptr, *B2_ptr = groups[1].B_ptr;
  char *SFA1_ptr = groups[0].SFA_ptr, *SFA2_ptr = groups[1].SFA_ptr;
  char *SFB1_ptr = groups[0].SFB_ptr, *SFB2_ptr = groups[1].SFB_ptr;
  half *C1_ptr = groups[0].C_ptr, *C2_ptr = groups[1].C_ptr;

  CUtensorMap a1_map, a2_map;
  const cuuint64_t globalDimA1[3] = {256, M1, SHAPE_K / 256};
  const cuuint64_t globalStridesA1[2] = {SHAPE_K / 2, 128};
  const cuuint64_t globalDimA2[3] = {256, M2, SHAPE_K / 256};
  const cuuint64_t globalStridesA2[2] = {SHAPE_K / 2, 128};
  const cuuint32_t boxDimA[3] = {256, BLOCK_M, BLOCK_K / 256};
  const cuuint32_t elementStridesA[3] = {1, 1, 1};
  cuTensorMapEncodeTiled(
      &a1_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, A1_ptr, globalDimA1,
      globalStridesA1, boxDimA, elementStridesA, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cuTensorMapEncodeTiled(
      &a2_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, A2_ptr, globalDimA2,
      globalStridesA2, boxDimA, elementStridesA, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CUtensorMap b1_map, b2_map;
  const cuuint64_t globalDimB1[3] = {256, N1, SHAPE_K / 256};
  const cuuint64_t globalStridesB1[2] = {SHAPE_K / 2, 128};
  const cuuint64_t globalDimB2[3] = {256, N2, SHAPE_K / 256};
  const cuuint64_t globalStridesB2[2] = {SHAPE_K / 2, 128};
  const cuuint32_t boxDimB[3] = {256, BLOCK_N / 2, BLOCK_K / 256};
  const cuuint32_t elementStridesB[3] = {1, 1, 1};
  cuTensorMapEncodeTiled(
      &b1_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, B1_ptr, globalDimB1,
      globalStridesB1, boxDimB, elementStridesB, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cuTensorMapEncodeTiled(
      &b2_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 3, B2_ptr, globalDimB2,
      globalStridesB2, boxDimB, elementStridesB, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CUtensorMap sfa1_map, sfa2_map;
  const cuuint64_t globalDimSFA1[1] = {(SHAPE_K / SF_VEC_SIZE) * M1 / 8};
  const cuuint64_t globalDimSFA2[1] = {(SHAPE_K / SF_VEC_SIZE) * M2 / 8};
  const cuuint64_t globalStridesSFA[0] = {};
  const cuuint32_t boxDimSFA[1] = {(BLOCK_K / SF_VEC_SIZE) * 128 / 8};
  const cuuint32_t elementStridesSFA[1] = {1};
  cuTensorMapEncodeTiled(
      &sfa1_map, CU_TENSOR_MAP_DATA_TYPE_INT64, 1, SFA1_ptr, globalDimSFA1,
      globalStridesSFA, boxDimSFA, elementStridesSFA,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cuTensorMapEncodeTiled(
      &sfa2_map, CU_TENSOR_MAP_DATA_TYPE_INT64, 1, SFA2_ptr, globalDimSFA2,
      globalStridesSFA, boxDimSFA, elementStridesSFA,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  CUtensorMap sfb1_map, sfb2_map;
  const cuuint64_t globalDimSFB1[1] = {(SHAPE_K / SF_VEC_SIZE) * N1 / 8};
  const cuuint64_t globalDimSFB2[1] = {(SHAPE_K / SF_VEC_SIZE) * N2 / 8};
  const cuuint64_t globalStridesSFB[0] = {};
  const cuuint32_t boxDimSFB[1] = {(BLOCK_K / SF_VEC_SIZE) * 128 / 8};
  const cuuint32_t elementStridesSFB[1] = {1};
  cuTensorMapEncodeTiled(
      &sfb1_map, CU_TENSOR_MAP_DATA_TYPE_INT64, 1, SFB1_ptr, globalDimSFB1,
      globalStridesSFB, boxDimSFB, elementStridesSFB,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  cuTensorMapEncodeTiled(
      &sfb2_map, CU_TENSOR_MAP_DATA_TYPE_INT64, 1, SFB2_ptr, globalDimSFB2,
      globalStridesSFB, boxDimSFB, elementStridesSFB,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  const bool swapped = groups[0].swap;
  const bool transpose = groups[0].transpose != groups[0].swap;
  const bool skip_check = swapped ? (N1 % BLOCK_N == 0) && (N2 % BLOCK_N == 0)
                                  : (M1 % BLOCK_N == 0) && (M2 % BLOCK_N == 0);

  const int32_t psum1 = CEIL_DIV(M1, BLOCK_M) * CEIL_DIV(N1, BLOCK_N);
  const int32_t psum2 = psum1 + CEIL_DIV(M2, BLOCK_M) * CEIL_DIV(N2, BLOCK_N);

  constexpr int32_t shmem_size_a = PIPE_DEPTH * (BLOCK_K * BLOCK_M) / 2;
  constexpr int32_t shmem_size_b = PIPE_DEPTH * (BLOCK_K * BLOCK_N / 2) / 2;
  constexpr int32_t shmem_size_sfa = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size_sfb = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size =
      shmem_size_a + shmem_size_b + shmem_size_sfa + shmem_size_sfb;
  static_assert(shmem_size <= 227 * 1024, "Shared memory size exceeds 227 KB");

#define LAUNCH(SWAPPED, TRANSPOSE, SKIP_CHECK)                                 \
  else if ((swapped == (SWAPPED)) && (transpose == (TRANSPOSE)) &&             \
           (skip_check == (SKIP_CHECK))) {                                     \
    cudaFuncSetAttribute(                                                      \
        nvfp4_group_gemm_g2_2sm_cutlass<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,    \
                                        BLOCK_K, PIPE_DEPTH, (SWAPPED),        \
                                        (TRANSPOSE), (SKIP_CHECK)>,            \
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);              \
    nvfp4_group_gemm_g2_2sm_cutlass<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N,        \
                                    BLOCK_K, PIPE_DEPTH, (SWAPPED),            \
                                    (TRANSPOSE), (SKIP_CHECK)>                 \
        <<<psum2, BLOCK_DIM, shmem_size>>>(                                    \
            psum1, M1, M2, N1, N2, a1_map, a2_map, b1_map, b2_map, sfa1_map,   \
            sfa2_map, sfb1_map, sfb2_map, C1_ptr, C2_ptr);                     \
  }

  if (false) {
  }
  LAUNCH(false, false, true)
  LAUNCH(true, false, false)
  LAUNCH(true, true, true)
  LAUNCH(true, true, false)
  LAUNCH(true, false, true)
  LAUNCH(false, true, true)
  LAUNCH(false, true, false)
  LAUNCH(false, false, false)

#undef LAUNCH
}

template <int32_t G, int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t PIPE_DEPTH, int32_t PIPE_DEPTH_EPI, int32_t NUM_SMS>
void launch_nvfp4_group_gemm_persistent(const Group *groups) {
  constexpr int32_t BLOCK_DIM = WARPGROUP_SIZE + 2 * WARP_SIZE;

  CUtensorMap initial_a_map;
  const cuuint64_t globalDimA[2] = {groups[0].K, groups[0].M};
  const cuuint64_t globalStridesA[1] = {groups[0].K / 2};
  const cuuint32_t boxDimA[2] = {BLOCK_K, BLOCK_M};
  const cuuint32_t elementStridesA[2] = {1, 1};
  CUDA_CHECK(cuTensorMapEncodeTiled(
      &initial_a_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, groups[0].A_ptr,
      globalDimA, globalStridesA, boxDimA, elementStridesA,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  CUtensorMap initial_b_map;
  const cuuint64_t globalDimB[2] = {groups[0].K, groups[0].N};
  const cuuint64_t globalStridesB[1] = {groups[0].K / 2};
  const cuuint32_t boxDimB[2] = {BLOCK_K, BLOCK_N};
  const cuuint32_t elementStridesB[2] = {1, 1};
  CUDA_CHECK(cuTensorMapEncodeTiled(
      &initial_b_map, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, groups[0].B_ptr,
      globalDimB, globalStridesB, boxDimB, elementStridesB,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  std::array<char *, G> A_ptrs;
  std::array<char *, G> B_ptrs;
  std::array<char *, G> SFA_ptrs;
  std::array<char *, G> SFB_ptrs;
  std::array<half *, G> C_ptrs;
  std::array<uint32_t, 4 * G> grouped_layout;

  int psum = 0;
  for (int g = 0; g < G; g++) {
    A_ptrs[g] = groups[g].A_ptr;
    B_ptrs[g] = groups[g].B_ptr;
    SFA_ptrs[g] = groups[g].SFA_ptr;
    SFB_ptrs[g] = groups[g].SFB_ptr;
    C_ptrs[g] = groups[g].C_ptr;

    int M = groups[g].M;
    int N = groups[g].N;
    int K = groups[g].K;

    int num_m_blocks = CEIL_DIV(M, BLOCK_M);
    int num_n_blocks = CEIL_DIV(N, BLOCK_N);
    psum += num_m_blocks * num_n_blocks;
    grouped_layout[g] = psum;
    if (groups[g].swap)
      grouped_layout[g] |= (1u << 31);
    if (groups[g].transpose != groups[g].swap)
      grouped_layout[g] |= (1u << 30);

    grouped_layout[G + 3 * g + 0] = M;
    grouped_layout[G + 3 * g + 1] = N;
    grouped_layout[G + 3 * g + 2] = K;
  }

  void *workspace;
  constexpr int32_t WORKSPACE_SIZE = NUM_SMS * 2 * 128;
  CUDA_CHECK(cudaMalloc(&workspace, WORKSPACE_SIZE));

  constexpr int32_t shmem_size_a = PIPE_DEPTH * (BLOCK_K * BLOCK_M) / 2;
  constexpr int32_t shmem_size_b = PIPE_DEPTH * (BLOCK_K * BLOCK_N) / 2;
  constexpr int32_t shmem_size_sfa = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size_sfb = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size =
      shmem_size_a + shmem_size_b + shmem_size_sfa + shmem_size_sfb;
  static_assert(shmem_size <= 227 * 1024, "Shared memory size exceeds 227 KB");
  cudaFuncSetAttribute(
      nvfp4_group_gemm_persistent<G, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH,
                                  PIPE_DEPTH_EPI, NUM_SMS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

  nvfp4_group_gemm_persistent<G, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH,
                              PIPE_DEPTH_EPI, NUM_SMS>
      <<<NUM_SMS, BLOCK_DIM, shmem_size>>>(
          initial_a_map, initial_b_map, A_ptrs, B_ptrs, SFA_ptrs, SFB_ptrs,
          C_ptrs, grouped_layout, reinterpret_cast<CUtensorMap *>(workspace));

  CUDA_CHECK(cudaFree(workspace));
}

#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_variable.h>

namespace py = pybind11;

py::object cuda_nvfp4_group_gemm(py::handle abc_tensors,
                                 py::handle sfasfb_tensors,
                                 py::handle problem_sizes) {
  PyObject *abc_obj = abc_tensors.ptr();
  PyObject *sfasfb_obj = sfasfb_tensors.ptr();
  PyObject *problem_sizes_obj = problem_sizes.ptr();

  const int G = (int)PyList_GET_SIZE(problem_sizes_obj);
  int SHAPE_N, SHAPE_K;
  bool swap = false, transpose = false;

  Group groups[8];
  PyObject *results = PyList_New(G);

#pragma unroll
  for (int g = 0; g < G; g++) {
    PyObject *abc_tuple = PyList_GET_ITEM(abc_obj, g);
    PyObject *sfasfb_tuple = PyList_GET_ITEM(sfasfb_obj, g);
    PyObject *problem_sizes_tuple = PyList_GET_ITEM(problem_sizes_obj, g);

    int M = (int)PyLong_AS_LONG(PyTuple_GET_ITEM(problem_sizes_tuple, 0));
    int N = (int)PyLong_AS_LONG(PyTuple_GET_ITEM(problem_sizes_tuple, 1));
    int K = (int)PyLong_AS_LONG(PyTuple_GET_ITEM(problem_sizes_tuple, 2));

    if (g == 0) {
      SHAPE_N = N;
      SHAPE_K = K;

      if (SHAPE_K == 7168) {
        swap = true;
        transpose = false;
      } else if (SHAPE_K == 2048) {
        swap = true;
        transpose = false;
      } else if (SHAPE_K == 4096) {
        swap = true;
        transpose = true;
      } else if (SHAPE_K == 1536) {
        swap = false;
        transpose = false;
      }
    }

    PyObject *A_obj = PyTuple_GET_ITEM(abc_tuple, 0);
    PyObject *B_obj = PyTuple_GET_ITEM(abc_tuple, 1);
    PyObject *C_obj = PyTuple_GET_ITEM(abc_tuple, 2);
    PyObject *SFA_obj = PyTuple_GET_ITEM(sfasfb_tuple, 0);
    PyObject *SFB_obj = PyTuple_GET_ITEM(sfasfb_tuple, 1);

    char *A_ptr =
        reinterpret_cast<char *>(((THPVariable *)A_obj)->cdata->data_ptr());
    char *B_ptr =
        reinterpret_cast<char *>(((THPVariable *)B_obj)->cdata->data_ptr());
    char *SFA_ptr =
        reinterpret_cast<char *>(((THPVariable *)SFA_obj)->cdata->data_ptr());
    char *SFB_ptr =
        reinterpret_cast<char *>(((THPVariable *)SFB_obj)->cdata->data_ptr());
    half *C_ptr =
        reinterpret_cast<half *>(((THPVariable *)C_obj)->cdata->data_ptr());

    int32_t old_M = M;
    int32_t old_N = N;

    if (swap) {
      std::swap(M, N);
      std::swap(A_ptr, B_ptr);
      std::swap(SFA_ptr, SFB_ptr);
    }

    groups[g] =
        Group{M, N, K, A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, swap, transpose};

    if (transpose) {
      const at::Tensor &C = *((THPVariable *)C_obj)->cdata;
      at::Tensor Ct = C.view({old_N, old_M, 1}).transpose(0, 1);
      PyObject *wrapped = py::cast(Ct).release().ptr();
      PyList_SET_ITEM(results, g, wrapped);
    } else {
      Py_INCREF(C_obj);
      PyList_SET_ITEM(results, g, C_obj);
    }
  }

  if (G == 8) {
    std::sort(groups, groups + G, [](const Group &a, const Group &b) {
      return std::min(a.M, a.N) > std::min(b.M, b.N);
    });
    if (SHAPE_K == 7168) {
      launch_nvfp4_group_gemm_g8<7168, 128, 128, 256, 6, 3, 128>(groups);
    } else if (SHAPE_K == 2048) {
      launch_nvfp4_group_gemm_g8<2048, 128, 64, 256, 8, 7, 148>(groups);
    } else {
      launch_nvfp4_group_gemm_persistent<8, 128, 128, 256, 6, 1, 148>(groups);
    }
  } else if (G == 2) {
    if (SHAPE_N == 3072 && SHAPE_K == 4096) {
      launch_nvfp4_group_gemm_g2<3072, 4096, 128, 128, 256, 6>(groups);
    } else if (SHAPE_N == 4096 && SHAPE_K == 1536) {
      launch_nvfp4_group_gemm_g2<4096, 1536, 128, 128, 256, 6>(groups);
    } else {
      launch_nvfp4_group_gemm_persistent<2, 128, 128, 256, 6, 1, 148>(groups);
    }
  } else if (G == 3) {
    launch_nvfp4_group_gemm_persistent<3, 128, 64, 256, 4, 1, 148>(groups);
  } else if (G == 4) {
    launch_nvfp4_group_gemm_persistent<4, 128, 64, 256, 4, 1, 148>(groups);
  }

  return py::reinterpret_steal<py::object>(results);
}
"""

my_module = load_inline(
    name="nvfp4_group_gemm",
    cpp_sources=[CPP_SRC],
    cuda_sources=CUDA_HEADERS + [CUDA_SRC],
    functions=["cuda_nvfp4_group_gemm"],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "--extra-device-vectorization",
        # "-lineinfo",
    ],
    extra_ldflags=["-lcuda"],
    verbose=True,
)
group_gemm = my_module.cuda_nvfp4_group_gemm


def custom_kernel(data: input_t) -> output_t:
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data
    return group_gemm(abc_tensors, sfasfb_reordered_tensors, problem_sizes)
