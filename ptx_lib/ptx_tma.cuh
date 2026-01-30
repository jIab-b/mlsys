#pragma once

#include "ptx_common.cuh"

// TMA bulk tensor loads (CTA group 1 only for now)

// Bulk global->shared copy (non-tensor)
PTX_DEVICE inline void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
              "[%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tma_1d_gmem2smem(int dst, const void *tmap_ptr, int x, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2}], [%3], %4;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3}], [%4], %5;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tma_1d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2}], [%3], %4, %5;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
              : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tma_2d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3}], [%4], %5, %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
              : "memory");
}

template <int CTA_GROUP = 1>
PTX_DEVICE inline void tma_3d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1, "Only CTA_GROUP=1 supported for now");
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
              : "memory");
}
