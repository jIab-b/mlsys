#pragma once

#include "ptx_common.cuh"

// TMA bulk tensor loads (CTA group 1/2)

// Bulk global->shared copy (non-tensor)
PTX_DEVICE void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
              "[%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_1d_gmem2smem(int dst, const void *tmap_ptr, int x, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2}], [%3], %4;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2}], [%3], %4;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3}], [%4], %5;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_4d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_5d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_1d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2}], [%3], %4, %5;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2}], [%3], %4, %5;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_2d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3}], [%4], %5, %6;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3}], [%4], %5, %6;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_3d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_4d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_5d_gmem2smem_mcast(int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8, %9;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8, %9;"
                :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_2d_gmem2smem_gather4(int dst, const void *tmap_ptr, int col, int row0, int row1, int row2, int row3, int mbar_addr, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                :: "r"(dst), "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                :: "r"(dst), "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(mbar_addr), "l"(cache_policy)
                : "memory");
  }
}

template <int CTA_GROUP = 1>
PTX_DEVICE void tma_2d_gmem2smem_gather4_mcast(int dst, const void *tmap_ptr, int col, int row0, int row1, int row2, int row3, int mbar_addr, int16_t cta_mask, uint64_t cache_policy) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2, "CTA_GROUP must be 1 or 2");
  PTX_ELECT_ONE();
  if constexpr (CTA_GROUP == 1) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8, %9;"
                :: "r"(dst), "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint "
                "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8, %9;"
                :: "r"(dst), "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(mbar_addr), "h"(cta_mask), "l"(cache_policy)
                : "memory");
  }
}

PTX_DEVICE void tma_1d_smem2gmem(const void *tmap_ptr, int x, int src, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group.L2::cache_hint "
              "[%0, {%1}], [%2], %3;"
              :: "l"(tmap_ptr), "r"(x), "r"(src), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_2d_smem2gmem(const void *tmap_ptr, int x, int y, int src, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint "
              "[%0, {%1, %2}], [%3], %4;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(src), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_3d_smem2gmem(const void *tmap_ptr, int x, int y, int z, int src, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint "
              "[%0, {%1, %2, %3}], [%4], %5;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(src), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_4d_smem2gmem(const void *tmap_ptr, int x, int y, int z, int w, int src, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group.L2::cache_hint "
              "[%0, {%1, %2, %3, %4}], [%5], %6;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(src), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_5d_smem2gmem(const void *tmap_ptr, int x, int y, int z, int w, int v, int src, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group.L2::cache_hint "
              "[%0, {%1, %2, %3, %4, %5}], [%6], %7;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(src), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_2d_smem2gmem_scatter4(const void *tmap_ptr, int col, int row0, int row1, int row2, int row3, int src, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group.L2::cache_hint "
              "[%0, {%1, %2, %3, %4, %5}], [%6], %7;"
              :: "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(src), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_prefetch_1d(const void *tmap_ptr, int x, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint "
              "[%0, {%1}], %2;"
              :: "l"(tmap_ptr), "r"(x), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_prefetch_2d(const void *tmap_ptr, int x, int y, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint "
              "[%0, {%1, %2}], %3;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_prefetch_3d(const void *tmap_ptr, int x, int y, int z, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint "
              "[%0, {%1, %2, %3}], %4;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_prefetch_4d(const void *tmap_ptr, int x, int y, int z, int w, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint "
              "[%0, {%1, %2, %3, %4}], %5;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_prefetch_5d(const void *tmap_ptr, int x, int y, int z, int w, int v, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint "
              "[%0, {%1, %2, %3, %4, %5}], %6;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "l"(cache_policy)
              : "memory");
}

PTX_DEVICE void tma_prefetch_2d_gather4(const void *tmap_ptr, int col, int row0, int row1, int row2, int row3, uint64_t cache_policy) {
  PTX_ELECT_ONE();
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint "
              "[%0, {%1, %2, %3, %4, %5}], %6;"
              :: "l"(tmap_ptr), "r"(col), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "l"(cache_policy)
              : "memory");
}
