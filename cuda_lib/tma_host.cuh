#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdint.h>

// Host-side helpers for CUtensorMap tiled encodes (no im2col variants).
// These are convenience wrappers around cuTensorMapEncodeTiled/ReplaceAddress.

inline void tma_check_cu(CUresult err, const char *context) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr = nullptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS || error_msg_ptr == nullptr) {
    error_msg_ptr = "unable to get error string";
  }
  // Keep this simple: callers can replace with TORCH_CHECK if desired.
  fprintf(stderr, "tma_host error (%s): %s\n", context, error_msg_ptr);
  abort();
}

inline void tma_replace_address(CUtensorMap *tmap, void *global_addr) {
  tma_check_cu(cuTensorMapReplaceAddress(tmap, global_addr), "cuTensorMapReplaceAddress");
}

template <int Rank>
inline void tma_encode_tiled(
    CUtensorMap *tmap,
    CUtensorMapDataType dtype,
    void *global_addr,
    const uint64_t (&global_dim)[Rank],
    const uint64_t (&global_stride)[Rank - 1],
    const uint32_t (&box_dim)[Rank],
    const uint32_t (&elem_stride)[Rank],
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  static_assert(Rank >= 1 && Rank <= 5, "Rank must be in [1, 5]");
  tma_check_cu(
      cuTensorMapEncodeTiled(
          tmap,
          dtype,
          Rank,
          global_addr,
          global_dim,
          global_stride,
          box_dim,
          elem_stride,
          interleave,
          swizzle,
          l2,
          oob),
      "cuTensorMapEncodeTiled");
}

// Explicit rank helpers (avoid template deduction pitfalls in C++ bindings).
inline void tma_encode_tiled_1d(
    CUtensorMap *tmap,
    CUtensorMapDataType dtype,
    void *global_addr,
    uint64_t g0,
    uint32_t b0,
    uint32_t e0,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  const uint64_t global_dim[1] = {g0};
  const uint64_t global_stride[1] = {0};
  const uint32_t box_dim[1] = {b0};
  const uint32_t elem_stride[1] = {e0};
  tma_encode_tiled<1>(
      tmap, dtype, global_addr, global_dim, global_stride, box_dim, elem_stride, interleave, swizzle, l2, oob);
}

inline void tma_encode_tiled_2d(
    CUtensorMap *tmap,
    CUtensorMapDataType dtype,
    void *global_addr,
    uint64_t g0,
    uint64_t g1,
    uint64_t s0,
    uint32_t b0,
    uint32_t b1,
    uint32_t e0,
    uint32_t e1,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  const uint64_t global_dim[2] = {g0, g1};
  const uint64_t global_stride[1] = {s0};
  const uint32_t box_dim[2] = {b0, b1};
  const uint32_t elem_stride[2] = {e0, e1};
  tma_encode_tiled<2>(
      tmap, dtype, global_addr, global_dim, global_stride, box_dim, elem_stride, interleave, swizzle, l2, oob);
}

inline void tma_encode_tiled_3d(
    CUtensorMap *tmap,
    CUtensorMapDataType dtype,
    void *global_addr,
    uint64_t g0,
    uint64_t g1,
    uint64_t g2,
    uint64_t s0,
    uint64_t s1,
    uint32_t b0,
    uint32_t b1,
    uint32_t b2,
    uint32_t e0,
    uint32_t e1,
    uint32_t e2,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  const uint64_t global_dim[3] = {g0, g1, g2};
  const uint64_t global_stride[2] = {s0, s1};
  const uint32_t box_dim[3] = {b0, b1, b2};
  const uint32_t elem_stride[3] = {e0, e1, e2};
  tma_encode_tiled<3>(
      tmap, dtype, global_addr, global_dim, global_stride, box_dim, elem_stride, interleave, swizzle, l2, oob);
}

inline void tma_encode_tiled_4d(
    CUtensorMap *tmap,
    CUtensorMapDataType dtype,
    void *global_addr,
    uint64_t g0,
    uint64_t g1,
    uint64_t g2,
    uint64_t g3,
    uint64_t s0,
    uint64_t s1,
    uint64_t s2,
    uint32_t b0,
    uint32_t b1,
    uint32_t b2,
    uint32_t b3,
    uint32_t e0,
    uint32_t e1,
    uint32_t e2,
    uint32_t e3,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  const uint64_t global_dim[4] = {g0, g1, g2, g3};
  const uint64_t global_stride[3] = {s0, s1, s2};
  const uint32_t box_dim[4] = {b0, b1, b2, b3};
  const uint32_t elem_stride[4] = {e0, e1, e2, e3};
  tma_encode_tiled<4>(
      tmap, dtype, global_addr, global_dim, global_stride, box_dim, elem_stride, interleave, swizzle, l2, oob);
}

inline void tma_encode_tiled_5d(
    CUtensorMap *tmap,
    CUtensorMapDataType dtype,
    void *global_addr,
    uint64_t g0,
    uint64_t g1,
    uint64_t g2,
    uint64_t g3,
    uint64_t g4,
    uint64_t s0,
    uint64_t s1,
    uint64_t s2,
    uint64_t s3,
    uint32_t b0,
    uint32_t b1,
    uint32_t b2,
    uint32_t b3,
    uint32_t b4,
    uint32_t e0,
    uint32_t e1,
    uint32_t e2,
    uint32_t e3,
    uint32_t e4,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2,
    CUtensorMapFloatOOBfill oob) {
  const uint64_t global_dim[5] = {g0, g1, g2, g3, g4};
  const uint64_t global_stride[4] = {s0, s1, s2, s3};
  const uint32_t box_dim[5] = {b0, b1, b2, b3, b4};
  const uint32_t elem_stride[5] = {e0, e1, e2, e3, e4};
  tma_encode_tiled<5>(
      tmap, dtype, global_addr, global_dim, global_stride, box_dim, elem_stride, interleave, swizzle, l2, oob);
}
