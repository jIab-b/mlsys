#pragma once

#include <stdint.h>

// Common helpers for PTX inline asm wrappers.

#if defined(__CUDA_ARCH__)
#define PTX_DEVICE __device__ __forceinline__

PTX_DEVICE inline uint32_t ptx_laneid() {
  uint32_t lane;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
  return lane;
}

PTX_DEVICE inline uint32_t ptx_activemask() {
  uint32_t mask;
  asm volatile("activemask.b32 %0;" : "=r"(mask));
  return mask;
}

PTX_DEVICE inline bool ptx_elect_one_sync() {
  uint32_t mask = ptx_activemask();
  int leader = __ffs(mask) - 1;
  return (int)ptx_laneid() == leader;
}

#else
#define PTX_DEVICE inline

PTX_DEVICE inline bool ptx_elect_one_sync() { return true; }
#endif

#ifndef PTX_NO_ELECT
#define PTX_ELECT_ONE()          \
  do {                           \
    if (!ptx_elect_one_sync()) { \
      return;                    \
    }                            \
  } while (0)
#else
#define PTX_ELECT_ONE() do { } while (0)
#endif
