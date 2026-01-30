#pragma once

#include <stdint.h>

// Common helpers for PTX inline asm wrappers.

#if defined(__CUDA_ARCH__)
#define PTX_DEVICE __device__ __forceinline__

PTX_DEVICE uint32_t ptx_laneid() {
  uint32_t lane;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(lane));
  return lane;
}

PTX_DEVICE uint32_t ptx_activemask() {
  uint32_t mask;
  asm volatile("activemask.b32 %0;" : "=r"(mask));
  return mask;
}

PTX_DEVICE bool ptx_elect_one_sync() {
  uint32_t mask = ptx_activemask();
  int leader = __ffs(mask) - 1;
  return (int)ptx_laneid() == leader;
}

PTX_DEVICE uint32_t ptx_elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}\n\t"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

#else
#define PTX_DEVICE __device__ __forceinline__

PTX_DEVICE uint32_t ptx_laneid() { return 0; }
PTX_DEVICE uint32_t ptx_activemask() { return 0xFFFFFFFF; }
PTX_DEVICE bool ptx_elect_one_sync() { return true; }
PTX_DEVICE uint32_t ptx_elect_sync() { return 1; }
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

PTX_DEVICE void ptx_bar_sync(int bar_id, int count) {
  asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count) : "memory");
}
