# Profiler Control

## 6.39.Â Profiler Control

This section describes the profiler control functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuProfilerStart](<#group__CUDA__PROFILER_1g8a5314de2292c2efac83ac7fcfa9190e>) ( void )
     Enable profiling.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuProfilerStop](<#group__CUDA__PROFILER_1g4d8edef6174fd90165e6ac838f320a5f>) ( void )
     Disable profiling.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuProfilerStart ( void )


Enable profiling.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Enables profile collection by the active profiling tool for the current context. If profiling is already enabled, then [cuProfilerStart()](<group__CUDA__PROFILER.html#group__CUDA__PROFILER_1g8a5314de2292c2efac83ac7fcfa9190e> "Enable profiling.") has no effect.

cuProfilerStart and cuProfilerStop APIs are used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuProfilerInitialize](<group__CUDA__PROFILER__DEPRECATED.html#group__CUDA__PROFILER__DEPRECATED_1gd15d4f964bf948988679232a54ce9fc1> "Initialize the profiling."), [cuProfilerStop](<group__CUDA__PROFILER.html#group__CUDA__PROFILER_1g4d8edef6174fd90165e6ac838f320a5f> "Disable profiling."), [cudaProfilerStart](<../cuda-runtime-api/group__CUDART__PROFILER.html#group__CUDART__PROFILER_1gf536d75bb382356e10e3b4e89f4a5374>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuProfilerStop ( void )


Disable profiling.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Disables profile collection by the active profiling tool for the current context. If profiling is already disabled, then [cuProfilerStop()](<group__CUDA__PROFILER.html#group__CUDA__PROFILER_1g4d8edef6174fd90165e6ac838f320a5f> "Disable profiling.") has no effect.

cuProfilerStart and cuProfilerStop APIs are used to programmatically control the profiling granularity by allowing profiling to be done only on selective pieces of code.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuProfilerInitialize](<group__CUDA__PROFILER__DEPRECATED.html#group__CUDA__PROFILER__DEPRECATED_1gd15d4f964bf948988679232a54ce9fc1> "Initialize the profiling."), [cuProfilerStart](<group__CUDA__PROFILER.html#group__CUDA__PROFILER_1g8a5314de2292c2efac83ac7fcfa9190e> "Enable profiling."), [cudaProfilerStop](<../cuda-runtime-api/group__CUDART__PROFILER.html#group__CUDART__PROFILER_1g826922d9d1d0090d4a9a6b8b249cebb5>)

* * *
