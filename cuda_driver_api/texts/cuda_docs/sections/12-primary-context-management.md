# Primary Context Management

## 6.7.Â Primary Context Management

This section describes the primary context management functions of the low-level CUDA driver application programming interface.

The primary context is unique per device and shared with the CUDA runtime API. These functions allow integration with other libraries using CUDA.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevicePrimaryCtxGetState](<#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, unsigned int*Â flags, int*Â active )
     Get the state of the primary context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevicePrimaryCtxRelease](<#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Release the primary context on the GPU.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevicePrimaryCtxReset](<#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Destroy all allocations and reset all state on the primary context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevicePrimaryCtxRetain](<#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Retain the primary context on the GPU.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevicePrimaryCtxSetFlags](<#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, unsigned int Â flags )
     Set flags for the primary context.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevicePrimaryCtxGetState ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, unsigned int*Â flags, int*Â active )


Get the state of the primary context.

######  Parameters

`dev`
    \- Device to get primary context flags for
`flags`
    \- Pointer to store flags
`active`
    \- Pointer to store context state; 0 = inactive, 1 = active

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns in `*flags` the flags for the primary context of `dev`, and in `*active` whether it is active. See [cuDevicePrimaryCtxSetFlags](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf> "Set flags for the primary context.") for flag values.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDevicePrimaryCtxSetFlags](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf> "Set flags for the primary context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxSetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1g66655c37602c8628eae3e40c82619f1e> "Sets the flags for the current context."), [cudaGetDeviceFlags](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf830794caf068b71638c6182bba8f77a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevicePrimaryCtxRelease ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Release the primary context on the GPU.

######  Parameters

`dev`
    \- Device which primary context is released

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Releases the primary context interop on the device. A retained context should always be released once the user is done using it. The context is automatically reset once the last reference to it is released. This behavior is different when the primary context was retained by the CUDA runtime from CUDA 4.0 and earlier. In this case, the primary context remains always active.

Releasing a primary context that has not been previously retained will fail with [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>).

Please note that unlike [cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context.") this method does not pop the context from stack in any circumstances.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDevicePrimaryCtxRetain](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300> "Retain the primary context on the GPU."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevicePrimaryCtxReset ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Destroy all allocations and reset all state on the primary context.

######  Parameters

`dev`
    \- Device for which primary context is destroyed

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90d74711f16f63a410d8d94d02e1f5480>)

###### Description

Explicitly destroys and cleans up all resources associated with the current device in the current process.

Note that it is responsibility of the calling function to ensure that no other module in the process is using the device any more. For that reason it is recommended to use [cuDevicePrimaryCtxRelease()](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad> "Release the primary context on the GPU.") in most cases. However it is safe for other modules to call [cuDevicePrimaryCtxRelease()](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad> "Release the primary context on the GPU.") even after resetting the device. Resetting the primary context does not release it, an application that has retained the primary context should explicitly release its usage.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDevicePrimaryCtxRetain](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300> "Retain the primary context on the GPU."), [cuDevicePrimaryCtxRelease](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad> "Release the primary context on the GPU."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cudaDeviceReset](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gef69dd5c6d0206c2b8d099abac61f217>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevicePrimaryCtxRetain ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Retain the primary context on the GPU.

######  Parameters

`pctx`
    \- Returned context handle of the new context
`dev`
    \- Device for which primary context is requested

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Retains the primary context on the device. Once the user successfully retains the primary context, the primary context will be active and available to the user until the user releases it with [cuDevicePrimaryCtxRelease()](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad> "Release the primary context on the GPU.") or resets it with [cuDevicePrimaryCtxReset()](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g5d38802e8600340283958a117466ce12> "Destroy all allocations and reset all state on the primary context."). Unlike [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") the newly retained context is not pushed onto the stack.

Retaining the primary context for the first time will fail with [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>) if the compute mode of the device is [CU_COMPUTEMODE_PROHIBITED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg637aab2eadb52e1c1c048b8bad9592d1db8a226241187db3b1f41999bb70eb47>). The function [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") can be used with [CU_DEVICE_ATTRIBUTE_COMPUTE_MODE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3f6669a29a6d42968047747cbfc501289>) to determine the compute mode of the device. The nvidia-smi tool can be used to set the compute mode for devices. Documentation for nvidia-smi can be obtained by passing a -h option to it.

Please note that the primary context always supports pinned allocations. Other flags can be specified by [cuDevicePrimaryCtxSetFlags()](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf> "Set flags for the primary context.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDevicePrimaryCtxRelease](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gf2a8bc16f8df0c88031f6a1ba3d6e8ad> "Release the primary context on the GPU."), [cuDevicePrimaryCtxSetFlags](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf> "Set flags for the primary context."), [cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevicePrimaryCtxSetFlags ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, unsigned int Â flags )


Set flags for the primary context.

######  Parameters

`dev`
    \- Device for which the primary context flags are set
`flags`
    \- New flags for the device

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the flags for the primary context on the device overwriting perviously set ones.

The three LSBs of the `flags` parameter can be used to control how the OS thread, which owns the CUDA context at the time of an API call, interacts with the OS scheduler when waiting for results from the GPU. Only one of the scheduling flags can be set when creating a context.

  * [CU_CTX_SCHED_SPIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4331d3ed1e0b55597258bd58346603afc>): Instruct CUDA to actively spin when waiting for results from the GPU. This can decrease latency when waiting for the GPU, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.


  * [CU_CTX_SCHED_YIELD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44bc43364906d8dd5a7d7c8ad46ccc548>): Instruct CUDA to yield its thread when waiting for results from the GPU. This can increase latency when waiting for the GPU, but can increase the performance of CPU threads performing work in parallel with the GPU.


  * [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>): Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the GPU to finish work.


  * [CU_CTX_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4b5bf395cc60a8cbded4c329ae9430b91>): Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the GPU to finish work.

**Deprecated:** This flag was deprecated as of CUDA 4.0 and was replaced with [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>).


  * [CU_CTX_SCHED_AUTO](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4d75f4c24f8c35ef2ee9d0793badfd88c>): The default value if the `flags` parameter is zero, uses a heuristic based on the number of active CUDA contexts in the process C and the number of logical processors in the system P. If C > P, then CUDA will yield to other OS threads when waiting for the GPU ([CU_CTX_SCHED_YIELD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44bc43364906d8dd5a7d7c8ad46ccc548>)), otherwise CUDA will not yield while waiting for results and actively spin on the processor ([CU_CTX_SCHED_SPIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4331d3ed1e0b55597258bd58346603afc>)). Additionally, on Tegra devices, [CU_CTX_SCHED_AUTO](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4d75f4c24f8c35ef2ee9d0793badfd88c>) uses a heuristic based on the power profile of the platform and may choose [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>) for low-powered devices.


  * [CU_CTX_LMEM_RESIZE_TO_MAX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4b5a83507c2a7e14d301621c40c343a81>): Instruct CUDA to not reduce local memory after resizing local memory for a kernel. This can prevent thrashing by local memory allocations when launching many kernels with high local memory usage at the cost of potentially increased memory usage.

**Deprecated:** This flag is deprecated and the behavior enabled by this flag is now the default and cannot be disabled.


  * [CU_CTX_COREDUMP_ENABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44c74aa37941a780fccfbc2aa23e97809>): If GPU coredumps have not been enabled globally with [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") or environment variables, this flag can be set during context creation to instruct CUDA to create a coredump if this context raises an exception during execution. These environment variables are described in the CUDA-GDB user guide under the "GPU core dump support" section. The initial settings will be taken from the global settings at the time of context creation. The other settings that control coredump output can be modified by calling [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context.") from the created context after it becomes current.


  * [CU_CTX_USER_COREDUMP_ENABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd40703ba07a50ffbd294cb1122e08370d5>): If user-triggered GPU coredumps have not been enabled globally with [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") or environment variables, this flag can be set during context creation to instruct CUDA to create a coredump if data is written to a certain pipe that is present in the OS space. These environment variables are described in the CUDA-GDB user guide under the "GPU core dump support" section. It is important to note that the pipe name *must* be set with [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") before creating the context if this flag is used. Setting this flag implies that [CU_CTX_COREDUMP_ENABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44c74aa37941a780fccfbc2aa23e97809>) is set. The initial settings will be taken from the global settings at the time of context creation. The other settings that control coredump output can be modified by calling [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context.") from the created context after it becomes current.


  * [CU_CTX_SYNC_MEMOPS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4e93e3bda3a8e71fe8f4d0de36aa881f0>): Ensures that synchronous memory operations initiated on this context will always synchronize. See further documentation in the section titled "API Synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDevicePrimaryCtxRetain](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300> "Retain the primary context on the GPU."), [cuDevicePrimaryCtxGetState](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g65f3e018721b6d90aa05cfb56250f469> "Get the state of the primary context."), [cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxSetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1g66655c37602c8628eae3e40c82619f1e> "Sets the flags for the current context."), [cudaSetDeviceFlags](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g69e73c7dda3fc05306ae7c811a690fac>)

* * *
