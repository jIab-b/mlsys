# Context Management (Deprecated)

## 6.9.Â Context Management [DEPRECATED]

This section describes the deprecated context management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxAttach](<#group__CUDA__CTX__DEPRECATED_1g3c9b7c5833d57e7ccea5aeaba6009f5d>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx, unsigned int Â flags )
     Increment a context's usage-count.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxDetach](<#group__CUDA__CTX__DEPRECATED_1g2da7d6b2651b46896871a068e2860551>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Decrement a context's usage-count.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetSharedMemConfig](<#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194>) ( [CUsharedconfig](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g92d66e95f602cb9fdaf0682c260c241b>)*Â pConfig )
     Returns the current shared memory configuration for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSetSharedMemConfig](<#group__CUDA__CTX__DEPRECATED_1gb1fef6f9fd5c252245214f85ae01ec23>) ( [CUsharedconfig](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g92d66e95f602cb9fdaf0682c260c241b>)Â config )
     Sets the shared memory configuration for the current context.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxAttach ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx, unsigned int Â flags )


Increment a context's usage-count.

######  Parameters

`pctx`
    \- Returned context handle of the current context
`flags`
    \- Context attach flags (must be 0)

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000004>)

Note that this function is deprecated and should not be used.

###### Description

Increments the usage count of the context and passes back a context handle in `*pctx` that must be passed to [cuCtxDetach()](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1g2da7d6b2651b46896871a068e2860551> "Decrement a context's usage-count.") when the application is done with the context. [cuCtxAttach()](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1g3c9b7c5833d57e7ccea5aeaba6009f5d> "Increment a context's usage-count.") fails if there is no context current to the thread.

Currently, the `flags` parameter must be 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxDetach](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1g2da7d6b2651b46896871a068e2860551> "Decrement a context's usage-count."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxDetach ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Decrement a context's usage-count.

######  Parameters

`ctx`
    \- Context to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000005>)

Note that this function is deprecated and should not be used.

###### Description

Decrements the usage count of the context `ctx`, and destroys the context if the usage count goes to 0. The context must be a handle that was passed back by [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") or [cuCtxAttach()](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1g3c9b7c5833d57e7ccea5aeaba6009f5d> "Increment a context's usage-count."), and must be current to the calling thread.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetSharedMemConfig ( [CUsharedconfig](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g92d66e95f602cb9fdaf0682c260c241b>)*Â pConfig )


Returns the current shared memory configuration for the current context.

######  Parameters

`pConfig`
    \- returned shared memory configuration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000006>)

###### Description

This function will return in `pConfig` the current size of shared memory banks in the current context. On devices with configurable shared memory banks, [cuCtxSetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gb1fef6f9fd5c252245214f85ae01ec23> "Sets the shared memory configuration for the current context.") can be used to change this setting, so that all subsequent kernel launches will by default use the new bank size. When [cuCtxGetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194> "Returns the current shared memory configuration for the current context.") is called on devices without configurable shared memory, it will return the fixed bank size of the hardware.

The returned bank configurations can be either:

  * [CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241b18d5d945c971d5d288d2693cbaa4d7dc>): shared memory bank width is four bytes.

  * [CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241b081c400b814b9832b8a934ad2934985c>): shared memory bank width will eight bytes.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cuCtxGetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194> "Returns the current shared memory configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cudaDeviceGetSharedMemConfig](<../cuda-runtime-api/group__CUDART__DEVICE__DEPRECATED.html#group__CUDART__DEVICE__DEPRECATED_1g542246258996a39a3ce2bc311bbb2421>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSetSharedMemConfig ( [CUsharedconfig](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g92d66e95f602cb9fdaf0682c260c241b>)Â config )


Sets the shared memory configuration for the current context.

######  Parameters

`config`
    \- requested shared memory configuration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000007>)

###### Description

On devices with configurable shared memory banks, this function will set the context's shared memory bank size which is used for subsequent kernel launches.

Changed the shared memory configuration between launches may insert a device side synchronization point between those launches.

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

The supported bank configurations are:

  * [CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241bd65d166d885bd3f41bf1ced4ab8e044e>): set bank width to the default initial setting (currently, four bytes).

  * [CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241b18d5d945c971d5d288d2693cbaa4d7dc>): set shared memory bank width to be natively four bytes.

  * [CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241b081c400b814b9832b8a934ad2934985c>): set shared memory bank width to be natively eight bytes.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cuCtxGetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194> "Returns the current shared memory configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cudaDeviceSetSharedMemConfig](<../cuda-runtime-api/group__CUDART__DEVICE__DEPRECATED.html#group__CUDART__DEVICE__DEPRECATED_1g76cb4f94c7af96c1247dfc7f105eabae>)

* * *
