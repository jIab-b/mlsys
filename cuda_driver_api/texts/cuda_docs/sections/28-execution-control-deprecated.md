# Execution Control (Deprecated)

## 6.23.Â Execution Control [DEPRECATED]

This section describes the deprecated execution control functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncSetBlockShape](<#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â x, int Â y, int Â z )
     Sets the block-dimensions for the function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncSetSharedMemConfig](<#group__CUDA__EXEC__DEPRECATED_1g3fe2417a78a7b5554a694c40355b54ce>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, [CUsharedconfig](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g92d66e95f602cb9fdaf0682c260c241b>)Â config )
     Sets the shared memory configuration for a device function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncSetSharedSize](<#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, unsigned int Â bytes )
     Sets the dynamic shared-memory size for the function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunch](<#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f )
     Launches a CUDA function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchGrid](<#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, int Â grid_width, int Â grid_height )
     Launches a CUDA function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchGridAsync](<#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, int Â grid_width, int Â grid_height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Launches a CUDA function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuParamSetSize](<#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, unsigned int Â numbytes )
     Sets the parameter size for the function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuParamSetTexRef](<#group__CUDA__EXEC__DEPRECATED_1g10fad4a11f4f6d0422f4929ff348fce5>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â texunit, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Adds a texture-reference to the function's argument list.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuParamSetf](<#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â offset, float Â value )
     Adds a floating-point parameter to the function's argument list.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuParamSeti](<#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â offset, unsigned int Â value )
     Adds an integer parameter to the function's argument list.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuParamSetv](<#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â offset, void*Â ptr, unsigned int Â numbytes )
     Adds arbitrary data to the function's argument list.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncSetBlockShape ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â x, int Â y, int Â z )


Sets the block-dimensions for the function.

######  Parameters

`hfunc`
    \- Kernel to specify dimensions of
`x`
    \- X dimension
`y`
    \- Y dimension
`z`
    \- Z dimension

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000011>)

###### Description

Specifies the `x`, `y`, and `z` dimensions of the thread blocks that are created when the kernel given by `hfunc` is launched.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncSetSharedMemConfig ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, [CUsharedconfig](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g92d66e95f602cb9fdaf0682c260c241b>)Â config )


Sets the shared memory configuration for a device function.

######  Parameters

`hfunc`
    \- kernel to be given a shared memory config
`config`
    \- requested shared memory configuration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000021>)

###### Description

On devices with configurable shared memory banks, this function will force all subsequent launches of the specified device function to have the given shared memory bank size configuration. On any given launch of the function, the shared memory configuration of the device will be temporarily changed if needed to suit the function's preferred configuration. Changes in shared memory configuration between subsequent launches of functions, may introduce a device side synchronization point.

Any per-function setting of shared memory bank size set via [cuFuncSetSharedMemConfig](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g3fe2417a78a7b5554a694c40355b54ce> "Sets the shared memory configuration for a device function.") will override the context wide setting set with [cuCtxSetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gb1fef6f9fd5c252245214f85ae01ec23> "Sets the shared memory configuration for the current context.").

Changing the shared memory bank size will not increase shared memory usage or affect occupancy of kernels, but may have major effects on performance. Larger bank sizes will allow for greater potential bandwidth to shared memory, but will change what kinds of accesses to shared memory will result in bank conflicts.

This function will do nothing on devices with fixed shared memory bank size.

The supported bank configurations are:

  * [CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241bd65d166d885bd3f41bf1ced4ab8e044e>): use the context's shared memory configuration when launching this function.

  * [CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241b18d5d945c971d5d288d2693cbaa4d7dc>): set shared memory bank width to be natively four bytes when launching this function.

  * [CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg92d66e95f602cb9fdaf0682c260c241b081c400b814b9832b8a934ad2934985c>): set shared memory bank width to be natively eight bytes when launching this function.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxGetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194> "Returns the current shared memory configuration for the current context."), [cuCtxSetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gb1fef6f9fd5c252245214f85ae01ec23> "Sets the shared memory configuration for the current context."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cudaFuncSetSharedMemConfig](<../cuda-runtime-api/group__CUDART__EXECUTION__DEPRECATED.html#group__CUDART__EXECUTION__DEPRECATED_1gbd189716def6fdb5f819dae77452d30b>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncSetSharedSize ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, unsigned int Â bytes )


Sets the dynamic shared-memory size for the function.

######  Parameters

`hfunc`
    \- Kernel to specify dynamic shared-memory size for
`bytes`
    \- Dynamic shared-memory size per thread in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000012>)

###### Description

Sets through `bytes` the amount of dynamic shared memory that will be available to each thread block when the kernel given by `hfunc` is launched.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunch ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f )


Launches a CUDA function.

######  Parameters

`f`
    \- Kernel to launch

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000017>)

###### Description

Invokes the kernel `f` on a 1 x 1 x 1 grid of blocks. The block contains the number of threads specified by a previous call to [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function.").

The block shape, dynamic shared memory size, and parameter information must be set using [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuParamSetSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetf()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), and [cuParamSetv()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list.") prior to calling this function.

Launching a function via [cuLaunchKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") invalidates the function's block shape, dynamic shared memory size, and parameter information. After launching via cuLaunchKernel, this state must be re-initialized prior to calling this function. Failure to do so results in undefined behavior.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchGrid ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, int Â grid_width, int Â grid_height )


Launches a CUDA function.

######  Parameters

`f`
    \- Kernel to launch
`grid_width`
    \- Width of grid in blocks
`grid_height`
    \- Height of grid in blocks

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000018>)

###### Description

Invokes the kernel `f` on a `grid_width` x `grid_height` grid of blocks. Each block contains the number of threads specified by a previous call to [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function.").

The block shape, dynamic shared memory size, and parameter information must be set using [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuParamSetSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetf()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), and [cuParamSetv()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list.") prior to calling this function.

Launching a function via [cuLaunchKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") invalidates the function's block shape, dynamic shared memory size, and parameter information. After launching via cuLaunchKernel, this state must be re-initialized prior to calling this function. Failure to do so results in undefined behavior.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchGridAsync ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, int Â grid_width, int Â grid_height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Launches a CUDA function.

######  Parameters

`f`
    \- Kernel to launch
`grid_width`
    \- Width of grid in blocks
`grid_height`
    \- Height of grid in blocks
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000019>)

###### Description

Invokes the kernel `f` on a `grid_width` x `grid_height` grid of blocks. Each block contains the number of threads specified by a previous call to [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function.").

The block shape, dynamic shared memory size, and parameter information must be set using [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuParamSetSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetf()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), and [cuParamSetv()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list.") prior to calling this function.

Launching a function via [cuLaunchKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") invalidates the function's block shape, dynamic shared memory size, and parameter information. After launching via cuLaunchKernel, this state must be re-initialized prior to calling this function. Failure to do so results in undefined behavior.

Note:

  * In certain cases where cubins are created with no ABI (i.e., using `ptxas``--abi-compile``no`), this function may serialize kernel launches. The CUDA driver retains asynchronous behavior by growing the per-thread stack as needed per launch and not shrinking it afterwards.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuParamSetSize ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, unsigned int Â numbytes )


Sets the parameter size for the function.

######  Parameters

`hfunc`
    \- Kernel to set parameter size for
`numbytes`
    \- Size of parameter list in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000013>)

###### Description

Sets through `numbytes` the total size in bytes needed by the function parameters of the kernel corresponding to `hfunc`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuParamSetTexRef ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â texunit, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Adds a texture-reference to the function's argument list.

######  Parameters

`hfunc`
    \- Kernel to add texture-reference to
`texunit`
    \- Texture unit (must be [CU_PARAM_TR_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3e9be6955a6a5c311ad5ea2debdd6613>))
`hTexRef`
    \- Texture-reference to add to argument list

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000020>)

###### Description

Makes the CUDA array or linear memory bound to the texture reference `hTexRef` available to a device program as a texture. In this version of CUDA, the texture-reference must be obtained via [cuModuleGetTexRef()](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference.") and the `texunit` parameter must be set to [CU_PARAM_TR_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3e9be6955a6a5c311ad5ea2debdd6613>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuParamSetf ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â offset, float Â value )


Adds a floating-point parameter to the function's argument list.

######  Parameters

`hfunc`
    \- Kernel to add parameter to
`offset`
    \- Offset to add parameter to argument list
`value`
    \- Value of parameter

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000015>)

###### Description

Sets a floating-point parameter that will be specified the next time the kernel corresponding to `hfunc` will be invoked. `offset` is a byte offset.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuParamSeti ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â offset, unsigned int Â value )


Adds an integer parameter to the function's argument list.

######  Parameters

`hfunc`
    \- Kernel to add parameter to
`offset`
    \- Offset to add parameter to argument list
`value`
    \- Value of parameter

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000014>)

###### Description

Sets an integer parameter that will be specified the next time the kernel corresponding to `hfunc` will be invoked. `offset` is a byte offset.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSetv](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuParamSetv ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, int Â offset, void*Â ptr, unsigned int Â numbytes )


Adds arbitrary data to the function's argument list.

######  Parameters

`hfunc`
    \- Kernel to add data to
`offset`
    \- Offset to add data to argument list
`ptr`
    \- Pointer to arbitrary data
`numbytes`
    \- Size of data to copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000016>)

###### Description

Copies an arbitrary amount of data (specified in `numbytes`) from `ptr` into the parameter space of the kernel corresponding to `hfunc`. `offset` is a byte offset.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncSetBlockShape](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuParamSetSize](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSetf](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSeti](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuLaunch](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gea7bd80835bcce59c73247120766f6ff> "Launches a CUDA function."), [cuLaunchGrid](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g0676b0afb5d5c63aa46801788e3d8ca5> "Launches a CUDA function."), [cuLaunchGridAsync](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gb0292382e6b0d059263acd2574aaf00b> "Launches a CUDA function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

* * *
