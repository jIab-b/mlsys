# Execution Control

## 6.22.Â Execution Control

This section describes the execution control functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncGetAttribute](<#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b>) ( int*Â pi, [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc )
     Returns information about a function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncGetModule](<#group__CUDA__EXEC_1g58f0fd1db9dadd3870440662622a27ef>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â hmod, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc )
     Returns a module handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncGetName](<#group__CUDA__EXEC_1gf60c6c51203cab164c07d6ddcc2b2e26>) ( const char**Â name, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc )
     Returns the function name for a CUfunction handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncGetParamInfo](<#group__CUDA__EXEC_1g6874b82bcf2803902085645e46e0ca0e>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, size_tÂ paramIndex, size_t*Â paramOffset, size_t*Â paramSize )
     Returns the offset and size of a kernel parameter in the device-side parameter layout.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncIsLoaded](<#group__CUDA__EXEC_1gfc6fed4bbe6c35e0445a49396774aa96>) ( CUfunctionLoadingState*Â state, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â function )
     Returns if the function is loaded.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncLoad](<#group__CUDA__EXEC_1g3b67024e8875bfd155534785708093ab>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â function )
     Loads a function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncSetAttribute](<#group__CUDA__EXEC_1g0e37dce0173bc883aa1e5b14dd747f26>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, int Â value )
     Sets information about a function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFuncSetCacheConfig](<#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)Â config )
     Sets the preferred cache configuration for a device function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchCooperativeKernel](<#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, unsigned int Â gridDimX, unsigned int Â gridDimY, unsigned int Â gridDimZ, unsigned int Â blockDimX, unsigned int Â blockDimY, unsigned int Â blockDimZ, unsigned int Â sharedMemBytes, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, void**Â kernelParams )
     Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchCooperativeKernelMultiDevice](<#group__CUDA__EXEC_1g1d34025bc4f8fcec82fbcfc18d07a6e2>) ( [CUDA_LAUNCH_PARAMS](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1>)*Â launchParamsList, unsigned int Â numDevices, unsigned int Â flags )
     Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchHostFunc](<#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUhostFn](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g262cd3570ff5d396db4e3dabede3c355>)Â fn, void*Â userData )
     Enqueues a host function call in a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchKernel](<#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, unsigned int Â gridDimX, unsigned int Â gridDimY, unsigned int Â gridDimZ, unsigned int Â blockDimX, unsigned int Â blockDimY, unsigned int Â blockDimZ, unsigned int Â sharedMemBytes, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, void**Â kernelParams, void**Â extra )
     Launches a CUDA function CUfunction or a CUDA kernel CUkernel.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLaunchKernelEx](<#group__CUDA__EXEC_1gb9c891eb6bb8f4089758e64c9c976db9>) ( const [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>)*Â config, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, void**Â kernelParams, void**Â extra )
     Launches a CUDA function CUfunction or a CUDA kernel CUkernel with launch-time configuration.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncGetAttribute ( int*Â pi, [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc )


Returns information about a function.

######  Parameters

`pi`
    \- Returned attribute value
`attrib`
    \- Attribute requested
`hfunc`
    \- Function to query attribute of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_FUNCTION_NOT_LOADED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9356546a00d65f2f6bf9fc65311edabdf>)

###### Description

Returns in `*pi` the integer value of the attribute `attrib` on the kernel given by `hfunc`. The supported attributes are:

  * [CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bca244b9a52d7426e6684acebf4c9e24b8>): The maximum number of threads per block, beyond which a launch of the function would fail. This number depends on both the function and the device on which the function is currently loaded.

  * [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc15651a634781263c9d4ee6070a3991f4>): The size in bytes of statically-allocated shared memory per block required by this function. This does not include dynamically-allocated shared memory requested by the user at runtime.

  * [CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4a255dc4e2b8542e84c9431c1953a952>): The size in bytes of user-allocated constant memory required by this function.

  * [CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc0318e60c17eb22c70ffb59f610c504dd>): The size in bytes of local memory used by each thread of this function.

  * [CU_FUNC_ATTRIBUTE_NUM_REGS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc217730c04b1edbc80bb1772c1d6a7752>): The number of registers used by each thread of this function.

  * [CU_FUNC_ATTRIBUTE_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bccbd28200668ad2de39035446a89bf930>): The PTX virtual architecture version for which the function was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.

  * [CU_FUNC_ATTRIBUTE_BINARY_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcc4f70f5d16889d0b75c3bf7a303eb437>): The binary architecture version for which the function was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.

  * CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the function has been compiled with user specified option "-Xptxas \--dlcm=ca" set .

  * [CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc75b33d145e83462ef7292575015be03e>): The maximum size in bytes of dynamically-allocated shared memory.

  * [CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcc75f6fd470b848653f026b8c82c10ae3>): Preferred shared memory-L1 cache split ratio in percent of total shared memory.

  * [CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc8bd0417b504a8006cc6f57c023b54c2b>): If this attribute is set, the kernel must launch with a valid cluster size specified.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc1c83b56a254f78ddd5bf75ccfd15f0cb>): The required cluster width in blocks.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc0c3f2eb7eaea02e3c85a4bedd02be331>): The required cluster height in blocks.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4ab3672ad6476ad4bfa973e3083cdb32>): The required cluster depth in blocks.

  * [CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bce5ea0cdab1af87e68ac45e19e4c52c5d>): Indicates whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed. A non-portable cluster size may only function on the specific SKUs the program is tested on. The launch might fail if the program is run on a different hardware platform. CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking whether the desired size can be launched on the current device. A portable cluster size is guaranteed to be functional on all compute capabilities higher than the target compute capability. The portable cluster size for sm_90 is 8 blocks per cluster. This value may increase for future compute capabilities. The specific hardware unit may support higher cluster sizes thatâs not guaranteed to be portable.

  * [CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcad30df41ca0cac5046c58a75d91326a6>): The block scheduling policy of a function. The value type is CUclusterSchedulingPolicy.


With a few execeptions, function attributes may also be queried on unloaded function handles returned from [cuModuleEnumerateFunctions](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g6bdb22a7d9cacf7df5bda2a18082ec50> "Returns the function handles within a module."). [CUDA_ERROR_FUNCTION_NOT_LOADED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9356546a00d65f2f6bf9fc65311edabdf>) is returned if the attribute requires a fully loaded function but the function is not loaded. The loading state of a function may be queried using cuFuncIsloaded. [cuFuncLoad](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g3b67024e8875bfd155534785708093ab> "Loads a function.") may be called to explicitly load a function before querying the following attributes that require the function to be loaded:

  * [CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bca244b9a52d7426e6684acebf4c9e24b8>)

  * [CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4a255dc4e2b8542e84c9431c1953a952>)

  * [CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc75b33d145e83462ef7292575015be03e>)


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cudaFuncGetAttributes](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0e78e02c6d12ebddd4577ac6ebadf494>), [cudaFuncSetAttribute](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g61085e9f04656b92573af16072bbc78d>), [cuFuncIsLoaded](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gfc6fed4bbe6c35e0445a49396774aa96> "Returns if the function is loaded."), [cuFuncLoad](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g3b67024e8875bfd155534785708093ab> "Loads a function."), [cuKernelGetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f> "Returns information about a kernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncGetModule ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â hmod, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc )


Returns a module handle.

######  Parameters

`hmod`
    \- Returned module handle
`hfunc`
    \- Function to retrieve module for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `*hmod` the handle of the module that function `hfunc` is located in. The lifetime of the module corresponds to the lifetime of the context it was loaded in or until the module is explicitly unloaded.

The CUDA runtime manages its own modules loaded into the primary context. If the handle returned by this API refers to a module loaded by the CUDA runtime, calling [cuModuleUnload()](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.") on that module will result in undefined behavior.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncGetName ( const char**Â name, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc )


Returns the function name for a CUfunction handle.

######  Parameters

`name`
    \- The returned name of the function
`hfunc`
    \- The function handle to retrieve the name for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns in `**name` the function name associated with the function handle `hfunc` . The function name is returned as a null-terminated string. The returned name is only valid when the function handle is valid. If the module is unloaded or reloaded, one must call the API again to get the updated name. This API may return a mangled name if the function is not declared as having C linkage. If either `**name` or `hfunc` is NULL, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncGetParamInfo ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, size_tÂ paramIndex, size_t*Â paramOffset, size_t*Â paramSize )


Returns the offset and size of a kernel parameter in the device-side parameter layout.

######  Parameters

`func`
    \- The function to query
`paramIndex`
    \- The parameter index to query
`paramOffset`
    \- Returns the offset into the device-side parameter layout at which the parameter resides
`paramSize`
    \- Optionally returns the size of the parameter in the device-side parameter layout

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Queries the kernel parameter at `paramIndex` into `func's` list of parameters, and returns in `paramOffset` and `paramSize` the offset and size, respectively, where the parameter will reside in the device-side parameter layout. This information can be used to update kernel node parameters from the device via [cudaGraphKernelNodeSetParam()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g0c2bd161eff1e47531eedce282e66d21>) and [cudaGraphKernelNodeUpdatesApply()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g2d558cf37c9616365c67447e61ac0d6a>). `paramIndex` must be less than the number of parameters that `func` takes. `paramSize` can be set to NULL if only the parameter offset is desired.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuKernelGetParamInfo](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ga61653c9f13f713527e189fb0c2fe235> "Returns the offset and size of a kernel parameter in the device-side parameter layout.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncIsLoaded ( CUfunctionLoadingState*Â state, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â function )


Returns if the function is loaded.

######  Parameters

`state`
    \- returned loading state
`function`
    \- the function to check

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `state` the loading state of `function`.

**See also:**

[cuFuncLoad](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g3b67024e8875bfd155534785708093ab> "Loads a function."), [cuModuleEnumerateFunctions](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g6bdb22a7d9cacf7df5bda2a18082ec50> "Returns the function handles within a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncLoad ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â function )


Loads a function.

######  Parameters

`function`
    \- the function to load

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Finalizes function loading for `function`. Calling this API with a fully loaded function has no effect.

**See also:**

[cuModuleEnumerateFunctions](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g6bdb22a7d9cacf7df5bda2a18082ec50> "Returns the function handles within a module."), [cuFuncIsLoaded](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gfc6fed4bbe6c35e0445a49396774aa96> "Returns if the function is loaded.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncSetAttribute ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, int Â value )


Sets information about a function.

######  Parameters

`hfunc`
    \- Function to query attribute of
`attrib`
    \- Attribute requested
`value`
    \- The value to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

This call sets the value of a specified attribute `attrib` on the kernel given by `hfunc` to an integer value specified by `val` This function returns CUDA_SUCCESS if the new value of the attribute could be successfully set. If the set fails, this call will return an error. Not all attributes can have values set. Attempting to set a value on a read-only attribute will result in an error (CUDA_ERROR_INVALID_VALUE)

Supported attributes for the cuFuncSetAttribute call are:

  * [CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc75b33d145e83462ef7292575015be03e>): This maximum size in bytes of dynamically-allocated shared memory. The value should contain the requested maximum size of dynamically-allocated shared memory. The sum of this value and the function attribute [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc15651a634781263c9d4ee6070a3991f4>) cannot exceed the device attribute [CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3e788564c0a95b866dc624fbc1b49dab3>). The maximal size of requestable dynamic shared memory may differ by GPU architecture.

  * [CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcc75f6fd470b848653f026b8c82c10ae3>): On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. See [CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a306a33c18889f6fc907412451c95154ed>) This is only a hint, and the driver can choose a different ratio if required to execute the function.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc1c83b56a254f78ddd5bf75ccfd15f0cb>): The required cluster width in blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc0c3f2eb7eaea02e3c85a4bedd02be331>): The required cluster height in blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4ab3672ad6476ad4bfa973e3083cdb32>): The required cluster depth in blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.

  * [CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bce5ea0cdab1af87e68ac45e19e4c52c5d>): Indicates whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed.

  * [CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcad30df41ca0cac5046c58a75d91326a6>): The block scheduling policy of a function. The value type is CUclusterSchedulingPolicy.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cudaFuncGetAttributes](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0e78e02c6d12ebddd4577ac6ebadf494>), [cudaFuncSetAttribute](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g61085e9f04656b92573af16072bbc78d>), [cuKernelSetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067> "Sets information about a kernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFuncSetCacheConfig ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â hfunc, [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)Â config )


Sets the preferred cache configuration for a device function.

######  Parameters

`hfunc`
    \- Kernel to configure cache for
`config`
    \- Requested cache configuration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through `config` the preferred cache configuration for the device function `hfunc`. This is only a preference. The driver will use the requested configuration if possible, but it is free to choose a different configuration if required to execute `hfunc`. Any context-wide preference set via [cuCtxSetCacheConfig()](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context.") will be overridden by this per-function setting unless the per-function setting is [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>). In that case, the current context-wide setting will be used.

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.

The supported cache configurations are:

  * [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>): no preference for shared memory or L1 (default)

  * [CU_FUNC_CACHE_PREFER_SHARED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec712f43defb051d7985317bce426cccc8>): prefer larger shared memory and smaller L1 cache

  * [CU_FUNC_CACHE_PREFER_L1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ecb1e6c4e889e1a70ed5283172be08f6a5>): prefer larger L1 cache and smaller shared memory

  * [CU_FUNC_CACHE_PREFER_EQUAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec4434321280821d844a15b02e4d6c80a9>): prefer equal sized L1 cache and shared memory


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cudaFuncSetCacheConfig](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7d9cc996fe45b6260ebb086caff1c685>), [cuKernelSetCacheConfig](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51> "Sets the preferred cache configuration for a device kernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchCooperativeKernel ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, unsigned int Â gridDimX, unsigned int Â gridDimY, unsigned int Â gridDimZ, unsigned int Â blockDimX, unsigned int Â blockDimY, unsigned int Â blockDimZ, unsigned int Â sharedMemBytes, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, void**Â kernelParams )


Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute.

######  Parameters

`f`
    \- Function [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) or Kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) to launch
`gridDimX`
    \- Width of grid in blocks
`gridDimY`
    \- Height of grid in blocks
`gridDimZ`
    \- Depth of grid in blocks
`blockDimX`
    \- X dimension of each thread block
`blockDimY`
    \- Y dimension of each thread block
`blockDimZ`
    \- Z dimension of each thread block
`sharedMemBytes`
    \- Dynamic shared-memory size per thread block in bytes
`hStream`
    \- Stream identifier
`kernelParams`
    \- Array of pointers to kernel parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d26f67e0acc1563f87ddb94c638478cd>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Invokes the function [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) or the kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)`f` on a `gridDimX` x `gridDimY` x `gridDimZ` grid of blocks. Each block contains `blockDimX` x `blockDimY` x `blockDimZ` threads.

`sharedMemBytes` sets the amount of dynamic shared memory that will be available to each thread block.

The device on which this kernel is invoked must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3e9d3b7631f9112048c541cdb08c8a4e6>).

The total number of blocks launched cannot exceed the maximum number of blocks per multiprocessor as returned by [cuOccupancyMaxActiveBlocksPerMultiprocessor](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98> "Returns occupancy of a function.") (or [cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g8f1da4d4983e5c3025447665423ae2c2> "Returns occupancy of a function.")) times the number of multiprocessors as specified by the device attribute [CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3b92d0a38a94a3f61d4c53e00114afcaa>).

The kernel cannot make use of CUDA dynamic parallelism.

Kernel parameters must be specified via `kernelParams`. If `f` has N parameters, then `kernelParams` needs to be an array of N pointers. Each of `kernelParams`[0] through `kernelParams`[N-1] must point to a region of memory from which the actual kernel parameter will be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that information is retrieved directly from the kernel's image.

Calling [cuLaunchCooperativeKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute.") sets persistent function state that is the same as function state set through [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") API

When the kernel `f` is launched via [cuLaunchCooperativeKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute."), the previous block shape, shared size and parameter info associated with `f` is overwritten.

Note that to use [cuLaunchCooperativeKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute."), the kernel `f` must either have been compiled with toolchain version 3.2 or later so that it will contain kernel parameter information, or have no kernel parameters. If either of these conditions is not met, then [cuLaunchCooperativeKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute.") will return [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>).

Note that the API can also be used to launch context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to launch the kernel on will either be taken from the specified stream `hStream` or the current context in case of NULL stream.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuLaunchCooperativeKernelMultiDevice](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g1d34025bc4f8fcec82fbcfc18d07a6e2> "Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute."), [cudaLaunchCooperativeKernel](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g7c4cb6c44a6c4608da36c44374499b31>), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuKernelSetCacheConfig](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51> "Sets the preferred cache configuration for a device kernel."), [cuKernelGetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f> "Returns information about a kernel."), [cuKernelSetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067> "Sets information about a kernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchCooperativeKernelMultiDevice ( [CUDA_LAUNCH_PARAMS](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1>)*Â launchParamsList, unsigned int Â numDevices, unsigned int Â flags )


Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute.

######  Parameters

`launchParamsList`
    \- List of launch parameters, one per device
`numDevices`
    \- Size of the `launchParamsList` array
`flags`
    \- Flags to control launch behavior

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d26f67e0acc1563f87ddb94c638478cd>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000010>)

This function is deprecated as of CUDA 11.3.

###### Description

Invokes kernels as specified in the `launchParamsList` array where each element of the array specifies all the parameters required to perform a single kernel launch. These kernels can cooperate and synchronize as they execute. The size of the array is specified by `numDevices`.

No two kernels can be launched on the same device. All the devices targeted by this multi-device launch must be identical. All devices must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a31d10266c0f67ba1fe57b23c9d311f0a3>).

All kernels launched must be identical with respect to the compiled code. Note that any __device__, __constant__ or __managed__ variables present in the module that owns the kernel launched on each device, are independently instantiated on every device. It is the application's responsibility to ensure these variables are initialized and used appropriately.

The size of the grids as specified in blocks, the size of the blocks themselves and the amount of shared memory used by each thread block must also match across all launched kernels.

The streams used to launch these kernels must have been created via either [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream.") or [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."). The NULL stream or [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) or [CU_STREAM_PER_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c>) cannot be used.

The total number of blocks launched per kernel cannot exceed the maximum number of blocks per multiprocessor as returned by [cuOccupancyMaxActiveBlocksPerMultiprocessor](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98> "Returns occupancy of a function.") (or [cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g8f1da4d4983e5c3025447665423ae2c2> "Returns occupancy of a function.")) times the number of multiprocessors as specified by the device attribute [CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3b92d0a38a94a3f61d4c53e00114afcaa>). Since the total number of blocks launched per device has to match across all devices, the maximum number of blocks that can be launched per device will be limited by the device with the least number of multiprocessors.

The kernels cannot make use of CUDA dynamic parallelism.

The CUDA_LAUNCH_PARAMS structure is defined as:


    â        typedef struct CUDA_LAUNCH_PARAMS_st
                  {
                      [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) function;
                      unsigned int gridDimX;
                      unsigned int gridDimY;
                      unsigned int gridDimZ;
                      unsigned int blockDimX;
                      unsigned int blockDimY;
                      unsigned int blockDimZ;
                      unsigned int sharedMemBytes;
                      [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>) hStream;
                      void **kernelParams;
                  } [CUDA_LAUNCH_PARAMS](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1>);

where:

  * [CUDA_LAUNCH_PARAMS::function](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1ad25005328ba7debf0f5bd2d39a5363c>) specifies the kernel to be launched. All functions must be identical with respect to the compiled code. Note that you can also specify context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). In this case, the context to launch the kernel on be taken from the specified stream [CUDA_LAUNCH_PARAMS::hStream](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1d7b7a742ef397fe918c18bf1f5e63576>).

  * [CUDA_LAUNCH_PARAMS::gridDimX](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_14328c3a1123bed3c08894d66ae9f0e8f>) is the width of the grid in blocks. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::gridDimY](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_175e5d98e55ad9c877acf3511f9e4c6bc>) is the height of the grid in blocks. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::gridDimZ](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_16e34c343c3dc355e93f3c47d5c3e3fbe>) is the depth of the grid in blocks. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::blockDimX](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1a31d64be7210f4f21404e89ed3c8bc09>) is the X dimension of each thread block. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::blockDimX](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1a31d64be7210f4f21404e89ed3c8bc09>) is the Y dimension of each thread block. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::blockDimZ](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1e35cc15ae81f10e20fc4c91dbc7356ea>) is the Z dimension of each thread block. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::sharedMemBytes](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_113811a872234e77c9903dd977f7c7ac3>) is the dynamic shared-memory size per thread block in bytes. This must match across all kernels launched.

  * [CUDA_LAUNCH_PARAMS::hStream](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1d7b7a742ef397fe918c18bf1f5e63576>) is the handle to the stream to perform the launch in. This cannot be the NULL stream or [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) or [CU_STREAM_PER_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c>). The CUDA context associated with this stream must match that associated with [CUDA_LAUNCH_PARAMS::function](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1ad25005328ba7debf0f5bd2d39a5363c>).

  * [CUDA_LAUNCH_PARAMS::kernelParams](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_16788671fda12351d95398142aa4c3bcd>) is an array of pointers to kernel parameters. If [CUDA_LAUNCH_PARAMS::function](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1ad25005328ba7debf0f5bd2d39a5363c>) has N parameters, then [CUDA_LAUNCH_PARAMS::kernelParams](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_16788671fda12351d95398142aa4c3bcd>) needs to be an array of N pointers. Each of [CUDA_LAUNCH_PARAMS::kernelParams](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_16788671fda12351d95398142aa4c3bcd>)[0] through [CUDA_LAUNCH_PARAMS::kernelParams](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_16788671fda12351d95398142aa4c3bcd>)[N-1] must point to a region of memory from which the actual kernel parameter will be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that information is retrieved directly from the kernel's image.


By default, the kernel won't begin execution on any GPU until all prior work in all the specified streams has completed. This behavior can be overridden by specifying the flag [CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g14fbb447864d154bdd6d82a7af51c5ab>). When this flag is specified, each kernel will only wait for prior work in the stream corresponding to that GPU to complete before it begins execution.

Similarly, by default, any subsequent work pushed in any of the specified streams will not begin execution until the kernels on all GPUs have completed. This behavior can be overridden by specifying the flag [CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb46080444b8784713763d4d0dc4e1c90>). When this flag is specified, any subsequent work pushed in any of the specified streams will only wait for the kernel launched on the GPU corresponding to that stream to complete before it begins execution.

Calling [cuLaunchCooperativeKernelMultiDevice()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g1d34025bc4f8fcec82fbcfc18d07a6e2> "Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute.") sets persistent function state that is the same as function state set through [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") API when called individually for each element in `launchParamsList`.

When kernels are launched via [cuLaunchCooperativeKernelMultiDevice()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g1d34025bc4f8fcec82fbcfc18d07a6e2> "Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute."), the previous block shape, shared size and parameter info associated with each [CUDA_LAUNCH_PARAMS::function](<structCUDA__LAUNCH__PARAMS__v1.html#structCUDA__LAUNCH__PARAMS__v1_1ad25005328ba7debf0f5bd2d39a5363c>) in `launchParamsList` is overwritten.

Note that to use [cuLaunchCooperativeKernelMultiDevice()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g1d34025bc4f8fcec82fbcfc18d07a6e2> "Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute."), the kernels must either have been compiled with toolchain version 3.2 or later so that it will contain kernel parameter information, or have no kernel parameters. If either of these conditions is not met, then [cuLaunchCooperativeKernelMultiDevice()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g1d34025bc4f8fcec82fbcfc18d07a6e2> "Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute.") will return [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>).

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cuLaunchCooperativeKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute."), cudaLaunchCooperativeKernelMultiDevice

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchHostFunc ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUhostFn](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g262cd3570ff5d396db4e3dabede3c355>)Â fn, void*Â userData )


Enqueues a host function call in a stream.

######  Parameters

`hStream`
    \- Stream to enqueue function call in
`fn`
    \- The function to call once preceding stream operations are complete
`userData`
    \- User-specified data to be passed to the function

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Enqueues a host function to run in a stream. The function will be called after currently enqueued work and will block work added after it.

The host function must not make any CUDA API calls. Attempting to use a CUDA API may result in [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), but this is not required. The host function must not perform any synchronization that may depend on outstanding CUDA work not mandated to run earlier. Host functions without a mandated order (such as in independent streams) execute in undefined order and may be serialized.

For the purposes of Unified Memory, execution makes a number of guarantees:

  * The stream is considered idle for the duration of the function's execution. Thus, for example, the function may always use memory attached to the stream it was enqueued in.

  * The start of execution of the function has the same effect as synchronizing an event recorded in the same stream immediately prior to the function. It thus synchronizes streams which have been "joined" prior to the function.

  * Adding device work to any stream does not have the effect of making the stream active until all preceding host functions and stream callbacks have executed. Thus, for example, a function might use global attached memory even if work has been added to another stream, if the work has been ordered behind the function call with an event.

  * Completion of the function does not cause a stream to become active except as described above. The stream will remain idle if no device work follows the function, and will remain idle across consecutive host functions or stream callbacks without device work in between. Thus, for example, stream synchronization can be done by signaling from a host function at the end of the stream.


Note that, in contrast to [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), the function will not be called in the event of an error in the CUDA context.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."), [cuStreamAttachMemAsync](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533> "Attach memory to a stream asynchronously."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchKernel ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, unsigned int Â gridDimX, unsigned int Â gridDimY, unsigned int Â gridDimZ, unsigned int Â blockDimX, unsigned int Â blockDimY, unsigned int Â blockDimZ, unsigned int Â sharedMemBytes, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, void**Â kernelParams, void**Â extra )


Launches a CUDA function CUfunction or a CUDA kernel CUkernel.

######  Parameters

`f`
    \- Function [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) or Kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) to launch
`gridDimX`
    \- Width of grid in blocks
`gridDimY`
    \- Height of grid in blocks
`gridDimZ`
    \- Depth of grid in blocks
`blockDimX`
    \- X dimension of each thread block
`blockDimY`
    \- Y dimension of each thread block
`blockDimZ`
    \- Z dimension of each thread block
`sharedMemBytes`
    \- Dynamic shared-memory size per thread block in bytes
`hStream`
    \- Stream identifier
`kernelParams`
    \- Array of pointers to kernel parameters
`extra`
    \- Extra options

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Invokes the function [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) or the kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)`f` on a `gridDimX` x `gridDimY` x `gridDimZ` grid of blocks. Each block contains `blockDimX` x `blockDimY` x `blockDimZ` threads.

`sharedMemBytes` sets the amount of dynamic shared memory that will be available to each thread block.

Kernel parameters to `f` can be specified in one of two ways:

1) Kernel parameters can be specified via `kernelParams`. If `f` has N parameters, then `kernelParams` needs to be an array of N pointers. Each of `kernelParams`[0] through `kernelParams`[N-1] must point to a region of memory from which the actual kernel parameter will be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that information is retrieved directly from the kernel's image.

2) Kernel parameters can also be packaged by the application into a single buffer that is passed in via the `extra` parameter. This places the burden on the application of knowing each kernel parameter's size and alignment/padding within the buffer. Here is an example of using the `extra` parameter in this manner:


    â    size_t argBufferSize;
              char argBuffer[256];

              // populate argBuffer and argBufferSize

              void *config[] = {
                  [CU_LAUNCH_PARAM_BUFFER_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g36d10d0b40c51372877578a2cffd6acd>), argBuffer,
                  [CU_LAUNCH_PARAM_BUFFER_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf339c057cd94562ead93a192e11c17e9>),    &argBufferSize,
                  [CU_LAUNCH_PARAM_END](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd5c11cff5adfa5a69d66829399653532>)
              };
              status = [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);

The `extra` parameter exists to allow [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") to take additional less commonly used arguments. `extra` specifies a list of names of extra settings and their corresponding values. Each extra setting name is immediately followed by the corresponding value. The list must be terminated with either NULL or [CU_LAUNCH_PARAM_END](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd5c11cff5adfa5a69d66829399653532>).

  * [CU_LAUNCH_PARAM_END](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd5c11cff5adfa5a69d66829399653532>), which indicates the end of the `extra` array;

  * [CU_LAUNCH_PARAM_BUFFER_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g36d10d0b40c51372877578a2cffd6acd>), which specifies that the next value in `extra` will be a pointer to a buffer containing all the kernel parameters for launching kernel `f`;

  * [CU_LAUNCH_PARAM_BUFFER_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf339c057cd94562ead93a192e11c17e9>), which specifies that the next value in `extra` will be a pointer to a size_t containing the size of the buffer specified with [CU_LAUNCH_PARAM_BUFFER_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g36d10d0b40c51372877578a2cffd6acd>);


The error [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned if kernel parameters are specified with both `kernelParams` and `extra` (i.e. both `kernelParams` and `extra` are non-NULL).

Calling [cuLaunchKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") invalidates the persistent function state set through the following deprecated APIs: [cuFuncSetBlockShape()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd98ab7e00740f68145972deb6ddab271> "Sets the block-dimensions for the function."), [cuFuncSetSharedSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g9b5a3f121142f7b42aea48366c72bf8b> "Sets the dynamic shared-memory size for the function."), [cuParamSetSize()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gf6896c37762d695f5d161ee56cf86e62> "Sets the parameter size for the function."), [cuParamSeti()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g07f1264a68f97f582353b0f5dd9ebd5c> "Adds an integer parameter to the function's argument list."), [cuParamSetf()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1gd5e7679999e3792203d477abad2958c5> "Adds a floating-point parameter to the function's argument list."), [cuParamSetv()](<group__CUDA__EXEC__DEPRECATED.html#group__CUDA__EXEC__DEPRECATED_1g24e5ceee66d1a84609b74e77672638b6> "Adds arbitrary data to the function's argument list.").

Note that to use [cuLaunchKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), the kernel `f` must either have been compiled with toolchain version 3.2 or later so that it will contain kernel parameter information, or have no kernel parameters. If either of these conditions is not met, then [cuLaunchKernel()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.") will return [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>).

Note that the API can also be used to launch context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to launch the kernel on will either be taken from the specified stream `hStream` or the current context in case of NULL stream.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cudaLaunchKernel](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g2c91bfe5e072fcd28de6606dd43cd64b>), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuKernelSetCacheConfig](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51> "Sets the preferred cache configuration for a device kernel."), [cuKernelGetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f> "Returns information about a kernel."), [cuKernelSetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067> "Sets information about a kernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLaunchKernelEx ( const [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>)*Â config, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â f, void**Â kernelParams, void**Â extra )


Launches a CUDA function CUfunction or a CUDA kernel CUkernel with launch-time configuration.

######  Parameters

`config`
    \- Config to launch
`f`
    \- Function [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) or Kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) to launch
`kernelParams`
    \- Array of pointers to kernel parameters
`extra`
    \- Extra options

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_LAUNCH_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94f270bc1011b152febc8154b2b1e1b8d>), [CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b5da09cc5697599a56a71a04184ffdaa>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>), [CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e99e36a98a3a2c5123d422b9a1b69dd5f6>), [CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d26f67e0acc1563f87ddb94c638478cd>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Invokes the function [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) or the kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)`f` with the specified launch-time configuration `config`.

The [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>) structure is defined as:


    â       typedef struct CUlaunchConfig_st {
               unsigned int gridDimX;
               unsigned int gridDimY;
               unsigned int gridDimZ;
               unsigned int blockDimX;
               unsigned int blockDimY;
               unsigned int blockDimZ;
               unsigned int sharedMemBytes;
               [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>) hStream;
               [CUlaunchAttribute](<structCUlaunchAttribute.html#structCUlaunchAttribute>) *attrs;
               unsigned int numAttrs;
           } [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>);

where:

  * [CUlaunchConfig::gridDimX](<structCUlaunchConfig.html#structCUlaunchConfig_1689d9cc16fd423b080ec828b5bc9d885>) is the width of the grid in blocks.

  * [CUlaunchConfig::gridDimY](<structCUlaunchConfig.html#structCUlaunchConfig_18c2c6a003a635f55703d3e00873c1a04>) is the height of the grid in blocks.

  * [CUlaunchConfig::gridDimZ](<structCUlaunchConfig.html#structCUlaunchConfig_12ee1e4c17d2976638d5c1def7aab2173>) is the depth of the grid in blocks.

  * [CUlaunchConfig::blockDimX](<structCUlaunchConfig.html#structCUlaunchConfig_1c6348b2aec5d4cbe883351f0a4ca2404>) is the X dimension of each thread block.

  * [CUlaunchConfig::blockDimX](<structCUlaunchConfig.html#structCUlaunchConfig_1c6348b2aec5d4cbe883351f0a4ca2404>) is the Y dimension of each thread block.

  * [CUlaunchConfig::blockDimZ](<structCUlaunchConfig.html#structCUlaunchConfig_114969c04364799742f22a4eb97501f75>) is the Z dimension of each thread block.

  * [CUlaunchConfig::sharedMemBytes](<structCUlaunchConfig.html#structCUlaunchConfig_139281cdd7b80edb790b0fa85b2bca38f>) is the dynamic shared-memory size per thread block in bytes.

  * [CUlaunchConfig::hStream](<structCUlaunchConfig.html#structCUlaunchConfig_18bbdd01ea0d4d380fad9e8be14fb928b>) is the handle to the stream to perform the launch in. The CUDA context associated with this stream must match that associated with function f.

  * [CUlaunchConfig::attrs](<structCUlaunchConfig.html#structCUlaunchConfig_189bd86e2a9d67c421d5ad9650e57f375>) is an array of [CUlaunchConfig::numAttrs](<structCUlaunchConfig.html#structCUlaunchConfig_1c64a4dd37ff79255de128a5868658e06>) continguous [CUlaunchAttribute](<structCUlaunchAttribute.html#structCUlaunchAttribute>) elements. The value of this pointer is not considered if [CUlaunchConfig::numAttrs](<structCUlaunchConfig.html#structCUlaunchConfig_1c64a4dd37ff79255de128a5868658e06>) is zero. However, in that case, it is recommended to set the pointer to NULL.

  * [CUlaunchConfig::numAttrs](<structCUlaunchConfig.html#structCUlaunchConfig_1c64a4dd37ff79255de128a5868658e06>) is the number of attributes populating the first [CUlaunchConfig::numAttrs](<structCUlaunchConfig.html#structCUlaunchConfig_1c64a4dd37ff79255de128a5868658e06>) positions of the [CUlaunchConfig::attrs](<structCUlaunchConfig.html#structCUlaunchConfig_189bd86e2a9d67c421d5ad9650e57f375>) array.


Launch-time configuration is specified by adding entries to [CUlaunchConfig::attrs](<structCUlaunchConfig.html#structCUlaunchConfig_189bd86e2a9d67c421d5ad9650e57f375>). Each entry is an attribute ID and a corresponding attribute value.

The [CUlaunchAttribute](<structCUlaunchAttribute.html#structCUlaunchAttribute>) structure is defined as:


    â       typedef struct CUlaunchAttribute_st {
               [CUlaunchAttributeID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>) id;
               [CUlaunchAttributeValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>) value;
           } [CUlaunchAttribute](<structCUlaunchAttribute.html#structCUlaunchAttribute>);

where:

  * [CUlaunchAttribute::id](<structCUlaunchAttribute.html#structCUlaunchAttribute_132aed095f6c0ffe51ea05d61ee83a5df>) is a unique enum identifying the attribute.

  * [CUlaunchAttribute::value](<structCUlaunchAttribute.html#structCUlaunchAttribute_1924768cf94d6cf1d94691d30e491fc55>) is a union that hold the attribute value.


An example of using the `config` parameter:


    â       [CUlaunchAttribute](<structCUlaunchAttribute.html#structCUlaunchAttribute>) coopAttr = {.id = [CU_LAUNCH_ATTRIBUTE_COOPERATIVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331a2a65a417546d2d54837c1516ceaec4d>),
                                         .value = 1};
           [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>) config = {... // set block and grid dimensions
                                  .attrs = &coopAttr,
                                  .numAttrs = 1};

           [cuLaunchKernelEx](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb9c891eb6bb8f4089758e64c9c976db9> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel with launch-time configuration.")(&config, kernel, NULL, NULL);

The [CUlaunchAttributeID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>) enum is defined as:


    â       typedef enum CUlaunchAttributeID_enum {
               [CU_LAUNCH_ATTRIBUTE_IGNORE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd3313df820dd1fa823081b1923ff294b95b6>) = 0,
               [CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd3313761deda73b2f2b9da73406c7c4e9553>)   = 1,
               [CU_LAUNCH_ATTRIBUTE_COOPERATIVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331a2a65a417546d2d54837c1516ceaec4d>)            = 2,
               [CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331455ddb66a56b148882e3c6f23cd57cf3>) = 3,
               [CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331f1a227eb6283f2a292cf5a38cdceb638>)                    = 4,
               [CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331ebd12c41bbb15196ea07644d90b9f55d>) = 5,
               [CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd33116b07ee1531b349fae46df8623a9fd24>)    = 6,
               [CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd3311667102e857f5d37bb4f460c530dfb13>)                   = 7,
               [CU_LAUNCH_ATTRIBUTE_PRIORITY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd33176a390a5743b2f72d392a429d413f64f>)               = 8,
               [CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331b05f66cedd0038b77034a8c62127a09d>)    = 9,
               [CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331b57049dbe6c473013088dbc3cbc41139>)        = 10,
               [CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331e0237963b795260b6d24842cdde29f3b>) = 11,
               [CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd33121248c3419121151076d819052270513>) = 12,
               [CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331e46e24e30991b52ec5b267e40e093a4b>) = 13,
           } [CUlaunchAttributeID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>);

and the corresponding [CUlaunchAttributeValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>) union as :


    â       typedef union CUlaunchAttributeValue_union {
               [CUaccessPolicyWindow](<structCUaccessPolicyWindow__v1.html#structCUaccessPolicyWindow__v1>) accessPolicyWindow;
               int cooperative;
               CUsynchronizationPolicy syncPolicy;
               struct {
                   unsigned int x;
                   unsigned int y;
                   unsigned int z;
               } clusterDim;
               [CUclusterSchedulingPolicy](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2d60fcf51c7e8a70bd27687f19543192>) clusterSchedulingPolicyPreference;
               int programmaticStreamSerializationAllowed;
               struct {
                   [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>) event;
                   int flags;
                   int triggerAtBlockStart;
               } programmaticEvent;
               int priority;
               [CUlaunchMemSyncDomainMap](<structCUlaunchMemSyncDomainMap.html#structCUlaunchMemSyncDomainMap>) memSyncDomainMap;
               [CUlaunchMemSyncDomain](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g471f645fa24df354626fe8107358e05f>) memSyncDomain;
               struct {
                   unsigned int x;
                   unsigned int y;
                   unsigned int z;
               } preferredClusterDim;
               struct {
                   [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>) event;
                   int flags;
               } launchCompletionEvent;
               struct {
                   int deviceUpdatable;
                   [CUgraphDeviceNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g90c635c072ada74bb594cdc06b155b4a>) devNode;
               } deviceUpdatableKernelNode;
           } [CUlaunchAttributeValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>);

Setting [CU_LAUNCH_ATTRIBUTE_COOPERATIVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331a2a65a417546d2d54837c1516ceaec4d>) to a non-zero value causes the kernel launch to be a cooperative launch, with exactly the same usage and semantics of [cuLaunchCooperativeKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute.").

Setting [CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd33116b07ee1531b349fae46df8623a9fd24>) to a non-zero values causes the kernel to use programmatic means to resolve its stream dependency -- enabling the CUDA runtime to opportunistically allow the grid's execution to overlap with the previous kernel in the stream, if that kernel requests the overlap.

[CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd3311667102e857f5d37bb4f460c530dfb13>) records an event along with the kernel launch. Event recorded through this launch attribute is guaranteed to only trigger after all block in the associated kernel trigger the event. A block can trigger the event through PTX launchdep.release or CUDA builtin function [cudaTriggerProgrammaticLaunchCompletion()](<../cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1ge194af462d927583bed3acf60d450218>). A trigger can also be inserted at the beginning of each block's execution if triggerAtBlockStart is set to non-0. Note that dependents (including the CPU thread calling [cuEventSynchronize()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete.")) are not guaranteed to observe the release precisely when it is released. For example, [cuEventSynchronize()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete.") may only observe the event trigger long after the associated kernel has completed. This recording type is primarily meant for establishing programmatic dependency between device tasks. The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. created with [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>) flag set).

[CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd33121248c3419121151076d819052270513>) records an event along with the kernel launch. Nominally, the event is triggered once all blocks of the kernel have begun execution. Currently this is a best effort. If a kernel B has a launch completion dependency on a kernel A, B may wait until A is complete. Alternatively, blocks of B may begin before all blocks of A have begun, for example:

  * If B can claim execution resources unavailable to A, for example if they run on different GPUs.

  * If B is a higher priority than A.


Exercise caution if such an ordering inversion could lead to deadlock. The event supplied must not be an interprocess or interop event. The event must disable timing (i.e. must be created with the [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>) flag set).

Setting [CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331e46e24e30991b52ec5b267e40e093a4b>) to 1 on a captured launch causes the resulting kernel node to be device-updatable. This attribute is specific to graphs, and passing it to a launch in a non-capturing stream results in an error. Passing a value other than 0 or 1 is not allowed.

On success, a handle will be returned via CUlaunchAttributeValue::deviceUpdatableKernelNode::devNode which can be passed to the various device-side update functions to update the node's kernel parameters from within another kernel. For more information on the types of device updates that can be made, as well as the relevant limitations thereof, see [cudaGraphKernelNodeUpdatesApply](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g2d558cf37c9616365c67447e61ac0d6a>).

Kernel nodes which are device-updatable have additional restrictions compared to regular kernel nodes. Firstly, device-updatable nodes cannot be removed from their graph via [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."). Additionally, once opted-in to this functionality, a node cannot opt out, and any attempt to set the attribute to 0 will result in an error. Graphs containing one or more device-updatable node also do not allow multiple instantiation.

[CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331e0237963b795260b6d24842cdde29f3b>) allows the kernel launch to specify a preferred substitute cluster dimension. Blocks may be grouped according to either the dimensions specified with this attribute (grouped into a "preferred substitute cluster"), or the one specified with [CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331f1a227eb6283f2a292cf5a38cdceb638>) attribute (grouped into a "regular cluster"). The cluster dimensions of a "preferred substitute cluster" shall be an integer multiple greater than zero of the regular cluster dimensions. The device will attempt - on a best-effort basis - to group thread blocks into preferred clusters over grouping them into regular clusters. When it deems necessary (primarily when the device temporarily runs out of physical resources to launch the larger preferred clusters), the device may switch to launch the regular clusters instead to attempt to utilize as much of the physical device resources as possible.

Each type of cluster will have its enumeration / coordinate setup as if the grid consists solely of its type of cluster. For example, if the preferred substitute cluster dimensions double the regular cluster dimensions, there might be simultaneously a regular cluster indexed at (1,0,0), and a preferred cluster indexed at (1,0,0). In this example, the preferred substitute cluster (1,0,0) replaces regular clusters (2,0,0) and (3,0,0) and groups their blocks.

This attribute will only take effect when a regular cluster dimension has been specified. The preferred substitute The preferred substitute cluster dimension must be an integer multiple greater than zero of the regular cluster dimension and must divide the grid. It must also be no more than `maxBlocksPerCluster`, if it is set in the kernel's `__launch_bounds__`. Otherwise it must be less than the maximum value the driver can support. Otherwise, setting this attribute to a value physically unable to fit on any particular device is permitted.

The effect of other attributes is consistent with their effect when set via persistent APIs.

See [cuStreamSetAttribute](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga2c5fc0292861a42f264af6ca48be8c0> "Sets stream attribute.") for

  * [CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd3313761deda73b2f2b9da73406c7c4e9553>)

  * [CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331455ddb66a56b148882e3c6f23cd57cf3>)


See [cuFuncSetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g0e37dce0173bc883aa1e5b14dd747f26> "Sets information about a function.") for

  * [CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331f1a227eb6283f2a292cf5a38cdceb638>)

  * [CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg6f6565b334be6bb3134868e10bbdd331ebd12c41bbb15196ea07644d90b9f55d>)


Kernel parameters to `f` can be specified in the same ways that they can be using [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.").

Note that the API can also be used to launch context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to launch the kernel on will either be taken from the specified stream [CUlaunchConfig::hStream](<structCUlaunchConfig.html#structCUlaunchConfig_18bbdd01ea0d4d380fad9e8be14fb928b>) or the current context in case of NULL stream.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function."), [cudaLaunchKernel](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g2c91bfe5e072fcd28de6606dd43cd64b>), [cudaLaunchKernelEx](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g98d60efe48c3400a1c17a1edb698e530>), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuKernelSetCacheConfig](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51> "Sets the preferred cache configuration for a device kernel."), [cuKernelGetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f> "Returns information about a kernel."), [cuKernelSetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067> "Sets information about a kernel.")

* * *
