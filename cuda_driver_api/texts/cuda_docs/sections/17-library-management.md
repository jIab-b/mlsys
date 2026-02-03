# Library Management

## 6.12.Â Library Management

This section describes the library management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelGetAttribute](<#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f>) ( int*Â pi, [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns information about a kernel.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelGetFunction](<#group__CUDA__LIBRARY_1ge4cf9abafaba338acb977585b0d7374a>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)*Â pFunc, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel )
     Returns a function handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelGetLibrary](<#group__CUDA__LIBRARY_1g10ca8b20e237abbf3cf5a070d70b9cb3>) ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)*Â pLib, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel )
     Returns a library handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelGetName](<#group__CUDA__LIBRARY_1ge758151073b777ef3ba11a45f7d22adf>) ( const char**Â name, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â hfunc )
     Returns the function name for a CUkernel handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelGetParamInfo](<#group__CUDA__LIBRARY_1ga61653c9f13f713527e189fb0c2fe235>) ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, size_tÂ paramIndex, size_t*Â paramOffset, size_t*Â paramSize )
     Returns the offset and size of a kernel parameter in the device-side parameter layout.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelSetAttribute](<#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067>) ( [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, int Â val, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Sets information about a kernel.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuKernelSetCacheConfig](<#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51>) ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)Â config, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Sets the preferred cache configuration for a device kernel.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryEnumerateKernels](<#group__CUDA__LIBRARY_1ga8ae2f42ab3a8fe789ac2dced8219608>) ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)*Â kernels, unsigned int Â numKernels, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â lib )
     Retrieve the kernel handles within a library.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryGetGlobal](<#group__CUDA__LIBRARY_1g98708b50c11bc1c0addd6ecab96ae4ab>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â bytes, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â name )
     Returns a global device pointer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryGetKernel](<#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a>) ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)*Â pKernel, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â name )
     Returns a kernel handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryGetKernelCount](<#group__CUDA__LIBRARY_1g142732b1c9afaa662f21cae9a558d2d4>) ( unsigned int*Â count, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â lib )
     Returns the number of kernels within a library.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryGetManaged](<#group__CUDA__LIBRARY_1ga03f44378227ea68e6decd9d11c28fdf>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â bytes, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â name )
     Returns a pointer to managed memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryGetModule](<#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â pMod, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library )
     Returns a module handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryGetUnifiedFunction](<#group__CUDA__LIBRARY_1gb1b0ea992d64345562b694fdcd2c0334>) ( void**Â fptr, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â symbol )
     Returns a pointer to a unified function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryLoadData](<#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d>) ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)*Â library, const void*Â code, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â jitOptions, void**Â jitOptionsValues, unsigned int Â numJitOptions, [CUlibraryOption](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a1cdb7004bb8a24f1342de9004add23>)*Â libraryOptions, void**Â libraryOptionValues, unsigned int Â numLibraryOptions )
     Load a library with specified code and options.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryLoadFromFile](<#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205>) ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)*Â library, const char*Â fileName, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â jitOptions, void**Â jitOptionsValues, unsigned int Â numJitOptions, [CUlibraryOption](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a1cdb7004bb8a24f1342de9004add23>)*Â libraryOptions, void**Â libraryOptionValues, unsigned int Â numLibraryOptions )
     Load a library with specified file and options.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLibraryUnload](<#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914>) ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library )
     Unloads a library.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelGetAttribute ( int*Â pi, [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns information about a kernel.

######  Parameters

`pi`
    \- Returned attribute value
`attrib`
    \- Attribute requested
`kernel`
    \- Kernel to query attribute of
`dev`
    \- Device to query attribute of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*pi` the integer value of the attribute `attrib` for the kernel `kernel` for the requested device `dev`. The supported attributes are:

  * [CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bca244b9a52d7426e6684acebf4c9e24b8>): The maximum number of threads per block, beyond which a launch of the kernel would fail. This number depends on both the kernel and the requested device.

  * [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc15651a634781263c9d4ee6070a3991f4>): The size in bytes of statically-allocated shared memory per block required by this kernel. This does not include dynamically-allocated shared memory requested by the user at runtime.

  * [CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4a255dc4e2b8542e84c9431c1953a952>): The size in bytes of user-allocated constant memory required by this kernel.

  * [CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc0318e60c17eb22c70ffb59f610c504dd>): The size in bytes of local memory used by each thread of this kernel.

  * [CU_FUNC_ATTRIBUTE_NUM_REGS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc217730c04b1edbc80bb1772c1d6a7752>): The number of registers used by each thread of this kernel.

  * [CU_FUNC_ATTRIBUTE_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bccbd28200668ad2de39035446a89bf930>): The PTX virtual architecture version for which the kernel was compiled. This value is the major PTX version * 10 + the minor PTX version, so a PTX version 1.3 function would return the value 13. Note that this may return the undefined value of 0 for cubins compiled prior to CUDA 3.0.

  * [CU_FUNC_ATTRIBUTE_BINARY_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcc4f70f5d16889d0b75c3bf7a303eb437>): The binary architecture version for which the kernel was compiled. This value is the major binary version * 10 + the minor binary version, so a binary version 1.3 function would return the value 13. Note that this will return a value of 10 for legacy cubins that do not have a properly-encoded binary architecture version.

  * CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the kernel has been compiled with user specified option "-Xptxas \--dlcm=ca" set.

  * [CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc75b33d145e83462ef7292575015be03e>): The maximum size in bytes of dynamically-allocated shared memory.

  * [CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcc75f6fd470b848653f026b8c82c10ae3>): Preferred shared memory-L1 cache split ratio in percent of total shared memory.

  * [CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc8bd0417b504a8006cc6f57c023b54c2b>): If this attribute is set, the kernel must launch with a valid cluster size specified.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc1c83b56a254f78ddd5bf75ccfd15f0cb>): The required cluster width in blocks.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc0c3f2eb7eaea02e3c85a4bedd02be331>): The required cluster height in blocks.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4ab3672ad6476ad4bfa973e3083cdb32>): The required cluster depth in blocks.

  * [CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bce5ea0cdab1af87e68ac45e19e4c52c5d>): Indicates whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed. A non-portable cluster size may only function on the specific SKUs the program is tested on. The launch might fail if the program is run on a different hardware platform. CUDA API provides cudaOccupancyMaxActiveClusters to assist with checking whether the desired size can be launched on the current device. A portable cluster size is guaranteed to be functional on all compute capabilities higher than the target compute capability. The portable cluster size for sm_90 is 8 blocks per cluster. This value may increase for future compute capabilities. The specific hardware unit may support higher cluster sizes thatâs not guaranteed to be portable.

  * [CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcad30df41ca0cac5046c58a75d91326a6>): The block scheduling policy of a function. The value type is CUclusterSchedulingPolicy.


Note:

If another thread is trying to set the same attribute on the same device using [cuKernelSetAttribute()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067> "Sets information about a kernel.") simultaneously, the attribute query will give the old or new value depending on the interleavings chosen by the OS scheduler and memory consistency.

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuKernelSetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g1093ade718915249de3b14320d567067> "Sets information about a kernel."), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuKernelGetFunction](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ge4cf9abafaba338acb977585b0d7374a> "Returns a function handle."), [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle."), [cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelGetFunction ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)*Â pFunc, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel )


Returns a function handle.

######  Parameters

`pFunc`
    \- Returned function handle
`kernel`
    \- Kernel to retrieve function for the requested context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Returns in `pFunc` the handle of the function for the requested kernel `kernel` and the current context. If function handle is not found, the call returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>).

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle."), [cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelGetLibrary ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)*Â pLib, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel )


Returns a library handle.

######  Parameters

`pLib`
    \- Returned library handle
`kernel`
    \- Kernel to retrieve library handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `pLib` the handle of the library for the requested kernel `kernel`

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelGetName ( const char**Â name, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â hfunc )


Returns the function name for a CUkernel handle.

######  Parameters

`name`
    \- The returned name of the function
`hfunc`
    \- The function handle to retrieve the name for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `**name` the function name associated with the kernel handle `hfunc` . The function name is returned as a null-terminated string. The returned name is only valid when the kernel handle is valid. If the library is unloaded or reloaded, one must call the API again to get the updated name. This API may return a mangled name if the function is not declared as having C linkage. If either `**name` or `hfunc` is NULL, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelGetParamInfo ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, size_tÂ paramIndex, size_t*Â paramOffset, size_t*Â paramSize )


Returns the offset and size of a kernel parameter in the device-side parameter layout.

######  Parameters

`kernel`
    \- The kernel to query
`paramIndex`
    \- The parameter index to query
`paramOffset`
    \- Returns the offset into the device-side parameter layout at which the parameter resides
`paramSize`
    \- Optionally returns the size of the parameter in the device-side parameter layout

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Queries the kernel parameter at `paramIndex` into `kernel's` list of parameters, and returns in `paramOffset` and `paramSize` the offset and size, respectively, where the parameter will reside in the device-side parameter layout. This information can be used to update kernel node parameters from the device via [cudaGraphKernelNodeSetParam()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g0c2bd161eff1e47531eedce282e66d21>) and [cudaGraphKernelNodeUpdatesApply()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g2d558cf37c9616365c67447e61ac0d6a>). `paramIndex` must be less than the number of parameters that `kernel` takes. `paramSize` can be set to NULL if only the parameter offset is desired.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuFuncGetParamInfo](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g6874b82bcf2803902085645e46e0ca0e> "Returns the offset and size of a kernel parameter in the device-side parameter layout.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelSetAttribute ( [CUfunction_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9d955dde0904a9b43ca4d875ac1551bc>)Â attrib, int Â val, [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Sets information about a kernel.

######  Parameters

`attrib`
    \- Attribute requested
`val`
    \- Value to set
`kernel`
    \- Kernel to set attribute of
`dev`
    \- Device to set attribute of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

This call sets the value of a specified attribute `attrib` on the kernel `kernel` for the requested device `dev` to an integer value specified by `val`. This function returns CUDA_SUCCESS if the new value of the attribute could be successfully set. If the set fails, this call will return an error. Not all attributes can have values set. Attempting to set a value on a read-only attribute will result in an error (CUDA_ERROR_INVALID_VALUE)

Note that attributes set using [cuFuncSetAttribute()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g0e37dce0173bc883aa1e5b14dd747f26> "Sets information about a function.") will override the attribute set by this API irrespective of whether the call to [cuFuncSetAttribute()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g0e37dce0173bc883aa1e5b14dd747f26> "Sets information about a function.") is made before or after this API call. However, [cuKernelGetAttribute()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f> "Returns information about a kernel.") will always return the attribute value set by this API.

Supported attributes are:

  * [CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc75b33d145e83462ef7292575015be03e>): This is the maximum size in bytes of dynamically-allocated shared memory. The value should contain the requested maximum size of dynamically-allocated shared memory. The sum of this value and the function attribute [CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc15651a634781263c9d4ee6070a3991f4>) cannot exceed the device attribute [CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3e788564c0a95b866dc624fbc1b49dab3>). The maximal size of requestable dynamic shared memory may differ by GPU architecture.

  * [CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcc75f6fd470b848653f026b8c82c10ae3>): On devices where the L1 cache and shared memory use the same hardware resources, this sets the shared memory carveout preference, in percent of the total shared memory. See [CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a306a33c18889f6fc907412451c95154ed>) This is only a hint, and the driver can choose a different ratio if required to execute the function.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc1c83b56a254f78ddd5bf75ccfd15f0cb>): The required cluster width in blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc0c3f2eb7eaea02e3c85a4bedd02be331>): The required cluster height in blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.

  * [CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bc4ab3672ad6476ad4bfa973e3083cdb32>): The required cluster depth in blocks. The width, height, and depth values must either all be 0 or all be positive. The validity of the cluster dimensions is checked at launch time. If the value is set during compile time, it cannot be set at runtime. Setting it at runtime will return CUDA_ERROR_NOT_PERMITTED.

  * [CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bce5ea0cdab1af87e68ac45e19e4c52c5d>): Indicates whether the function can be launched with non-portable cluster size. 1 is allowed, 0 is disallowed.

  * [CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9d955dde0904a9b43ca4d875ac1551bcad30df41ca0cac5046c58a75d91326a6>): The block scheduling policy of a function. The value type is CUclusterSchedulingPolicy.


Note:

The API has stricter locking requirements in comparison to its legacy counterpart [cuFuncSetAttribute()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g0e37dce0173bc883aa1e5b14dd747f26> "Sets information about a function.") due to device-wide semantics. If multiple threads are trying to set the same attribute on the same device simultaneously, the attribute setting will depend on the interleavings chosen by the OS scheduler and memory consistency.

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuKernelGetAttribute](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1gd98317cb151b99fbd95767418122071f> "Returns information about a kernel."), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuKernelGetFunction](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ge4cf9abafaba338acb977585b0d7374a> "Returns a function handle."), [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle."), [cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuFuncSetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g0e37dce0173bc883aa1e5b14dd747f26> "Sets information about a function.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuKernelSetCacheConfig ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)Â kernel, [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)Â config, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Sets the preferred cache configuration for a device kernel.

######  Parameters

`kernel`
    \- Kernel to configure cache for
`config`
    \- Requested cache configuration
`dev`
    \- Device to set attribute of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through `config` the preferred cache configuration for the device kernel `kernel` on the requested device `dev`. This is only a preference. The driver will use the requested configuration if possible, but it is free to choose a different configuration if required to execute `kernel`. Any context-wide preference set via [cuCtxSetCacheConfig()](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context.") will be overridden by this per-kernel setting.

Note that attributes set using [cuFuncSetCacheConfig()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function.") will override the attribute set by this API irrespective of whether the call to [cuFuncSetCacheConfig()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function.") is made before or after this API call.

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.

The supported cache configurations are:

  * [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>): no preference for shared memory or L1 (default)

  * [CU_FUNC_CACHE_PREFER_SHARED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec712f43defb051d7985317bce426cccc8>): prefer larger shared memory and smaller L1 cache

  * [CU_FUNC_CACHE_PREFER_L1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ecb1e6c4e889e1a70ed5283172be08f6a5>): prefer larger L1 cache and smaller shared memory

  * [CU_FUNC_CACHE_PREFER_EQUAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec4434321280821d844a15b02e4d6c80a9>): prefer equal sized L1 cache and shared memory


Note:

The API has stricter locking requirements in comparison to its legacy counterpart [cuFuncSetCacheConfig()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function.") due to device-wide semantics. If multiple threads are trying to set a config on the same device simultaneously, the cache config setting will depend on the interleavings chosen by the OS scheduler and memory consistency.

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuLibraryGetKernel](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle."), [cuKernelGetFunction](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ge4cf9abafaba338acb977585b0d7374a> "Returns a function handle."), [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle."), [cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryEnumerateKernels ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)*Â kernels, unsigned int Â numKernels, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â lib )


Retrieve the kernel handles within a library.

######  Parameters

`kernels`
    \- Buffer where the kernel handles are returned to
`numKernels`
    \- Maximum number of kernel handles may be returned to the buffer
`lib`
    \- Library to query from

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `kernels` a maximum number of `numKernels` kernel handles within `lib`. The returned kernel handle becomes invalid when the library is unloaded.

**See also:**

[cuLibraryGetKernelCount](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g142732b1c9afaa662f21cae9a558d2d4> "Returns the number of kernels within a library.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryGetGlobal ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â bytes, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â name )


Returns a global device pointer.

######  Parameters

`dptr`
    \- Returned global device pointer for the requested context
`bytes`
    \- Returned global size in bytes
`library`
    \- Library to retrieve global from
`name`
    \- Name of global to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Returns in `*dptr` and `*bytes` the base pointer and size of the global with name `name` for the requested library `library` and the current context. If no global for the requested name `name` exists, the call returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>). One of the parameters `dptr` or `bytes` (not both) can be NULL in which case it is ignored.

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryGetKernel ( [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>)*Â pKernel, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â name )


Returns a kernel handle.

######  Parameters

`pKernel`
    \- Returned kernel handle
`library`
    \- Library to retrieve kernel from
`name`
    \- Name of kernel to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `pKernel` the handle of the kernel with name `name` located in library `library`. If kernel handle is not found, the call returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>).

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuKernelGetFunction](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1ge4cf9abafaba338acb977585b0d7374a> "Returns a function handle."), [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle."), [cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryGetKernelCount ( unsigned int*Â count, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â lib )


Returns the number of kernels within a library.

######  Parameters

`count`
    \- Number of kernels found within the library
`lib`
    \- Library to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `count` the number of kernels in `lib`.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryGetManaged ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â bytes, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â name )


Returns a pointer to managed memory.

######  Parameters

`dptr`
    \- Returned pointer to the managed memory
`bytes`
    \- Returned memory size in bytes
`library`
    \- Library to retrieve managed memory from
`name`
    \- Name of managed memory to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `*dptr` and `*bytes` the base pointer and size of the managed memory with name `name` for the requested library `library`. If no managed memory with the requested name `name` exists, the call returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>). One of the parameters `dptr` or `bytes` (not both) can be NULL in which case it is ignored. Note that managed memory for library `library` is shared across devices and is registered when the library is loaded into atleast one context.

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryGetModule ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â pMod, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library )


Returns a module handle.

######  Parameters

`pMod`
    \- Returned module handle
`library`
    \- Library to retrieve module from

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Returns in `pMod` the module handle associated with the current context located in library `library`. If module handle is not found, the call returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>).

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryGetUnifiedFunction ( void**Â fptr, [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library, const char*Â symbol )


Returns a pointer to a unified function.

######  Parameters

`fptr`
    \- Returned pointer to a unified function
`library`
    \- Library to retrieve function pointer memory from
`symbol`
    \- Name of function pointer to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `*fptr` the function pointer to a unified function denoted by `symbol`. If no unified function with name `symbol` exists, the call returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>). If there is no device with attribute [CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3f64e13496d96670082083817ba6c6266>) present in the system, the call may return [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>).

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryLoadData ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)*Â library, const void*Â code, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â jitOptions, void**Â jitOptionsValues, unsigned int Â numJitOptions, [CUlibraryOption](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a1cdb7004bb8a24f1342de9004add23>)*Â libraryOptions, void**Â libraryOptionValues, unsigned int Â numLibraryOptions )


Load a library with specified code and options.

######  Parameters

`library`
    \- Returned library
`code`
    \- Code to load
`jitOptions`
    \- Options for JIT
`jitOptionsValues`
    \- Option values for JIT
`numJitOptions`
    \- Number of options
`libraryOptions`
    \- Options for loading
`libraryOptionValues`
    \- Option values for loading
`numLibraryOptions`
    \- Number of options for loading

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>), [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95ab6c0086a6130b5b895ff15ce841ee6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Takes a pointer `code` and loads the corresponding library `library` based on the application defined library loading mode:

  * If module loading is set to EAGER, via the environment variables described in "Module loading", `library` is loaded eagerly into all contexts at the time of the call and future contexts at the time of creation until the library is unloaded with [cuLibraryUnload()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library.").

  * If the environment variables are set to LAZY, `library` is not immediately loaded onto all existent contexts and will only be loaded when a function is needed for that context, such as a kernel launch.


These environment variables are described in the CUDA programming guide under the "CUDA environment variables" section.

The `code` may be a cubin or fatbin as output by **nvcc** , or a NULL-terminated PTX, either as output by **nvcc** or hand-written, or Tile IR data. A fatbin should also contain relocatable code when doing separate compilation.

Options are passed as an array via `jitOptions` and any corresponding parameters are passed in `jitOptionsValues`. The number of total JIT options is supplied via `numJitOptions`. Any outputs will be returned via `jitOptionsValues`.

Library load options are passed as an array via `libraryOptions` and any corresponding parameters are passed in `libraryOptionValues`. The number of total library load options is supplied via `numLibraryOptions`.

Note:

If the library contains managed variables and no device in the system supports managed variables this call is expected to return [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

**See also:**

[cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryLoadFromFile ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)*Â library, const char*Â fileName, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â jitOptions, void**Â jitOptionsValues, unsigned int Â numJitOptions, [CUlibraryOption](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a1cdb7004bb8a24f1342de9004add23>)*Â libraryOptions, void**Â libraryOptionValues, unsigned int Â numLibraryOptions )


Load a library with specified file and options.

######  Parameters

`library`
    \- Returned library
`fileName`
    \- File to load from
`jitOptions`
    \- Options for JIT
`jitOptionsValues`
    \- Option values for JIT
`numJitOptions`
    \- Number of options
`libraryOptions`
    \- Options for loading
`libraryOptionValues`
    \- Option values for loading
`numLibraryOptions`
    \- Number of options for loading

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>), [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95ab6c0086a6130b5b895ff15ce841ee6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Takes a pointer `code` and loads the corresponding library `library` based on the application defined library loading mode:

  * If module loading is set to EAGER, via the environment variables described in "Module loading", `library` is loaded eagerly into all contexts at the time of the call and future contexts at the time of creation until the library is unloaded with [cuLibraryUnload()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library.").

  * If the environment variables are set to LAZY, `library` is not immediately loaded onto all existent contexts and will only be loaded when a function is needed for that context, such as a kernel launch.


These environment variables are described in the CUDA programming guide under the "CUDA environment variables" section.

The file should be a cubin file as output by **nvcc** , or a PTX file either as output by **nvcc** or handwritten, or a fatbin file as output by **nvcc** or hand-written, or Tile IR file. A fatbin should also contain relocatable code when doing separate compilation.

Options are passed as an array via `jitOptions` and any corresponding parameters are passed in `jitOptionsValues`. The number of total options is supplied via `numJitOptions`. Any outputs will be returned via `jitOptionsValues`.

Library load options are passed as an array via `libraryOptions` and any corresponding parameters are passed in `libraryOptionValues`. The number of total library load options is supplied via `numLibraryOptions`.

Note:

If the library contains managed variables and no device in the system supports managed variables this call is expected to return [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryUnload](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g24969cb24138171edf465bc8669d5914> "Unloads a library."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLibraryUnload ( [CUlibrary](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb3c679dac8f1ce28d437bedd0fc907d7>)Â library )


Unloads a library.

######  Parameters

`library`
    \- Library to unload

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Unloads the library specified with `library`

**See also:**

[cuLibraryLoadData](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g957f12ff5af4166f43c89d17cfb0a74d> "Load a library with specified code and options."), [cuLibraryLoadFromFile](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g88cff489fab37c7fd1985ceb61023205> "Load a library with specified file and options."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

* * *
