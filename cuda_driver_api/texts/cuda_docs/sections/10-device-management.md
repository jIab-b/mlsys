# Device Management

## 6.5.Â Device Management

This section describes the device management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGet](<#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device, int Â ordinal )
     Returns a handle to a compute device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetAttribute](<#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266>) ( int*Â pi, [CUdevice_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge12b8a782bebe21b1ac0091bf9f4e2a3>)Â attrib, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns information about the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetCount](<#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74>) ( int*Â count )
     Returns the number of compute-capable devices.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetDefaultMemPool](<#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool_out, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns the default mempool of a device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetExecAffinitySupport](<#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d>) ( int*Â pi, [CUexecAffinityType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g14e6345acf2bda65be91eda77cf03f5c>)Â type, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns information about the execution affinity support of the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetHostAtomicCapabilities](<#group__CUDA__DEVICE_1g801bf845c6bd488103a2234379b15ce6>) ( unsigned int*Â capabilities, const [CUatomicOperation *](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfd7012de0abfe50cee089f3f00d6dcf3>)*Â operations, unsigned int Â count, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Queries details about atomic operations supported between the device and host.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetLuid](<#group__CUDA__DEVICE_1g630073c868f8878e89705ea831c49249>) ( char*Â luid, unsigned int*Â deviceNodeMask, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Return an LUID and device node mask for the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetMemPool](<#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Gets the current mempool for a device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetName](<#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f>) ( char*Â name, int Â len, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns an identifier string for the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetNvSciSyncAttributes](<#group__CUDA__DEVICE_1g0991e2b2b3cedee1ca77d6376e581335>) ( void*Â nvSciSyncAttrList, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, int Â flags )
     Return NvSciSync attributes that this device can support.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetTexture1DLinearMaxWidth](<#group__CUDA__DEVICE_1gb41b3a675bae9932bffa1c0ae969b1e0>) ( size_t*Â maxWidthInElements, [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>)Â format, unsignedÂ numChannels, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns the maximum number of elements allocatable in a 1D linear texture for a given texture element size.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetUuid](<#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4>) ( CUuuid*Â uuid, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Return an UUID for the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceSetMemPool](<#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool )
     Sets the current memory pool of a device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceTotalMem](<#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d>) ( size_t*Â bytes, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns the total amount of memory on the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuFlushGPUDirectRDMAWrites](<#group__CUDA__DEVICE_1g265e3c82ef0f0fe035f85c4c45a8fbdf>) ( [CUflushGPUDirectRDMAWritesTarget](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g01251451232c43bc5c7cb067ed2c28ef>)Â target, [CUflushGPUDirectRDMAWritesScope](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9eafc4def87e0f6600f905e756ec99d1>)Â scope )
     Blocks until remote writes are visible to the specified scope.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGet ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device, int Â ordinal )


Returns a handle to a compute device.

######  Parameters

`device`
    \- Returned device handle
`ordinal`
    \- Device number to get handle for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*device` a device handle given an ordinal in the range **[0,[cuDeviceGetCount()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices.")-1]**.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGetLuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g630073c868f8878e89705ea831c49249> "Return an LUID and device node mask for the device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device."), [cuDeviceGetExecAffinitySupport](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d> "Returns information about the execution affinity support of the device.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetAttribute ( int*Â pi, [CUdevice_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge12b8a782bebe21b1ac0091bf9f4e2a3>)Â attrib, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns information about the device.

######  Parameters

`pi`
    \- Returned device attribute value
`attrib`
    \- Device attribute to query
`dev`
    \- Device handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*pi` the integer value of the attribute `attrib` on device `dev`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device."), [cuDeviceGetExecAffinitySupport](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d> "Returns information about the execution affinity support of the device."), [cudaDeviceGetAttribute](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gb22e8256592b836df9a9cc36c9db7151>), [cudaGetDeviceProperties](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetCount ( int*Â count )


Returns the number of compute-capable devices.

######  Parameters

`count`
    \- Returned number of compute-capable devices

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*count` the number of devices with compute capability greater than or equal to 2.0 that are available for execution. If there is no such device, [cuDeviceGetCount()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices.") returns 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGetLuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g630073c868f8878e89705ea831c49249> "Return an LUID and device node mask for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device."), [cuDeviceGetExecAffinitySupport](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d> "Returns information about the execution affinity support of the device."), [cudaGetDeviceCount](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g18808e54893cfcaafefeab31a73cc55f>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetDefaultMemPool ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool_out, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns the default mempool of a device.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>)[CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

The default mempool of a device contains device memory from that device.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemPoolTrimTo](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g9c7e267e3460945b0ca76c48314bb669> "Tries to release memory back to the OS."), [cuMemPoolGetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gd45ea7c43e4a1add4b971d06fa72eda4> "Gets attributes of a memory pool."), [cuMemPoolSetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g223e786cb217709235a06e41bccaec00> "Sets attributes of a memory pool."), [cuMemPoolSetAccess](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gff3ce33e252443f4b087b94e42913406> "Controls visibility of pools between devices."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetExecAffinitySupport ( int*Â pi, [CUexecAffinityType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g14e6345acf2bda65be91eda77cf03f5c>)Â type, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns information about the execution affinity support of the device.

######  Parameters

`pi`
    \- 1 if the execution affinity type `type` is supported by the device, or 0 if not
`type`
    \- Execution affinity type to query
`dev`
    \- Device handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*pi` whether execution affinity type `type` is supported by device `dev`. The supported types are:

  * [CU_EXEC_AFFINITY_TYPE_SM_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg14e6345acf2bda65be91eda77cf03f5cc7764c90ce81e15aba5f26a3507cd00c>): 1 if context with limited SMs is supported by the device, or 0 if not;


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetHostAtomicCapabilities ( unsigned int*Â capabilities, const [CUatomicOperation *](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfd7012de0abfe50cee089f3f00d6dcf3>)*Â operations, unsigned int Â count, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Queries details about atomic operations supported between the device and host.

######  Parameters

`capabilities`
    \- Returned capability details of each requested operation
`operations`
    \- Requested operations
`count`
    \- Count of requested operations and size of capabilities
`dev`
    \- Device handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*capabilities` the details about requested atomic `*operations` over the the link between `dev` and the host. The allocated size of `*operations` and `*capabilities` must be `count`.

For each [CUatomicOperation](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfd7012de0abfe50cee089f3f00d6dcf3>) in `*operations`, the corresponding result in `*capabilities` will be a bitmask indicating which of [CUatomicOperationCapability](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf8fbc0f84fd6461c5611b3935b26e22c>) the link supports natively.

Returns [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>) if `dev` is not valid.

Returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if `*capabilities` or `*operations` is NULL, if `count` is 0, or if any of `*operations` is not valid.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetP2PAtomicCapabilities](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1gfd989876c8fd3291b520c0b561d5282d> "Queries details about atomic operations supported between two devices."), cudaDeviceGeHostAtomicCapabilities

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetLuid ( char*Â luid, unsigned int*Â deviceNodeMask, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Return an LUID and device node mask for the device.

######  Parameters

`luid`
    \- Returned LUID
`deviceNodeMask`
    \- Returned device node mask
`dev`
    \- Device to get identifier string for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Return identifying information (`luid` and `deviceNodeMask`) to allow matching device with graphics APIs.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device."), [cuDeviceGetExecAffinitySupport](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d> "Returns information about the execution affinity support of the device."), [cudaGetDeviceProperties](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetMemPool ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Gets the current mempool for a device.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the last pool provided to [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.") for this device or the device's default memory pool if [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.") has never been called. By default the current mempool is the default mempool for a device. Otherwise the returned pool must have been set with [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.").

**See also:**

[cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetName ( char*Â name, int Â len, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns an identifier string for the device.

######  Parameters

`name`
    \- Returned identifier string for the device
`len`
    \- Maximum length of string to store in `name`
`dev`
    \- Device to get identifier string for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns an ASCII string identifying the device `dev` in the NULL-terminated string pointed to by `name`. `len` specifies the maximum length of the string that may be returned. `name` is shortened to the specified `len`, if `len` is less than the device name

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGetLuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g630073c868f8878e89705ea831c49249> "Return an LUID and device node mask for the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device."), [cuDeviceGetExecAffinitySupport](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d> "Returns information about the execution affinity support of the device."), [cudaGetDeviceProperties](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetNvSciSyncAttributes ( void*Â nvSciSyncAttrList, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, int Â flags )


Return NvSciSync attributes that this device can support.

######  Parameters

`nvSciSyncAttrList`
    \- Return NvSciSync attributes supported.
`dev`
    \- Valid Cuda Device to get NvSciSync attributes for.
`flags`
    \- flags describing NvSciSync usage.

###### Description

Returns in `nvSciSyncAttrList`, the properties of NvSciSync that this CUDA device, `dev` can support. The returned `nvSciSyncAttrList` can be used to create an NvSciSync object that matches this device's capabilities.

If NvSciSyncAttrKey_RequiredPerm field in `nvSciSyncAttrList` is already set this API will return [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

The applications should set `nvSciSyncAttrList` to a valid NvSciSyncAttrList failing which this API will return [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>).

The `flags` controls how applications intends to use the NvSciSync created from the `nvSciSyncAttrList`. The valid flags are:

  * [CUDA_NVSCISYNC_ATTR_SIGNAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8709811eaf5a7849ad235aae65471a06>), specifies that the applications intends to signal an NvSciSync on this CUDA device.

  * [CUDA_NVSCISYNC_ATTR_WAIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd7a00d09b6061b828e13360b238cf9b4>), specifies that the applications intends to wait on an NvSciSync on this CUDA device.


At least one of these flags must be set, failing which the API returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>). Both the flags are orthogonal to one another: a developer may set both these flags that allows to set both wait and signal specific attributes in the same `nvSciSyncAttrList`.

Note that this API updates the input `nvSciSyncAttrList` with values equivalent to the following public attribute key-values: NvSciSyncAttrKey_RequiredPerm is set to

  * NvSciSyncAccessPerm_SignalOnly if [CUDA_NVSCISYNC_ATTR_SIGNAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8709811eaf5a7849ad235aae65471a06>) is set in `flags`.

  * NvSciSyncAccessPerm_WaitOnly if [CUDA_NVSCISYNC_ATTR_WAIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd7a00d09b6061b828e13360b238cf9b4>) is set in `flags`.

  * NvSciSyncAccessPerm_WaitSignal if both [CUDA_NVSCISYNC_ATTR_WAIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd7a00d09b6061b828e13360b238cf9b4>) and [CUDA_NVSCISYNC_ATTR_SIGNAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8709811eaf5a7849ad235aae65471a06>) are set in `flags`. NvSciSyncAttrKey_PrimitiveInfo is set to

  * NvSciSyncAttrValPrimitiveType_SysmemSemaphore on any valid `device`.

  * NvSciSyncAttrValPrimitiveType_Syncpoint if `device` is a Tegra device.

  * NvSciSyncAttrValPrimitiveType_SysmemSemaphorePayload64b if `device` is GA10X+. NvSciSyncAttrKey_GpuId is set to the same UUID that is returned for this `device` from [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device.").


[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

**See also:**

[cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuDestroyExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g7f13444973542fa50b7e75bcfb2f923d> "Destroys an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetTexture1DLinearMaxWidth ( size_t*Â maxWidthInElements, [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>)Â format, unsignedÂ numChannels, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns the maximum number of elements allocatable in a 1D linear texture for a given texture element size.

######  Parameters

`maxWidthInElements`
    \- Returned maximum number of texture elements allocatable for given `format` and `numChannels`.
`format`
    \- Texture format.
`numChannels`
    \- Number of channels per texture element.
`dev`
    \- Device handle.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `maxWidthInElements` the maximum number of texture elements allocatable in a 1D linear texture for given `format` and `numChannels`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cudaMemGetInfo](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g376b97f5ab20321ca46f7cfa9511b978>), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetUuid ( CUuuid*Â uuid, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Return an UUID for the device.

######  Parameters

`uuid`
    \- Returned UUID
`dev`
    \- Device to get identifier string for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns 16-octets identifying the device `dev` in the structure pointed by the `uuid`. If the device is in MIG mode, returns its MIG UUID which uniquely identifies the subscribed MIG compute instance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetLuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g630073c868f8878e89705ea831c49249> "Return an LUID and device node mask for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device."), [cudaGetDeviceProperties](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceSetMemPool ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool )


Sets the current memory pool of a device.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

The memory pool must be local to the specified device. [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") allocates from the current mempool of the provided stream's device. By default, a device's current memory pool is its default memory pool.

Note:

Use [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.") to specify asynchronous allocations from a device different than the one the stream runs on.

**See also:**

[cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuMemPoolDestroy](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1ge0e211115e5ad1c79250b9dd425b77f7> "Destroys the specified memory pool."), [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceTotalMem ( size_t*Â bytes, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns the total amount of memory on the device.

######  Parameters

`bytes`
    \- Returned memory available on device in bytes
`dev`
    \- Device handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*bytes` the total amount of memory available on the device `dev` in bytes.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceGetExecAffinitySupport](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g7f0091850e0841f367f13d623456427d> "Returns information about the execution affinity support of the device."), [cudaMemGetInfo](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g376b97f5ab20321ca46f7cfa9511b978>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuFlushGPUDirectRDMAWrites ( [CUflushGPUDirectRDMAWritesTarget](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g01251451232c43bc5c7cb067ed2c28ef>)Â target, [CUflushGPUDirectRDMAWritesScope](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9eafc4def87e0f6600f905e756ec99d1>)Â scope )


Blocks until remote writes are visible to the specified scope.

######  Parameters

`target`
    \- The target of the operation, see [CUflushGPUDirectRDMAWritesTarget](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g01251451232c43bc5c7cb067ed2c28ef>)
`scope`
    \- The scope of the operation, see [CUflushGPUDirectRDMAWritesScope](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9eafc4def87e0f6600f905e756ec99d1>)

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Blocks until GPUDirect RDMA writes to the target context via mappings created through APIs like nvidia_p2p_get_pages (see <https://docs.nvidia.com/cuda/gpudirect-rdma> for more information), are visible to the specified scope.

If the scope equals or lies within the scope indicated by [CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3dfb3df2c0e9347a7c13f71043e961c50>), the call will be a no-op and can be safely omitted for performance. This can be determined by comparing the numerical values between the two enums, with smaller scopes having smaller values.

On platforms that support GPUDirect RDMA writes via more than one path in hardware (see [CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75c53565b19e5c434edc5a65a6a7ab20ff810d1182d50bd1385eb543478b99f5>)), the user should consider those paths as belonging to separate ordering domains. Note that in such cases CUDA driver will report both RDMA writes ordering and RDMA write scope as ALL_DEVICES and a call to cuFlushGPUDirectRDMA will be a no-op, but when these multiple paths are used simultaneously, it is the user's responsibility to ensure ordering by using mechanisms outside the scope of CUDA.

Users may query support for this API via CU_DEVICE_ATTRIBUTE_FLUSH_FLUSH_GPU_DIRECT_RDMA_OPTIONS.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

* * *
