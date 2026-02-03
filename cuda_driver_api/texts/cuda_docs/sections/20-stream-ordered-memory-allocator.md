# Stream Ordered Memory Allocator

## 6.15.Â Stream Ordered Memory Allocator

This section describes the stream ordered memory allocator exposed by the low-level CUDA driver application programming interface.

**overview**

The asynchronous allocator allows the user to allocate and free in stream order. All asynchronous accesses of the allocation must happen between the stream executions of the allocation and the free. If the memory is accessed outside of the promised stream order, a use before allocation / use after free error will cause undefined behavior.

The allocator is free to reallocate the memory as long as it can guarantee that compliant memory accesses will not overlap temporally. The allocator may refer to internal stream ordering as well as inter-stream dependencies (such as CUDA events and null stream dependencies) when establishing the temporal guarantee. The allocator may also insert inter-stream dependencies to establish the temporal guarantee.

**Supported Platforms**

Whether or not a device supports the integrated stream ordered memory allocator may be queried by calling [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") with the device attribute [CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3f867cfe1025bda03e88ee109eeaa178e>)

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAllocAsync](<#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Allocates memory with stream ordered semantics.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAllocFromPoolAsync](<#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Allocates memory from a specified pool with stream ordered semantics.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemFreeAsync](<#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Frees memory with stream ordered semantics.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetDefaultMemPool](<#group__CUDA__MALLOC__ASYNC_1gfe5111eb15c977cd8d87132ff481072f>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool_out, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>)Â type )
     Returns the default memory pool for a given location and allocation type.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetMemPool](<#group__CUDA__MALLOC__ASYNC_1g5283d28ee187477e1a2b06fd731ec575>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>)Â type )
     Gets the current memory pool for a memory location and of a particular allocation type.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolCreate](<#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool, const [CUmemPoolProps](<structCUmemPoolProps__v1.html#structCUmemPoolProps__v1>)*Â poolProps )
     Creates a memory pool.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolDestroy](<#group__CUDA__MALLOC__ASYNC_1ge0e211115e5ad1c79250b9dd425b77f7>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool )
     Destroys the specified memory pool.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolExportPointer](<#group__CUDA__MALLOC__ASYNC_1gfe89f0478d26edaa91eb8a2e0349329d>) ( [CUmemPoolPtrExportData](<structCUmemPoolPtrExportData__v1.html#structCUmemPoolPtrExportData__v1>)*Â shareData_out, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )
     Export data to share a memory pool allocation between processes.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolExportToShareableHandle](<#group__CUDA__MALLOC__ASYNC_1g79ed285fdfffb76932871fb96fbba8f8>) ( void*Â handle_out, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â handleType, unsigned long longÂ flags )
     Exports a memory pool to the requested handle type.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolGetAccess](<#group__CUDA__MALLOC__ASYNC_1g838f28fd535a1cbd06c5f7fe0edbdcc7>) ( [CUmemAccess_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfba87b8c4a8cd091554d8e2c3fc9b40a>)*Â flags, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â memPool, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location )
     Returns the accessibility of a pool from a device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolGetAttribute](<#group__CUDA__MALLOC__ASYNC_1gd45ea7c43e4a1add4b971d06fa72eda4>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemPool_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5af6ea9ddd7633be98cb7de1bbf1d9f0>)Â attr, void*Â value )
     Gets attributes of a memory pool.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolImportFromShareableHandle](<#group__CUDA__MALLOC__ASYNC_1g02b4f18dd8a1c45b7f302800e90cec5b>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool_out, void*Â handle, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â handleType, unsigned long longÂ flags )
     imports a memory pool from a shared handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolImportPointer](<#group__CUDA__MALLOC__ASYNC_1g2620bb972ed5edcce312d3689454acbd>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â ptr_out, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemPoolPtrExportData](<structCUmemPoolPtrExportData__v1.html#structCUmemPoolPtrExportData__v1>)*Â shareData )
     Import a memory pool allocation from another process.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolSetAccess](<#group__CUDA__MALLOC__ASYNC_1gff3ce33e252443f4b087b94e42913406>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, const [CUmemAccessDesc](<structCUmemAccessDesc__v1.html#structCUmemAccessDesc__v1>)*Â map, size_tÂ count )
     Controls visibility of pools between devices.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolSetAttribute](<#group__CUDA__MALLOC__ASYNC_1g223e786cb217709235a06e41bccaec00>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemPool_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5af6ea9ddd7633be98cb7de1bbf1d9f0>)Â attr, void*Â value )
     Sets attributes of a memory pool.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPoolTrimTo](<#group__CUDA__MALLOC__ASYNC_1g9c7e267e3460945b0ca76c48314bb669>) ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, size_tÂ minBytesToKeep )
     Tries to release memory back to the OS.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemSetMemPool](<#group__CUDA__MALLOC__ASYNC_1g779e76c810c5f088210ea907730e17c9>) ( [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>)Â type, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool )
     Sets the current memory pool for a memory location and allocation type.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAllocAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Allocates memory with stream ordered semantics.

######  Parameters

`dptr`
    \- Returned device pointer
`bytesize`
    \- Number of bytes to allocate
`hStream`
    \- The stream establishing the stream ordering contract and the memory pool to allocate from

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) (default stream specified with no current context), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Inserts an allocation operation into `hStream`. A pointer to the allocated memory is returned immediately in *dptr. The allocation must not be accessed until the the allocation operation completes. The allocation comes from the memory pool current to the stream's device.

Note:

  * The default memory pool of a device contains device memory from that device.

  * Basic stream ordering allows future work submitted into the same stream to use the allocation. Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation operation completes before work submitted in a separate stream runs.

  * During stream capture, this function results in the creation of an allocation node. In this case, the allocation is owned by the graph instead of the memory pool. The memory pool's properties are used to set the node's creation parameters.


**See also:**

[cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuMemPoolSetAccess](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gff3ce33e252443f4b087b94e42913406> "Controls visibility of pools between devices."), [cuMemPoolSetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g223e786cb217709235a06e41bccaec00> "Sets attributes of a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAllocFromPoolAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Allocates memory from a specified pool with stream ordered semantics.

######  Parameters

`dptr`
    \- Returned device pointer
`bytesize`
    \- Number of bytes to allocate
`pool`
    \- The pool to allocate from
`hStream`
    \- The stream establishing the stream ordering semantic

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) (default stream specified with no current context), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Inserts an allocation operation into `hStream`. A pointer to the allocated memory is returned immediately in *dptr. The allocation must not be accessed until the the allocation operation completes. The allocation comes from the specified memory pool.

Note:

  * The specified memory pool may be from a device different than that of the specified `hStream`.


  * Basic stream ordering allows future work submitted into the same stream to use the allocation. Stream query, stream synchronize, and CUDA events can be used to guarantee that the allocation operation completes before work submitted in a separate stream runs.


Note:

During stream capture, this function results in the creation of an allocation node. In this case, the allocation is owned by the graph instead of the memory pool. The memory pool's properties are used to set the node's creation parameters.

**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuMemPoolSetAccess](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gff3ce33e252443f4b087b94e42913406> "Controls visibility of pools between devices."), [cuMemPoolSetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g223e786cb217709235a06e41bccaec00> "Sets attributes of a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemFreeAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Frees memory with stream ordered semantics.

######  Parameters

`dptr`
    \- memory to free
`hStream`
    \- The stream establishing the stream ordering contract.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) (default stream specified with no current context), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Inserts a free operation into `hStream`. The allocation must not be accessed after stream execution reaches the free. After this API returns, accessing the memory from any subsequent work launched on the GPU or querying its pointer attributes results in undefined behavior.

Note:

During stream capture, this function results in the creation of a free node and must therefore be passed the address of a graph allocation.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetDefaultMemPool ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool_out, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>)Â type )


Returns the default memory pool for a given location and allocation type.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>)[CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

The memory location can be of one of [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) or [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>). The allocation type can be one of [CU_MEM_ALLOCATION_TYPE_PINNED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b646624651d13be111040ffdf1161511c>) or [CU_MEM_ALLOCATION_TYPE_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b774fc1109cfbb0a357d6701483177cc1>). When the allocation type is [CU_MEM_ALLOCATION_TYPE_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b774fc1109cfbb0a357d6701483177cc1>), the location type can also be [CU_MEM_LOCATION_TYPE_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ecfc8f2ab14e813f7afe8019052526fa4>) to indicate no preferred location for the managed memory pool. In all other cases, the call returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemPoolTrimTo](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g9c7e267e3460945b0ca76c48314bb669> "Tries to release memory back to the OS."), [cuMemPoolGetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gd45ea7c43e4a1add4b971d06fa72eda4> "Gets attributes of a memory pool."), [cuMemPoolSetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g223e786cb217709235a06e41bccaec00> "Sets attributes of a memory pool."), [cuMemPoolSetAccess](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gff3ce33e252443f4b087b94e42913406> "Controls visibility of pools between devices."), [cuMemGetMemPool](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g5283d28ee187477e1a2b06fd731ec575> "Gets the current memory pool for a memory location and of a particular allocation type."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetMemPool ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>)Â type )


Gets the current memory pool for a memory location and of a particular allocation type.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

The memory location can be of one of [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) or [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>). The allocation type can be one of [CU_MEM_ALLOCATION_TYPE_PINNED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b646624651d13be111040ffdf1161511c>) or [CU_MEM_ALLOCATION_TYPE_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b774fc1109cfbb0a357d6701483177cc1>). When the allocation type is [CU_MEM_ALLOCATION_TYPE_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b774fc1109cfbb0a357d6701483177cc1>), the location type can also be [CU_MEM_LOCATION_TYPE_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ecfc8f2ab14e813f7afe8019052526fa4>) to indicate no preferred location for the managed memory pool. In all other cases, the call returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

Returns the last pool provided to [cuMemSetMemPool](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g779e76c810c5f088210ea907730e17c9> "Sets the current memory pool for a memory location and allocation type.") or [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.") for this location and allocation type or the location's default memory pool if [cuMemSetMemPool](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g779e76c810c5f088210ea907730e17c9> "Sets the current memory pool for a memory location and allocation type.") or [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.") for that allocType and location has never been called. By default the current mempool of a location is the default mempool for a device. Otherwise the returned pool must have been set with [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.").

**See also:**

[cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device."), [cuMemSetMemPool](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g779e76c810c5f088210ea907730e17c9> "Sets the current memory pool for a memory location and allocation type.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolCreate ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool, const [CUmemPoolProps](<structCUmemPoolProps__v1.html#structCUmemPoolProps__v1>)*Â poolProps )


Creates a memory pool.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Creates a CUDA memory pool and returns the handle in `pool`. The `poolProps` determines the properties of the pool such as the backing device and IPC capabilities.

To create a memory pool for HOST memory not targeting a specific NUMA node, applications must set set CUmemPoolProps::CUmemLocation::type to [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>). CUmemPoolProps::CUmemLocation::id is ignored for such pools. Pools created with the type [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) are not IPC capable and [CUmemPoolProps::handleTypes](<structCUmemPoolProps__v1.html#structCUmemPoolProps__v1_169e75d604b122dbd39a8e3e3eacbe660>) must be 0, any other values will result in [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>). To create a memory pool targeting a specific host NUMA node, applications must set CUmemPoolProps::CUmemLocation::type to [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) and CUmemPoolProps::CUmemLocation::id must specify the NUMA ID of the host memory node. Specifying [CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e55b82116b2124510a1a3b6c52096daaa>) as the CUmemPoolProps::CUmemLocation::type will result in [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>). By default, the pool's memory will be accessible from the device it is allocated on. In the case of pools created with [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) or [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>), their default accessibility will be from the host CPU. Applications can control the maximum size of the pool by specifying a non-zero value for [CUmemPoolProps::maxSize](<structCUmemPoolProps__v1.html#structCUmemPoolProps__v1_10f9278cc88653f1eee70ab6a7a2ad7f3>). If set to 0, the maximum size of the pool will default to a system dependent value.

Applications that intend to use [CU_MEM_HANDLE_TYPE_FABRIC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg450a23153d86fce0afe30e25d63caef9e38a182adb450da6c1a3f29cd5dca032>) based memory sharing must ensure: (1) `nvidia-caps-imex-channels` character device is created by the driver and is listed under /proc/devices (2) have at least one IMEX channel file accessible by the user launching the application.

When exporter and importer CUDA processes have been granted access to the same IMEX channel, they can securely share memory.

The IMEX channel security model works on a per user basis. Which means all processes under a user can share memory if the user has access to a valid IMEX channel. When multi-user isolation is desired, a separate IMEX channel is required for each user.

These channel files exist in /dev/nvidia-caps-imex-channels/channel* and can be created using standard OS native calls like mknod on Linux. For example: To create channel0 with the major number from /proc/devices users can execute the following command: `mknod /dev/nvidia-caps-imex-channels/channel0 c <major number>=""> 0`

To create a managed memory pool, applications must set [CUmemPoolProps::CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>) to CU_MEM_ALLOCATION_TYPE_MANAGED. [CUmemPoolProps::CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>) must also be set to CU_MEM_HANDLE_TYPE_NONE since IPC is not supported. For managed memory pools, CUmemPoolProps::CUmemLocation will be treated as the preferred location for all allocations created from the pool. An application can also set CU_MEM_LOCATION_TYPE_NONE to indicate no preferred location. [CUmemPoolProps::maxSize](<structCUmemPoolProps__v1.html#structCUmemPoolProps__v1_10f9278cc88653f1eee70ab6a7a2ad7f3>) must be set to zero for managed memory pools. [CUmemPoolProps::usage](<structCUmemPoolProps__v1.html#structCUmemPoolProps__v1_181dcd809a228c4346f8e732bb8e9070b>) should be zero as decompress for managed memory is not supported. For managed memory pools, all devices on the system must have non-zero concurrentManagedAccess. If not, this call returns CUDA_ERROR_NOT_SUPPORTED

Note:

Specifying CU_MEM_HANDLE_TYPE_NONE creates a memory pool that will not support IPC.

**See also:**

[cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics."), [cuMemPoolExportToShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g79ed285fdfffb76932871fb96fbba8f8> "Exports a memory pool to the requested handle type.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolDestroy ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool )


Destroys the specified memory pool.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

If any pointers obtained from this pool haven't been freed or the pool has free operations that haven't completed when [cuMemPoolDestroy](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1ge0e211115e5ad1c79250b9dd425b77f7> "Destroys the specified memory pool.") is invoked, the function will return immediately and the resources associated with the pool will be released automatically once there are no more outstanding allocations.

Destroying the current mempool of a device sets the default mempool of that device as the current mempool for that device.

Note:

A device's default memory pool cannot be destroyed.

**See also:**

[cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolExportPointer ( [CUmemPoolPtrExportData](<structCUmemPoolPtrExportData__v1.html#structCUmemPoolPtrExportData__v1>)*Â shareData_out, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )


Export data to share a memory pool allocation between processes.

######  Parameters

`shareData_out`
    \- Returned export data
`ptr`
    \- pointer to memory being exported

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Constructs `shareData_out` for sharing a specific allocation from an already shared memory pool. The recipient process can import the allocation with the [cuMemPoolImportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g2620bb972ed5edcce312d3689454acbd> "Import a memory pool allocation from another process.") api. The data is not a handle and may be shared through any IPC mechanism.

**See also:**

[cuMemPoolExportToShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g79ed285fdfffb76932871fb96fbba8f8> "Exports a memory pool to the requested handle type."), [cuMemPoolImportFromShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g02b4f18dd8a1c45b7f302800e90cec5b> "imports a memory pool from a shared handle."), [cuMemPoolImportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g2620bb972ed5edcce312d3689454acbd> "Import a memory pool allocation from another process.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolExportToShareableHandle ( void*Â handle_out, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â handleType, unsigned long longÂ flags )


Exports a memory pool to the requested handle type.

######  Parameters

`handle_out`
    \- Returned OS handle
`pool`
    \- pool to export
`handleType`
    \- the type of handle to create
`flags`
    \- must be 0

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Given an IPC capable mempool, create an OS handle to share the pool with another process. A recipient process can convert the shareable handle into a mempool with [cuMemPoolImportFromShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g02b4f18dd8a1c45b7f302800e90cec5b> "imports a memory pool from a shared handle."). Individual pointers can then be shared with the [cuMemPoolExportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gfe89f0478d26edaa91eb8a2e0349329d> "Export data to share a memory pool allocation between processes.") and [cuMemPoolImportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g2620bb972ed5edcce312d3689454acbd> "Import a memory pool allocation from another process.") APIs. The implementation of what the shareable handle is and how it can be transferred is defined by the requested handle type.

Note:

: To create an IPC capable mempool, create a mempool with a CUmemAllocationHandleType other than CU_MEM_HANDLE_TYPE_NONE.

**See also:**

[cuMemPoolImportFromShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g02b4f18dd8a1c45b7f302800e90cec5b> "imports a memory pool from a shared handle."), [cuMemPoolExportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gfe89f0478d26edaa91eb8a2e0349329d> "Export data to share a memory pool allocation between processes."), [cuMemPoolImportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g2620bb972ed5edcce312d3689454acbd> "Import a memory pool allocation from another process."), [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuMemPoolSetAccess](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gff3ce33e252443f4b087b94e42913406> "Controls visibility of pools between devices."), [cuMemPoolSetAttribute](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g223e786cb217709235a06e41bccaec00> "Sets attributes of a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolGetAccess ( [CUmemAccess_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfba87b8c4a8cd091554d8e2c3fc9b40a>)*Â flags, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â memPool, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location )


Returns the accessibility of a pool from a device.

######  Parameters

`flags`
    \- the accessibility of the pool from the specified location
`memPool`
    \- the pool being queried
`location`
    \- the location accessing the pool

###### Description

Returns the accessibility of the pool's memory from the specified location.

**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolGetAttribute ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemPool_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5af6ea9ddd7633be98cb7de1bbf1d9f0>)Â attr, void*Â value )


Gets attributes of a memory pool.

######  Parameters

`pool`
    \- The memory pool to get attributes of
`attr`
    \- The attribute to get
`value`
    \- Retrieved value

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Supported attributes are:

  * [CU_MEMPOOL_ATTR_RELEASE_THRESHOLD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f050fa2c731b01422ffbeb8c16ce0ba9a8>): (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)

  * [CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f0e11d636e9d56f9ce8a449b887fe2917f>): (value type = int) Allow [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") to use memory asynchronously freed in another stream as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled)

  * [CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f0d1016ed326b53c1eb06e8da5d40359cb>): (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)

  * [CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f03b7973c1b89f3a05f26702ade1124e9f>): (value type = int) Allow [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics.") (default enabled).

  * [CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f0ecd4938eb06ce224e04d4c56fea476c6>): (value type = cuuint64_t) Amount of backing memory currently allocated for the mempool

  * [CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f05ccb2736772fd27a735f15905118cbf6>): (value type = cuuint64_t) High watermark of backing memory allocated for the mempool since the last time it was reset.

  * [CU_MEMPOOL_ATTR_USED_MEM_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f03f17742ca47490659f188bc75be9b85c>): (value type = cuuint64_t) Amount of memory from the pool that is currently in use by the application.

  * [CU_MEMPOOL_ATTR_USED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f027fd3cf6d5a152572e55e9736da6987b>): (value type = cuuint64_t) High watermark of the amount of memory from the pool that was in use by the application.


**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolImportFromShareableHandle ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)*Â pool_out, void*Â handle, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â handleType, unsigned long longÂ flags )


imports a memory pool from a shared handle.

######  Parameters

`pool_out`
    \- Returned memory pool
`handle`
    \- OS handle of the pool to open
`handleType`
    \- The type of handle being imported
`flags`
    \- must be 0

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Specific allocations can be imported from the imported pool with cuMemPoolImportPointer.

If `handleType` is [CU_MEM_HANDLE_TYPE_FABRIC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg450a23153d86fce0afe30e25d63caef9e38a182adb450da6c1a3f29cd5dca032>) and the importer process has not been granted access to the same IMEX channel as the exporter process, this API will error as [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>).

Note:

Imported memory pools do not support creating new allocations. As such imported memory pools may not be used in cuDeviceSetMemPool or [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.") calls.

**See also:**

[cuMemPoolExportToShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g79ed285fdfffb76932871fb96fbba8f8> "Exports a memory pool to the requested handle type."), [cuMemPoolExportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gfe89f0478d26edaa91eb8a2e0349329d> "Export data to share a memory pool allocation between processes."), [cuMemPoolImportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g2620bb972ed5edcce312d3689454acbd> "Import a memory pool allocation from another process.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolImportPointer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â ptr_out, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemPoolPtrExportData](<structCUmemPoolPtrExportData__v1.html#structCUmemPoolPtrExportData__v1>)*Â shareData )


Import a memory pool allocation from another process.

######  Parameters

`ptr_out`
    \- pointer to imported memory
`pool`
    \- pool from which to import
`shareData`
    \- data specifying the memory to import

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Returns in `ptr_out` a pointer to the imported memory. The imported memory must not be accessed before the allocation operation completes in the exporting process. The imported memory must be freed from all importing processes before being freed in the exporting process. The pointer may be freed with cuMemFree or cuMemFreeAsync. If cuMemFreeAsync is used, the free must be completed on the importing process before the free operation on the exporting process.

Note:

The cuMemFreeAsync api may be used in the exporting process before the cuMemFreeAsync operation completes in its stream as long as the cuMemFreeAsync in the exporting process specifies a stream with a stream dependency on the importing process's cuMemFreeAsync.

**See also:**

[cuMemPoolExportToShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g79ed285fdfffb76932871fb96fbba8f8> "Exports a memory pool to the requested handle type."), [cuMemPoolImportFromShareableHandle](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g02b4f18dd8a1c45b7f302800e90cec5b> "imports a memory pool from a shared handle."), [cuMemPoolExportPointer](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gfe89f0478d26edaa91eb8a2e0349329d> "Export data to share a memory pool allocation between processes.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolSetAccess ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, const [CUmemAccessDesc](<structCUmemAccessDesc__v1.html#structCUmemAccessDesc__v1>)*Â map, size_tÂ count )


Controls visibility of pools between devices.

######  Parameters

`pool`
    \- The pool being modified
`map`
    \- Array of access descriptors. Each descriptor instructs the access to enable for a single gpu.
`count`
    \- Number of descriptors in the map array.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolSetAttribute ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, [CUmemPool_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5af6ea9ddd7633be98cb7de1bbf1d9f0>)Â attr, void*Â value )


Sets attributes of a memory pool.

######  Parameters

`pool`
    \- The memory pool to modify
`attr`
    \- The attribute to modify
`value`
    \- Pointer to the value to assign

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Supported attributes are:

  * [CU_MEMPOOL_ATTR_RELEASE_THRESHOLD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f050fa2c731b01422ffbeb8c16ce0ba9a8>): (value type = cuuint64_t) Amount of reserved memory in bytes to hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or context synchronize. (default 0)

  * [CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f0e11d636e9d56f9ce8a449b887fe2917f>): (value type = int) Allow [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") to use memory asynchronously freed in another stream as long as a stream ordering dependency of the allocating stream on the free action exists. Cuda events and null stream interactions can create the required stream ordered dependencies. (default enabled)

  * [CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f0d1016ed326b53c1eb06e8da5d40359cb>): (value type = int) Allow reuse of already completed frees when there is no dependency between the free and allocation. (default enabled)

  * [CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f03b7973c1b89f3a05f26702ade1124e9f>): (value type = int) Allow [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") to insert new stream dependencies in order to establish the stream ordering required to reuse a piece of memory released by [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics.") (default enabled).

  * [CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f05ccb2736772fd27a735f15905118cbf6>): (value type = cuuint64_t) Reset the high watermark that tracks the amount of backing memory that was allocated for the memory pool. It is illegal to set this attribute to a non-zero value.

  * [CU_MEMPOOL_ATTR_USED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5af6ea9ddd7633be98cb7de1bbf1d9f027fd3cf6d5a152572e55e9736da6987b>): (value type = cuuint64_t) Reset the high watermark that tracks the amount of used memory that was allocated for the memory pool.


**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPoolTrimTo ( [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool, size_tÂ minBytesToKeep )


Tries to release memory back to the OS.

######  Parameters

`pool`
    \- The memory pool to trim
`minBytesToKeep`
    \- If the pool has less than minBytesToKeep reserved, the TrimTo operation is a no-op. Otherwise the pool will be guaranteed to have at least minBytesToKeep bytes reserved after the operation.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Releases memory back to the OS until the pool contains fewer than minBytesToKeep reserved bytes, or there is no more memory that the allocator can safely release. The allocator cannot release OS allocations that back outstanding asynchronous allocations. The OS allocations may happen at different granularity from the user allocations.

Note:

  * : Allocations that have not been freed count as outstanding.

  * : Allocations that have been asynchronously freed but whose completion has not been observed on the host (eg. by a synchronize) can count as outstanding.


**See also:**

[cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemSetMemPool ( [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUmemAllocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7ed3482e0df8712d79a99bcb3bc4a95b>)Â type, [CUmemoryPool](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b96b1ef79f0cb312b51169e9f50e722>)Â pool )


Sets the current memory pool for a memory location and allocation type.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

The memory location can be of one of [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) or [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>). The allocation type can be one of [CU_MEM_ALLOCATION_TYPE_PINNED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b646624651d13be111040ffdf1161511c>) or [CU_MEM_ALLOCATION_TYPE_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b774fc1109cfbb0a357d6701483177cc1>). When the allocation type is [CU_MEM_ALLOCATION_TYPE_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b774fc1109cfbb0a357d6701483177cc1>), the location type can also be [CU_MEM_LOCATION_TYPE_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ecfc8f2ab14e813f7afe8019052526fa4>) to indicate no preferred location for the managed memory pool. In all other cases, the call returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

When a memory pool is set as the current memory pool, the location parameter should be the same as the location of the pool. The location and allocation type specified must match those of the pool otherwise [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned. By default, a memory location's current memory pool is its default memory pool that can be obtained via [cuMemGetDefaultMemPool](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gfe5111eb15c977cd8d87132ff481072f> "Returns the default memory pool for a given location and allocation type."). If the location type is [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) and the allocation type is [CU_MEM_ALLOCATION_TYPE_PINNED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7ed3482e0df8712d79a99bcb3bc4a95b646624651d13be111040ffdf1161511c>), then this API is the equivalent of calling [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.") with the location id as the device. For further details on the implications, please refer to the documentation for [cuDeviceSetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g4f2f276b84d9c2eaefdc76d6274db4a0> "Sets the current memory pool of a device.").

Note:

Use [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.") to specify asynchronous allocations from a device different than the one the stream runs on.

**See also:**

[cuDeviceGetDefaultMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc8bca3c97a78816303b8aa5773b741f2> "Returns the default mempool of a device."), [cuDeviceGetMemPool](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gdf186e9559d53a5eb18e572d48c1121b> "Gets the current mempool for a device."), [cuMemGetMemPool](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g5283d28ee187477e1a2b06fd731ec575> "Gets the current memory pool for a memory location and of a particular allocation type."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuMemPoolDestroy](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1ge0e211115e5ad1c79250b9dd425b77f7> "Destroys the specified memory pool."), [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.")

* * *
