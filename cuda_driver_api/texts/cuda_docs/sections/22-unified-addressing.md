# Unified Addressing

## 6.17.Â Unified Addressing

This section describes the unified addressing functions of the low-level CUDA driver application programming interface.

**Overview**

CUDA devices can share a unified address space with the host. For these devices there is no distinction between a device pointer and a host pointer -- the same pointer value may be used to access memory from the host program and from a kernel running on the device (with exceptions enumerated below).

**Supported Platforms**

Whether or not a device supports unified addressing may be queried by calling [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") with the device attribute [CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3dc11dd6d9f149a7bae32499f2b802c0d>).

Unified addressing is automatically enabled in 64-bit processes

**Looking Up Information from Pointer Values**

It is possible to look up information about the memory which backs a pointer value. For instance, one may want to know if a pointer points to host or device memory. As another example, in the case of device memory, one may want to know on which CUDA device the memory resides. These properties may be queried using the function [cuPointerGetAttribute()](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c> "Returns information about a pointer.")

Since pointers are unique, it is not necessary to specify information about the pointers specified to the various copy functions in the CUDA API. The function [cuMemcpy()](<group__CUDA__MEM.html#group__CUDA__MEM_1g8d0ff510f26d4b87bd3a51e731e7f698> "Copies memory.") may be used to perform a copy between two pointers, ignoring whether they point to host or device memory (making [cuMemcpyHtoD()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyDtoD()](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), and [cuMemcpyDtoH()](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host.") unnecessary for devices supporting unified addressing). For multidimensional copies, the memory type [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) may be used to specify that the CUDA driver should infer the location of the pointer from its value.

**Automatic Mapping of Host Allocated Host Memory**

All host memory allocated in all contexts using [cuMemAllocHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") and [cuMemHostAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") is always directly accessible from all contexts on all devices that support unified addressing. This is the case regardless of whether or not the flags [CU_MEMHOSTALLOC_PORTABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g50f4528d46bda58b592551654a7ee0ff>) and [CU_MEMHOSTALLOC_DEVICEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g054589ee2a0f188e664d93965d81113d>) are specified.

The pointer value through which allocated host memory may be accessed in kernels on all devices that support unified addressing is the same as the pointer value through which that memory is accessed on the host, so it is not necessary to call [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") to get the device pointer for these allocations.

Note that this is not the case for memory allocated using the flag [CU_MEMHOSTALLOC_WRITECOMBINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7361580951deecace15352c97a210038>), as discussed below.

**Automatic Registration of Peer Memory**

Upon enabling direct access from a context that supports unified addressing to another peer context that supports unified addressing using [cuCtxEnablePeerAccess()](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context.") all memory allocated in the peer context using [cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") and [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") will immediately be accessible by the current context. The device pointer value through which any peer memory may be accessed in the current context is the same pointer value through which that memory may be accessed in the peer context.

**Exceptions, Disjoint Addressing**

Not all memory may be accessed on devices through the same pointer value through which they are accessed on the host. These exceptions are host memory registered using [cuMemHostRegister()](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA.") and host memory allocated using the flag [CU_MEMHOSTALLOC_WRITECOMBINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7361580951deecace15352c97a210038>). For these exceptions, there exists a distinct host and device address for the memory. The device address is guaranteed to not overlap any valid host pointer range and is guaranteed to have the same value across all contexts that support unified addressing.

This device address may be queried using [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") when a context using unified addressing is current. Either the host or the unified device pointer value may be used to refer to this memory through [cuMemcpy()](<group__CUDA__MEM.html#group__CUDA__MEM_1g8d0ff510f26d4b87bd3a51e731e7f698> "Copies memory.") and similar functions using the [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) memory type.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAdvise](<#group__CUDA__UNIFIED_1gaac8924b2f5a2a93f8775fb81c1a643f>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count, [CUmem_advise](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gcfe2ed2d4567745dd4ad41034136fff3>)Â advice, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)Â location )
     Advise about the usage of a given memory range.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemDiscardAndPrefetchBatchAsync](<#group__CUDA__UNIFIED_1g2a8dbb3c95608cff2269226298fd8f28>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptrs, size_t*Â sizes, size_tÂ count, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â prefetchLocs, size_t*Â prefetchLocIdxs, size_tÂ numPrefetchLocs, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Performs a batch of memory discards and prefetches asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemDiscardBatchAsync](<#group__CUDA__UNIFIED_1gf517eb7d44e9bae70cf6a1ec8d9ece4e>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptrs, size_t*Â sizes, size_tÂ count, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Performs a batch of memory discards asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPrefetchAsync](<#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)Â location, unsigned int Â flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Prefetches memory to the specified destination location.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemPrefetchBatchAsync](<#group__CUDA__UNIFIED_1g97fe632183ff5ff791813c3174fc5121>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptrs, size_t*Â sizes, size_tÂ count, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â prefetchLocs, size_t*Â prefetchLocIdxs, size_tÂ numPrefetchLocs, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Performs a batch of memory prefetches asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemRangeGetAttribute](<#group__CUDA__UNIFIED_1g1c92408a7d0d8875e19b1a58af56f67d>) ( void*Â data, size_tÂ dataSize, [CUmem_range_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3754eb90f64ed3c19b4e550d21d124fc>)Â attribute, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count )
     Query an attribute of a given memory range.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemRangeGetAttributes](<#group__CUDA__UNIFIED_1gc7ce142e60f8613cfb7d722b87dc9d12>) ( void**Â data, size_t*Â dataSizes, [CUmem_range_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3754eb90f64ed3c19b4e550d21d124fc>)*Â attributes, size_tÂ numAttributes, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count )
     Query attributes of a given memory range.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuPointerGetAttribute](<#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c>) ( void*Â data, [CUpointer_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc2cce590e35080745e72633dfc6e0b60>)Â attribute, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )
     Returns information about a pointer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuPointerGetAttributes](<#group__CUDA__UNIFIED_1gf65e9ea532e311dd049166e4894955ad>) ( unsigned int Â numAttributes, [CUpointer_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc2cce590e35080745e72633dfc6e0b60>)*Â attributes, void**Â data, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )
     Returns information about a pointer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuPointerSetAttribute](<#group__CUDA__UNIFIED_1g89f7ad29a657e574fdea2624b74d138e>) ( const void*Â value, [CUpointer_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc2cce590e35080745e72633dfc6e0b60>)Â attribute, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )
     Set attributes on a previously allocated memory region.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAdvise ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count, [CUmem_advise](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gcfe2ed2d4567745dd4ad41034136fff3>)Â advice, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)Â location )


Advise about the usage of a given memory range.

######  Parameters

`devPtr`
    \- Pointer to memory to set the advice for
`count`
    \- Size in bytes of the memory range
`advice`
    \- Advice to be applied for the specified memory range
`location`
    \- location to apply the advice for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Advise the Unified Memory subsystem about the usage pattern for the memory range starting at `devPtr` with a size of `count` bytes. The start address and end address of the memory range will be rounded down and rounded up respectively to be aligned to CPU page size before the advice is applied. The memory range must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") or declared via __managed__ variables. The memory range could also refer to system-allocated pageable memory provided it represents a valid, host-accessible region of memory and all additional constraints imposed by `advice` as outlined below are also satisfied. Specifying an invalid system-allocated pageable memory range results in an error being returned.

The `advice` parameter can take the following values:

  * [CU_MEM_ADVISE_SET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff35a99fb44378c84c56668550b94157fc0>): This implies that the data is mostly going to be read from and only occasionally written to. Any read accesses from any processor to this region will create a read-only copy of at least the accessed pages in that processor's memory. Additionally, if [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location.") is called on this region, it will create a read-only copy of the data on the destination processor. If the target location for [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location.") is a host NUMA node and a read-only copy already exists on another host NUMA node, that copy will be migrated to the targeted host NUMA node. If any processor writes to this region, all copies of the corresponding page will be invalidated except for the one where the write occurred. If the writing processor is the CPU and the preferred location of the page is a host NUMA node, then the page will also be migrated to that host NUMA node. The `location` argument is ignored for this advice. Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU that has a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). Also, if a context is created on a device that does not have the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) set, then read-duplication will not occur until all such contexts are destroyed. If the memory region refers to valid system-allocated pageable memory, then the accessing device must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>) for a read-only copy to be created on that device. Note however that if the accessing device also has a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>), then setting this advice will not create a read-only copy when that device accesses this memory region.


  * [CU_MEM_ADVISE_UNSET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff383a9e9744ef151f7b25b5c902ba6baca>): Undoes the effect of [CU_MEM_ADVISE_SET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff35a99fb44378c84c56668550b94157fc0>) and also prevents the Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated copies of the data will be collapsed into a single copy. The location for the collapsed copy will be the preferred location if the page has a preferred location and one of the read-duplicated copies was resident at that location. Otherwise, the location chosen is arbitrary. Note: The `location` argument is ignored for this advice.


  * [CU_MEM_ADVISE_SET_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3ddee285dc5e0e7d26469009ffd583cea>): This advice sets the preferred location for the data to be the memory belonging to `location`. When [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>), [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) is ignored and the preferred location is set to be host memory. To set the preferred location to a specific host NUMA node, applications must set [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) to [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) and [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) must specify the NUMA ID of the host NUMA node. If [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is set to [CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e55b82116b2124510a1a3b6c52096daaa>), [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) will be ignored and the the host NUMA node closest to the calling thread's CPU will be used as the preferred location. If [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is a [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), then [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) must be a valid device ordinal and the device must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). Setting the preferred location does not cause data to migrate to that location immediately. Instead, it guides the migration policy when a fault occurs on that memory region. If the data is already in its preferred location and the faulting processor can establish a mapping without requiring the data to be migrated, then data migration will be avoided. On the other hand, if the data is not in its preferred location or if a direct mapping cannot be established, then it will be migrated to the processor accessing it. It is important to note that setting the preferred location does not prevent data prefetching done using [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location."). Having a preferred location can override the page thrash detection and resolution logic in the Unified Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device memory, the page may eventually be pinned to host memory by the Unified Memory driver. But if the preferred location is set as device memory, then the page will continue to thrash indefinitely. If [CU_MEM_ADVISE_SET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff35a99fb44378c84c56668550b94157fc0>) is also set on this memory region or any subset of it, then the policies associated with that advice will override the policies of this advice, unless read accesses from `location` will not result in a read-only copy being created on that procesor as outlined in description for the advice [CU_MEM_ADVISE_SET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff35a99fb44378c84c56668550b94157fc0>). If the memory region refers to valid system-allocated pageable memory, and [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is CU_MEM_LOCATION_TYPE_DEVICE then [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) must be a valid device that has a non-zero alue for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>).


  * [CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff31d0d3a6b5273abd3f758d55a020bb6ca>): Undoes the effect of [CU_MEM_ADVISE_SET_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3ddee285dc5e0e7d26469009ffd583cea>) and changes the preferred location to none. The `location` argument is ignored for this advice.


  * [CU_MEM_ADVISE_SET_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3a44e5c0ad0a77d05332739848e181a2d>): This advice implies that the data will be accessed by processor `location`. The [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) must be either [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) with [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) representing a valid device ordinal or [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) and [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) will be ignored. All other location types are invalid. If [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) is a GPU, then the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) must be non-zero. This advice does not cause data migration and has no impact on the location of the data per se. Instead, it causes the data to always be mapped in the specified processor's page tables, as long as the location of the data permits a mapping to be established. If the data gets migrated for any reason, the mappings are updated accordingly. This advice is recommended in scenarios where data locality is not important, but avoiding faults is. Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data over to the other GPUs is not as important because the accesses are infrequent and the overhead of migration may be too high. But preventing faults can still help improve performance, and so having a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated to host memory because the CPU typically cannot access device memory directly. Any GPU that had the [CU_MEM_ADVISE_SET_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3a44e5c0ad0a77d05332739848e181a2d>) flag set for this data will now have its mapping updated to point to the page in host memory. If [CU_MEM_ADVISE_SET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff35a99fb44378c84c56668550b94157fc0>) is also set on this memory region or any subset of it, then the policies associated with that advice will override the policies of this advice. Additionally, if the preferred location of this memory region or any subset of it is also `location`, then the policies associated with [CU_MEM_ADVISE_SET_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3ddee285dc5e0e7d26469009ffd583cea>) will override the policies of this advice. If the memory region refers to valid system-allocated pageable memory, and [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) then device in [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>). Additionally, if [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) has a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>), then this call has no effect.


  * [CU_MEM_ADVISE_UNSET_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3f8118635c5f39d76432654ec13a726a5>): Undoes the effect of [CU_MEM_ADVISE_SET_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3a44e5c0ad0a77d05332739848e181a2d>). Any mappings to the data from `location` may be removed at any time causing accesses to result in non-fatal page faults. If the memory region refers to valid system-allocated pageable memory, and [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) then device in [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>). Additionally, if [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) has a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>), then this call has no effect.


Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuMemcpy](<group__CUDA__MEM.html#group__CUDA__MEM_1g8d0ff510f26d4b87bd3a51e731e7f698> "Copies memory."), [cuMemcpyPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a> "Copies device memory between two contexts."), [cuMemcpyAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g5f26aaf5582ade791e5688727a178d78> "Copies memory asynchronously."), [cuMemcpy3DPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8> "Copies memory between contexts asynchronously."), [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location."), [cudaMemAdvise](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g5584e2dac446bebc695da3bb1c162607>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemDiscardAndPrefetchBatchAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptrs, size_t*Â sizes, size_tÂ count, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â prefetchLocs, size_t*Â prefetchLocIdxs, size_tÂ numPrefetchLocs, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Performs a batch of memory discards and prefetches asynchronously.

######  Parameters

`dptrs`
    \- Array of pointers to be discarded
`sizes`
    \- Array of sizes for memory discard operations.
`count`
    \- Size of `dptrs` and `sizes` arrays.
`prefetchLocs`
    \- Array of locations to prefetch to.
`prefetchLocIdxs`
    \- Array of indices to specify which operands each entry in the `prefetchLocs` array applies to. The locations specified in prefetchLocs[k] will be applied to operations starting from prefetchLocIdxs[k] through prefetchLocIdxs[k+1] - 1. Also prefetchLocs[numPrefetchLocs - 1] will apply to copies starting from prefetchLocIdxs[numPrefetchLocs \- 1] through count - 1.
`numPrefetchLocs`
    \- Size of `prefetchLocs` and `prefetchLocIdxs` arrays.
`flags`
    \- Flags reserved for future use. Must be zero.
`hStream`
    \- The stream to enqueue the operations in. Must not be legacy NULL stream.

###### Description

Performs a batch of memory discards followed by prefetches. The batch as a whole executes in stream order but operations within a batch are not guaranteed to execute in any specific order. All devices in the system must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) otherwise the API will return an error.

Calling [cuMemDiscardAndPrefetchBatchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g2a8dbb3c95608cff2269226298fd8f28> "Performs a batch of memory discards and prefetches asynchronously.") is semantically equivalent to calling [cuMemDiscardBatchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gf517eb7d44e9bae70cf6a1ec8d9ece4e> "Performs a batch of memory discards asynchronously.") followed by [cuMemPrefetchBatchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g97fe632183ff5ff791813c3174fc5121> "Performs a batch of memory prefetches asynchronously."), but is more optimal. For more details on what discarding and prefetching imply, please refer to [cuMemDiscardBatchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gf517eb7d44e9bae70cf6a1ec8d9ece4e> "Performs a batch of memory discards asynchronously.") and [cuMemPrefetchBatchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g97fe632183ff5ff791813c3174fc5121> "Performs a batch of memory prefetches asynchronously.") respectively. Note that any reads, writes or prefetches to any part of the memory range that occur simultaneously with this combined discard+prefetch operation result in undefined behavior.

Performs memory discard and prefetch on address ranges specified in `dptrs` and `sizes`. Both arrays must be of the same length as specified by `count`. Each memory range specified must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") or declared via __managed__ variables or it may also refer to system-allocated memory when all devices have a non-zero value for [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>). Every operation in the batch has to be associated with a valid location to prefetch the address range to and specified in the `prefetchLocs` array. Each entry in this array can apply to more than one operation. This can be done by specifying in the `prefetchLocIdxs` array, the index of the first operation that the corresponding entry in the `prefetchLocs` array applies to. Both `prefetchLocs` and `prefetchLocIdxs` must be of the same length as specified by `numPrefetchLocs`. For example, if a batch has 10 operations listed in dptrs/sizes, the first 6 of which are to be prefetched to one location and the remaining 4 are to be prefetched to another, then `numPrefetchLocs` will be 2, `prefetchLocIdxs` will be {0, 6} and `prefetchLocs` will contain the two set of locations. Note the first entry in `prefetchLocIdxs` must always be 0. Also, each entry must be greater than the previous entry and the last entry should be less than `count`. Furthermore, `numPrefetchLocs` must be lesser than or equal to `count`.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemDiscardBatchAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptrs, size_t*Â sizes, size_tÂ count, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Performs a batch of memory discards asynchronously.

######  Parameters

`dptrs`
    \- Array of pointers to be discarded
`sizes`
    \- Array of sizes for memory discard operations.
`count`
    \- Size of `dptrs` and `sizes` arrays.
`flags`
    \- Flags reserved for future use. Must be zero.
`hStream`
    \- The stream to enqueue the operations in. Must not be legacy NULL stream.

###### Description

Performs a batch of memory discards. The batch as a whole executes in stream order but operations within a batch are not guaranteed to execute in any specific order. All devices in the system must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) otherwise the API will return an error.

Discarding a memory range informs the driver that the contents of that range are no longer useful. Discarding memory ranges allows the driver to optimize certain data migrations and can also help reduce memory pressure. This operation can be undone on any part of the range by either writing to it or prefetching it via [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location.") or [cuMemPrefetchBatchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g97fe632183ff5ff791813c3174fc5121> "Performs a batch of memory prefetches asynchronously."). Reading from a discarded range, without a subsequent write or prefetch to that part of the range, will return an indeterminate value. Note that any reads, writes or prefetches to any part of the memory range that occur simultaneously with the discard operation result in undefined behavior.

Performs memory discard on address ranges specified in `dptrs` and `sizes`. Both arrays must be of the same length as specified by `count`. Each memory range specified must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") or declared via __managed__ variables or it may also refer to system-allocated memory when all devices have a non-zero value for [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>).

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPrefetchAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)Â location, unsigned int Â flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Prefetches memory to the specified destination location.

######  Parameters

`devPtr`
    \- Pointer to be prefetched
`count`
    \- Size in bytes
`location`
    \- Location to prefetch to
`flags`
    \- flags for future use, must be zero now.
`hStream`
    \- Stream to enqueue prefetch operation

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Prefetches memory to the specified destination location. `devPtr` is the base device pointer of the memory to be prefetched and `location` specifies the destination location. `count` specifies the number of bytes to copy. `hStream` is the stream in which the operation is enqueued. The memory range must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."), via cuMemAllocFromPool from a managed memory pool or declared via __managed__ variables.

Specifying [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) for [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) will prefetch memory to GPU specified by device ordinal [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) which must have non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). Additionally, `hStream` must be associated with a device that has a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). Specifying [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) as [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) will prefetch data to host memory. Applications can request prefetching memory to a specific host NUMA node by specifying [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) for [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) and a valid host NUMA node id in [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) Users can also request prefetching memory to the host NUMA node closest to the current thread's CPU by specifying [CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e55b82116b2124510a1a3b6c52096daaa>) for [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>). Note when [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) is etiher [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) OR [CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e55b82116b2124510a1a3b6c52096daaa>), [CUmemLocation::id](<structCUmemLocation__v1.html#structCUmemLocation__v1_11d7b65b482228d640b6f953196e460dc>) will be ignored.

The start address and end address of the memory range will be rounded down and rounded up respectively to be aligned to CPU page size before the prefetch operation is enqueued in the stream.

If no physical memory has been allocated for this region, then this memory region will be populated and mapped on the destination device. If there's insufficient memory to prefetch the desired region, the Unified Memory driver may evict pages from other [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") allocations to host memory in order to make room. Device memory allocated using [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") or [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array.") will not be evicted.

By default, any mappings to the previous location of the migrated pages are removed and mappings for the new location are only setup on the destination location. The exact behavior however also depends on the settings applied to this memory range via [cuMemAdvise](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaac8924b2f5a2a93f8775fb81c1a643f> "Advise about the usage of a given memory range.") as described below:

If [CU_MEM_ADVISE_SET_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff35a99fb44378c84c56668550b94157fc0>) was set on any subset of this memory range, then that subset will create a read-only copy of the pages on destination location. If however the destination location is a host NUMA node, then any pages of that subset that are already in another host NUMA node will be transferred to the destination.

If [CU_MEM_ADVISE_SET_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3ddee285dc5e0e7d26469009ffd583cea>) was called on any subset of this memory range, then the pages will be migrated to `location` even if `location` is not the preferred location of any pages in the memory range.

If [CU_MEM_ADVISE_SET_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3a44e5c0ad0a77d05332739848e181a2d>) was called on any subset of this memory range, then mappings to those pages from all the appropriate processors are updated to refer to the new location if establishing such a mapping is possible. Otherwise, those mappings are cleared.

Note that this API is not required for functionality and only serves to improve performance by allowing the application to migrate data to a suitable location before it is accessed. Memory accesses to this range are always coherent and are allowed even when the data is actively being migrated.

Note that this function is asynchronous with respect to the host and all work on other devices.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuMemcpy](<group__CUDA__MEM.html#group__CUDA__MEM_1g8d0ff510f26d4b87bd3a51e731e7f698> "Copies memory."), [cuMemcpyPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a> "Copies device memory between two contexts."), [cuMemcpyAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g5f26aaf5582ade791e5688727a178d78> "Copies memory asynchronously."), [cuMemcpy3DPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8> "Copies memory between contexts asynchronously."), [cuMemAdvise](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaac8924b2f5a2a93f8775fb81c1a643f> "Advise about the usage of a given memory range."), [cudaMemPrefetchAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g856fa41c8c0d28655e37b778cb9ffc65>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemPrefetchBatchAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptrs, size_t*Â sizes, size_tÂ count, [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â prefetchLocs, size_t*Â prefetchLocIdxs, size_tÂ numPrefetchLocs, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Performs a batch of memory prefetches asynchronously.

######  Parameters

`dptrs`
    \- Array of pointers to be prefetched
`sizes`
    \- Array of sizes for memory prefetch operations.
`count`
    \- Size of `dptrs` and `sizes` arrays.
`prefetchLocs`
    \- Array of locations to prefetch to.
`prefetchLocIdxs`
    \- Array of indices to specify which operands each entry in the `prefetchLocs` array applies to. The locations specified in prefetchLocs[k] will be applied to copies starting from prefetchLocIdxs[k] through prefetchLocIdxs[k+1] - 1. Also prefetchLocs[numPrefetchLocs - 1] will apply to prefetches starting from prefetchLocIdxs[numPrefetchLocs \- 1] through count - 1.
`numPrefetchLocs`
    \- Size of `prefetchLocs` and `prefetchLocIdxs` arrays.
`flags`
    \- Flags reserved for future use. Must be zero.
`hStream`
    \- The stream to enqueue the operations in. Must not be legacy NULL stream.

###### Description

Performs a batch of memory prefetches. The batch as a whole executes in stream order but operations within a batch are not guaranteed to execute in any specific order. All devices in the system must have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) otherwise the API will return an error.

The semantics of the individual prefetch operations are as described in [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location.").

Performs memory prefetch on address ranges specified in `dptrs` and `sizes`. Both arrays must be of the same length as specified by `count`. Each memory range specified must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") or declared via __managed__ variables or it may also refer to system-allocated memory when all devices have a non-zero value for [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>). The prefetch location for every operation in the batch is specified in the `prefetchLocs` array. Each entry in this array can apply to more than one operation. This can be done by specifying in the `prefetchLocIdxs` array, the index of the first prefetch operation that the corresponding entry in the `prefetchLocs` array applies to. Both `prefetchLocs` and `prefetchLocIdxs` must be of the same length as specified by `numPrefetchLocs`. For example, if a batch has 10 prefetches listed in dptrs/sizes, the first 4 of which are to be prefetched to one location and the remaining 6 are to be prefetched to another, then `numPrefetchLocs` will be 2, `prefetchLocIdxs` will be {0, 4} and `prefetchLocs` will contain the two locations. Note the first entry in `prefetchLocIdxs` must always be 0. Also, each entry must be greater than the previous entry and the last entry should be less than `count`. Furthermore, `numPrefetchLocs` must be lesser than or equal to `count`.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemRangeGetAttribute ( void*Â data, size_tÂ dataSize, [CUmem_range_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3754eb90f64ed3c19b4e550d21d124fc>)Â attribute, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count )


Query an attribute of a given memory range.

######  Parameters

`data`
    \- A pointers to a memory location where the result of each attribute query will be written to.
`dataSize`
    \- Array containing the size of data
`attribute`
    \- The attribute to query
`devPtr`
    \- Start of the range to query
`count`
    \- Size of the range to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Query an attribute about the memory range starting at `devPtr` with a size of `count` bytes. The memory range must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") or declared via __managed__ variables.

The `attribute` parameter can take the following values:

  * [CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fcb21856250c588cd795462323c71fef7b>): If this attribute is specified, `data` will be interpreted as a 32-bit integer, and `dataSize` must be 4. The result returned will be 1 if all pages in the given memory range have read-duplication enabled, or 0 otherwise.

  * [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fcc68a2f5771f8e1b83ad29c1d65ab4875>): If this attribute is specified, `data` will be interpreted as a 32-bit integer, and `dataSize` must be 4. The result returned will be a GPU device id if all pages in the memory range have that GPU as their preferred location, or it will be CU_DEVICE_CPU if all pages in the memory range have the CPU as their preferred location, or it will be CU_DEVICE_INVALID if either all the pages don't have the same preferred location or some of the pages don't have a preferred location at all. Note that the actual location of the pages in the memory range at the time of the query may be different from the preferred location.

  * [CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc2511c59b3d309b83f410af68280891a4>): If this attribute is specified, `data` will be interpreted as an array of 32-bit integers, and `dataSize` must be a non-zero multiple of 4. The result returned will be a list of device ids that had [CU_MEM_ADVISE_SET_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcfe2ed2d4567745dd4ad41034136fff3a44e5c0ad0a77d05332739848e181a2d>) set for that entire memory range. If any device does not have that advice set for the entire memory range, that device will not be included. If `data` is larger than the number of devices that have that advice set for that memory range, CU_DEVICE_INVALID will be returned in all the extra space provided. For ex., if `dataSize` is 12 (i.e. `data` has 3 elements) and only device 0 has the advice set, then the result returned will be { 0, CU_DEVICE_INVALID, CU_DEVICE_INVALID }. If `data` is smaller than the number of devices that have that advice set, then only as many devices will be returned as can fit in the array. There is no guarantee on which specific devices will be returned, however.

  * [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc5213666bebedfcea6128fbebaa2f7ade>): If this attribute is specified, `data` will be interpreted as a 32-bit integer, and `dataSize` must be 4. The result returned will be the last location to which all pages in the memory range were prefetched explicitly via [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location."). This will either be a GPU id or CU_DEVICE_CPU depending on whether the last location for prefetch was a GPU or the CPU respectively. If any page in the memory range was never explicitly prefetched or if all pages were not prefetched to the same location, CU_DEVICE_INVALID will be returned. Note that this simply returns the last location that the application requested to prefetch the memory range to. It gives no indication as to whether the prefetch operation to that location has completed or even begun.

  * [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc4403ec1d9f21a5fd1d6eb7745d0aa5c3>): If this attribute is specified, `data` will be interpreted as a [CUmemLocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g75cfd5b9fa5c1c6ee2be2547bfbe882e>), and `dataSize` must be sizeof(CUmemLocationType). The [CUmemLocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g75cfd5b9fa5c1c6ee2be2547bfbe882e>) returned will be [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) if all pages in the memory range have the same GPU as their preferred location, or [CUmemLocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g75cfd5b9fa5c1c6ee2be2547bfbe882e>) will be [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) if all pages in the memory range have the CPU as their preferred location, or it will be [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) if all the pages in the memory range have the same host NUMA node ID as their preferred location or it will be CU_MEM_LOCATION_TYPE_INVALID if either all the pages don't have the same preferred location or some of the pages don't have a preferred location at all. Note that the actual location type of the pages in the memory range at the time of the query may be different from the preferred location type.
    * [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc7c9808f73124e7e13dfa7d0138fc3b3e>): If this attribute is specified, `data` will be interpreted as a 32-bit integer, and `dataSize` must be 4. If the [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc4403ec1d9f21a5fd1d6eb7745d0aa5c3>) query for the same address range returns [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), it will be a valid device ordinal or if it returns [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>), it will be a valid host NUMA node ID or if it returns any other location type, the id should be ignored.

  * [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc74331e0361af97f526652db843f705f8>): If this attribute is specified, `data` will be interpreted as a [CUmemLocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g75cfd5b9fa5c1c6ee2be2547bfbe882e>), and `dataSize` must be sizeof(CUmemLocationType). The result returned will be the last location to which all pages in the memory range were prefetched explicitly via [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location."). The [CUmemLocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g75cfd5b9fa5c1c6ee2be2547bfbe882e>) returned will be [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>) if the last prefetch location was a GPU or [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>) if it was the CPU or [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) if the last prefetch location was a specific host NUMA node. If any page in the memory range was never explicitly prefetched or if all pages were not prefetched to the same location, [CUmemLocationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g75cfd5b9fa5c1c6ee2be2547bfbe882e>) will be CU_MEM_LOCATION_TYPE_INVALID. Note that this simply returns the last location type that the application requested to prefetch the memory range to. It gives no indication as to whether the prefetch operation to that location has completed or even begun.
    * [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fcb31ca7fca0ad44f236bfed3e33790b35>): If this attribute is specified, `data` will be interpreted as a 32-bit integer, and `dataSize` must be 4. If the [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc74331e0361af97f526652db843f705f8>) query for the same address range returns [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), it will be a valid device ordinal or if it returns [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>), it will be a valid host NUMA node ID or if it returns any other location type, the id should be ignored.


Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuMemRangeGetAttributes](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gc7ce142e60f8613cfb7d722b87dc9d12> "Query attributes of a given memory range."), [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location."), [cuMemAdvise](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaac8924b2f5a2a93f8775fb81c1a643f> "Advise about the usage of a given memory range."), [cudaMemRangeGetAttribute](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8048f6ea5ad77917444567656c140c5a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemRangeGetAttributes ( void**Â data, size_t*Â dataSizes, [CUmem_range_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3754eb90f64ed3c19b4e550d21d124fc>)*Â attributes, size_tÂ numAttributes, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â devPtr, size_tÂ count )


Query attributes of a given memory range.

######  Parameters

`data`
    \- A two-dimensional array containing pointers to memory locations where the result of each attribute query will be written to.
`dataSizes`
    \- Array containing the sizes of each result
`attributes`
    \- An array of attributes to query (numAttributes and the number of attributes in this array should match)
`numAttributes`
    \- Number of attributes to query
`devPtr`
    \- Start of the range to query
`count`
    \- Size of the range to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Query attributes of the memory range starting at `devPtr` with a size of `count` bytes. The memory range must refer to managed memory allocated via [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") or declared via __managed__ variables. The `attributes` array will be interpreted to have `numAttributes` entries. The `dataSizes` array will also be interpreted to have `numAttributes` entries. The results of the query will be stored in `data`.

The list of supported attributes are given below. Please refer to [cuMemRangeGetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g1c92408a7d0d8875e19b1a58af56f67d> "Query an attribute of a given memory range.") for attribute descriptions and restrictions.

  * [CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fcb21856250c588cd795462323c71fef7b>)

  * [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fcc68a2f5771f8e1b83ad29c1d65ab4875>)

  * [CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc2511c59b3d309b83f410af68280891a4>)

  * [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc5213666bebedfcea6128fbebaa2f7ade>)

  * [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc4403ec1d9f21a5fd1d6eb7745d0aa5c3>)

  * [CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc7c9808f73124e7e13dfa7d0138fc3b3e>)

  * [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fc74331e0361af97f526652db843f705f8>)

  * [CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3754eb90f64ed3c19b4e550d21d124fcb31ca7fca0ad44f236bfed3e33790b35>)


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemRangeGetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g1c92408a7d0d8875e19b1a58af56f67d> "Query an attribute of a given memory range."), [cuMemAdvise](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaac8924b2f5a2a93f8775fb81c1a643f> "Advise about the usage of a given memory range."), [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location."), [cudaMemRangeGetAttributes](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g1a9199e7709c7817d1c715cfbe174d05>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuPointerGetAttribute ( void*Â data, [CUpointer_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc2cce590e35080745e72633dfc6e0b60>)Â attribute, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )


Returns information about a pointer.

######  Parameters

`data`
    \- Returned pointer attribute value
`attribute`
    \- Pointer attribute to query
`ptr`
    \- Pointer

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

The supported attributes are:

  * [CU_POINTER_ATTRIBUTE_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60f0470fdbd1a5ff72c341f762f49506ab>):


Returns in `*data` the [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) in which `ptr` was allocated or registered. The type of `data` must be [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) *.

If `ptr` was not allocated by, mapped by, or registered with a [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) which uses unified virtual addressing then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

  * [CU_POINTER_ATTRIBUTE_MEMORY_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b600409e16293b60b383f30a9b417b2917c>):


Returns in `*data` the physical memory type of the memory that `ptr` addresses as a [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) enumerated value. The type of `data` must be unsigned int.

If `ptr` addresses device memory then `*data` is set to [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>). The particular [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) on which the memory resides is the [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) of the [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) returned by the [CU_POINTER_ATTRIBUTE_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60f0470fdbd1a5ff72c341f762f49506ab>) attribute of `ptr`.

If `ptr` addresses host memory then `*data` is set to [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>).

If `ptr` was not allocated by, mapped by, or registered with a [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) which uses unified virtual addressing then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

If the current [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) does not support unified virtual addressing then [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned.

  * [CU_POINTER_ATTRIBUTE_DEVICE_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60b5446064bbfa484ea8d13025f1573d5d>):


Returns in `*data` the device pointer value through which `ptr` may be accessed by kernels running in the current [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>). The type of `data` must be CUdeviceptr *.

If there exists no device pointer value through which kernels running in the current [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) may access `ptr` then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

If there is no current [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) then [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned.

Except in the exceptional disjoint addressing cases discussed below, the value returned in `*data` will equal the input value `ptr`.

  * [CU_POINTER_ATTRIBUTE_HOST_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60ab17d9902b1b631982ae6a3a9a436fdc>):


Returns in `*data` the host pointer value through which `ptr` may be accessed by by the host program. The type of `data` must be void **. If there exists no host pointer value through which the host program may directly access `ptr` then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Except in the exceptional disjoint addressing cases discussed below, the value returned in `*data` will equal the input value `ptr`.

  * [CU_POINTER_ATTRIBUTE_P2P_TOKENS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60995003218508cc15bfcf197aa9b30a1b>):


Returns in `*data` two tokens for use with the nv-p2p.h Linux kernel interface. `data` must be a struct of type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS.

`ptr` must be a pointer to memory obtained from :[cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."). Note that p2pToken and vaSpaceToken are only valid for the lifetime of the source allocation. A subsequent allocation at the same address may return completely different tokens. Querying this attribute has a side effect of setting the attribute [CU_POINTER_ATTRIBUTE_SYNC_MEMOPS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60db5380c3201afdae3556cce8834504e1>) for the region of memory that `ptr` points to.

  * [CU_POINTER_ATTRIBUTE_SYNC_MEMOPS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60db5380c3201afdae3556cce8834504e1>):


A boolean attribute which when set, ensures that synchronous memory operations initiated on the region of memory that `ptr` points to will always synchronize. See further documentation in the section titled "API synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior.

  * [CU_POINTER_ATTRIBUTE_BUFFER_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60d891336d10c16c3ea8140fa581c5446f>):


Returns in `*data` a buffer ID which is guaranteed to be unique within the process. `data` must point to an unsigned long long.

`ptr` must be a pointer to memory obtained from a CUDA memory allocation API. Every memory allocation from any of the CUDA memory allocation APIs will have a unique ID over a process lifetime. Subsequent allocations do not reuse IDs from previous freed allocations. IDs are only unique within a single process.

  * [CU_POINTER_ATTRIBUTE_IS_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60893cfce785d799fb93826537fbb72a1d>):


Returns in `*data` a boolean that indicates whether the pointer points to managed memory or not.

If `ptr` is not a valid CUDA pointer then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

  * [CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60cb7636b0198450ddb390ab87e98d83a0>):


Returns in `*data` an integer representing a device ordinal of a device against which the memory was allocated or registered.

  * [CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60b9f25a1b90274b624254eaaf836102fc>):


Returns in `*data` a boolean that indicates if this pointer maps to an allocation that is suitable for [cudaIpcGetMemHandle](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539>).

  * [CU_POINTER_ATTRIBUTE_RANGE_START_ADDR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b6060f2fa33283dda44d552e828bdcc4cdc>):


Returns in `*data` the starting address for the allocation referenced by the device pointer `ptr`. Note that this is not necessarily the address of the mapped region, but the address of the mappable address range `ptr` references (e.g. from [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation.")).

  * [CU_POINTER_ATTRIBUTE_RANGE_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60339f4a7e16696c0a2a4bcc4acf394ec4>):


Returns in `*data` the size for the allocation referenced by the device pointer `ptr`. Note that this is not necessarily the size of the mapped region, but the size of the mappable address range `ptr` references (e.g. from [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation.")). To retrieve the size of the mapped region, see [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations.")

  * [CU_POINTER_ATTRIBUTE_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b600bae89da103451e804fc005fb5f3ea78>):


Returns in `*data` a boolean that indicates if this pointer is in a valid address range that is mapped to a backing allocation.

  * [CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60a1f518cc4336f0f286b36a748ad91a0e>):


Returns a bitmask of the allowed handle types for an allocation that may be passed to [cuMemExportToShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56> "Exports an allocation to a requested shareable handle type.").

  * [CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b6023a9869c7f811f700ed67dfd762c26e0>):


Returns in `*data` the handle to the mempool that the allocation was obtained from.

  * [CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60820cbe95f809088e8045fcb1c9857bf5>):


Returns in `*data` a boolean that indicates whether the pointer points to memory that is capable to be used for hardware accelerated decompression.

Note that for most allocations in the unified virtual address space the host and device pointer for accessing the allocation will be the same. The exceptions to this are

  * user memory registered using [cuMemHostRegister](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA.")

  * host memory allocated using [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") with the [CU_MEMHOSTALLOC_WRITECOMBINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7361580951deecace15352c97a210038>) flag For these types of allocation there will exist separate, disjoint host and device addresses for accessing the allocation. In particular

  * The host address will correspond to an invalid unmapped device address (which will result in an exception if accessed from the device)

  * The device address will correspond to an invalid unmapped host address (which will result in an exception if accessed from the host). For these types of allocations, querying [CU_POINTER_ATTRIBUTE_HOST_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60ab17d9902b1b631982ae6a3a9a436fdc>) and [CU_POINTER_ATTRIBUTE_DEVICE_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60b5446064bbfa484ea8d13025f1573d5d>) may be used to retrieve the host and device addresses from either address.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuPointerSetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g89f7ad29a657e574fdea2624b74d138e> "Set attributes on a previously allocated memory region."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostRegister](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA."), [cuMemHostUnregister](<group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14> "Unregisters a memory range that was registered with cuMemHostRegister."), [cudaPointerGetAttributes](<../cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED_1gd89830e17d399c064a2f3c3fa8bb4390>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuPointerGetAttributes ( unsigned int Â numAttributes, [CUpointer_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc2cce590e35080745e72633dfc6e0b60>)*Â attributes, void**Â data, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )


Returns information about a pointer.

######  Parameters

`numAttributes`
    \- Number of attributes to query
`attributes`
    \- An array of attributes to query (numAttributes and the number of attributes in this array should match)
`data`
    \- A two-dimensional array containing pointers to memory locations where the result of each attribute query will be written to.
`ptr`
    \- Pointer to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

The supported attributes are (refer to [cuPointerGetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c> "Returns information about a pointer.") for attribute descriptions and restrictions):

  * [CU_POINTER_ATTRIBUTE_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60f0470fdbd1a5ff72c341f762f49506ab>)

  * [CU_POINTER_ATTRIBUTE_MEMORY_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b600409e16293b60b383f30a9b417b2917c>)

  * [CU_POINTER_ATTRIBUTE_DEVICE_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60b5446064bbfa484ea8d13025f1573d5d>)

  * [CU_POINTER_ATTRIBUTE_HOST_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60ab17d9902b1b631982ae6a3a9a436fdc>)

  * [CU_POINTER_ATTRIBUTE_SYNC_MEMOPS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60db5380c3201afdae3556cce8834504e1>)

  * [CU_POINTER_ATTRIBUTE_BUFFER_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60d891336d10c16c3ea8140fa581c5446f>)

  * [CU_POINTER_ATTRIBUTE_IS_MANAGED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60893cfce785d799fb93826537fbb72a1d>)

  * [CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60cb7636b0198450ddb390ab87e98d83a0>)

  * [CU_POINTER_ATTRIBUTE_RANGE_START_ADDR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b6060f2fa33283dda44d552e828bdcc4cdc>)

  * [CU_POINTER_ATTRIBUTE_RANGE_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60339f4a7e16696c0a2a4bcc4acf394ec4>)

  * [CU_POINTER_ATTRIBUTE_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b600bae89da103451e804fc005fb5f3ea78>)

  * [CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60b9f25a1b90274b624254eaaf836102fc>)

  * [CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60a1f518cc4336f0f286b36a748ad91a0e>)

  * [CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b6023a9869c7f811f700ed67dfd762c26e0>)

  * [CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60820cbe95f809088e8045fcb1c9857bf5>)


Unlike [cuPointerGetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c> "Returns information about a pointer."), this function will not return an error when the `ptr` encountered is not a valid CUDA pointer. Instead, the attributes are assigned default NULL values and CUDA_SUCCESS is returned.

If `ptr` was not allocated by, mapped by, or registered with a [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) which uses UVA (Unified Virtual Addressing), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuPointerGetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c> "Returns information about a pointer."), [cuPointerSetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g89f7ad29a657e574fdea2624b74d138e> "Set attributes on a previously allocated memory region."), [cudaPointerGetAttributes](<../cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED_1gd89830e17d399c064a2f3c3fa8bb4390>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuPointerSetAttribute ( const void*Â value, [CUpointer_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc2cce590e35080745e72633dfc6e0b60>)Â attribute, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )


Set attributes on a previously allocated memory region.

######  Parameters

`value`
    \- Pointer to memory containing the value to be set
`attribute`
    \- Pointer attribute to set
`ptr`
    \- Pointer to a memory region allocated using CUDA memory allocation APIs

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

The supported attributes are:

  * [CU_POINTER_ATTRIBUTE_SYNC_MEMOPS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60db5380c3201afdae3556cce8834504e1>):


A boolean attribute that can either be set (1) or unset (0). When set, the region of memory that `ptr` points to is guaranteed to always synchronize memory operations that are synchronous. If there are some previously initiated synchronous memory operations that are pending when this attribute is set, the function does not return until those memory operations are complete. See further documentation in the section titled "API synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior. `value` will be considered as a pointer to an unsigned integer to which this attribute is to be set.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuPointerGetAttribute](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c> "Returns information about a pointer."), [cuPointerGetAttributes](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gf65e9ea532e311dd049166e4894955ad> "Returns information about a pointer."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostRegister](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA."), [cuMemHostUnregister](<group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14> "Unregisters a memory range that was registered with cuMemHostRegister.")

* * *
