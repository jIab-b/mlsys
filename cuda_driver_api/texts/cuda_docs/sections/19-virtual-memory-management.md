# Virtual Memory Management

## 6.14.Â Virtual Memory Management

This section describes the virtual memory management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAddressFree](<#group__CUDA__VA_1g6993ecea2ea03e1b802b8255edc2da5b>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size )
     Free an address range reservation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAddressReserve](<#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â ptr, size_tÂ size, size_tÂ alignment, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â addr, unsigned long longÂ flags )
     Allocate an address range reservation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemCreate](<#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c>) ( CUmemGenericAllocationHandle*Â handle, size_tÂ size, const [CUmemAllocationProp](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1>)*Â prop, unsigned long longÂ flags )
     Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemExportToShareableHandle](<#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56>) ( void*Â shareableHandle, CUmemGenericAllocationHandleÂ handle, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â handleType, unsigned long longÂ flags )
     Exports an allocation to a requested shareable handle type.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetAccess](<#group__CUDA__VA_1g4b5627b4f2d3972d0b62cc4ba1931125>) ( unsigned long long*Â flags, const [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )
     Get the access `flags` set for the given `location` and `ptr`.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetAllocationGranularity](<#group__CUDA__VA_1g30ee906c2cf66a0347b3dfec3d7eb31a>) ( size_t*Â granularity, const [CUmemAllocationProp](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1>)*Â prop, [CUmemAllocationGranularity_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3a202e4d32ae296db1af7efe75ce365d>)Â option )
     Calculates either the minimal or recommended granularity.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetAllocationPropertiesFromHandle](<#group__CUDA__VA_1gc1c4c812caba5a21401c2cb4ab4512b1>) ( [CUmemAllocationProp](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1>)*Â prop, CUmemGenericAllocationHandleÂ handle )
     Retrieve the contents of the property structure defining properties for this handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemImportFromShareableHandle](<#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463>) ( CUmemGenericAllocationHandle*Â handle, void*Â osHandle, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â shHandleType )
     Imports an allocation from a requested shareable handle type.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemMap](<#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size, size_tÂ offset, CUmemGenericAllocationHandleÂ handle, unsigned long longÂ flags )
     Maps an allocation handle to a reserved virtual address range.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemMapArrayAsync](<#group__CUDA__VA_1g5dc41a62a9feb68f2e943b438c83e5ab>) ( [CUarrayMapInfo](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1>)*Â mapInfoList, unsigned int Â count, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemRelease](<#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68>) ( CUmemGenericAllocationHandleÂ handle )
     Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemRetainAllocationHandle](<#group__CUDA__VA_1g1ddca5437c502782155f95bf98e775c6>) ( CUmemGenericAllocationHandle*Â handle, void*Â addr )
     Given an address `addr`, returns the allocation handle of the backing memory allocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemSetAccess](<#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size, const [CUmemAccessDesc](<structCUmemAccessDesc__v1.html#structCUmemAccessDesc__v1>)*Â desc, size_tÂ count )
     Set the access flags for each location specified in `desc` for the given virtual address range.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemUnmap](<#group__CUDA__VA_1gfb50aac00c848fd7087e858f59bf7e2a>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size )
     Unmap the backing memory of a given address range.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAddressFree ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size )


Free an address range reservation.

######  Parameters

`ptr`
    \- Starting address of the virtual address range to free
`size`
    \- Size of the virtual address region to free

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Frees a virtual address range reserved by cuMemAddressReserve. The size must match what was given to memAddressReserve and the ptr given must match what was returned from memAddressReserve.

**See also:**

[cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAddressReserve ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â ptr, size_tÂ size, size_tÂ alignment, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â addr, unsigned long longÂ flags )


Allocate an address range reservation.

######  Parameters

`ptr`
    \- Resulting pointer to start of virtual address range allocated
`size`
    \- Size of the reserved virtual address range requested
`alignment`
    \- Alignment of the reserved virtual address range requested
`addr`
    \- Hint address for the start of the address range
`flags`
    \- Currently unused, must be zero

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Reserves a virtual address range based on the given parameters, giving the starting address of the range in `ptr`. This API requires a system that supports UVA. The size and address parameters must be a multiple of the host page size and the alignment must be a power of two or zero for default alignment. If `addr` is 0, then the driver chooses the address at which to place the start of the reservation whereas when it is non-zero then the driver treats it as a hint about where to place the reservation.

**See also:**

[cuMemAddressFree](<group__CUDA__VA.html#group__CUDA__VA_1g6993ecea2ea03e1b802b8255edc2da5b> "Free an address range reservation.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemCreate ( CUmemGenericAllocationHandle*Â handle, size_tÂ size, const [CUmemAllocationProp](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1>)*Â prop, unsigned long longÂ flags )


Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.

######  Parameters

`handle`
    \- Value of handle returned. All operations on this allocation are to be performed using this handle.
`size`
    \- Size of the allocation requested
`prop`
    \- Properties of the allocation to create.
`flags`
    \- flags for future use, must be zero now.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

This creates a memory allocation on the target device specified through the `prop` structure. The created allocation will not have any device or host mappings. The generic memory `handle` for the allocation can be mapped to the address space of calling process via [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."). This handle cannot be transmitted directly to other processes (see [cuMemExportToShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56> "Exports an allocation to a requested shareable handle type.")). On Windows, the caller must also pass an LPSECURITYATTRIBUTE in `prop` to be associated with this handle which limits or allows access to this handle for a recipient process (see [CUmemAllocationProp::win32HandleMetaData](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1_1542262ad88e1d00f02b306c641270168>) for more). The `size` of this allocation must be a multiple of the the value given via [cuMemGetAllocationGranularity](<group__CUDA__VA.html#group__CUDA__VA_1g30ee906c2cf66a0347b3dfec3d7eb31a> "Calculates either the minimal or recommended granularity.") with the [CU_MEM_ALLOC_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3a202e4d32ae296db1af7efe75ce365dc74872d07341bb1ac24ccc4a1c9c2f56>) flag. To create a CPU allocation that doesn't target any specific NUMA nodes, applications must set CUmemAllocationProp::CUmemLocation::type to [CU_MEM_LOCATION_TYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e46800776121a71c8dc2904518a21065a>). CUmemAllocationProp::CUmemLocation::id is ignored for HOST allocations. HOST allocations are not IPC capable and [CUmemAllocationProp::requestedHandleTypes](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1_1e2e852e72e5d2053b771fbac49495efd>) must be 0, any other value will result in [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>). To create a CPU allocation targeting a specific host NUMA node, applications must set CUmemAllocationProp::CUmemLocation::type to [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) and CUmemAllocationProp::CUmemLocation::id must specify the NUMA ID of the CPU. On systems where NUMA is not available CUmemAllocationProp::CUmemLocation::id must be set to 0. Specifying [CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882e55b82116b2124510a1a3b6c52096daaa>) as the [CUmemLocation::type](<structCUmemLocation__v1.html#structCUmemLocation__v1_1a34fc29f2a55d501f00f912d92152d1b>) will result in [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

Applications that intend to use [CU_MEM_HANDLE_TYPE_FABRIC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg450a23153d86fce0afe30e25d63caef9e38a182adb450da6c1a3f29cd5dca032>) based memory sharing must ensure: (1) `nvidia-caps-imex-channels` character device is created by the driver and is listed under /proc/devices (2) have at least one IMEX channel file accessible by the user launching the application.

When exporter and importer CUDA processes have been granted access to the same IMEX channel, they can securely share memory.

The IMEX channel security model works on a per user basis. Which means all processes under a user can share memory if the user has access to a valid IMEX channel. When multi-user isolation is desired, a separate IMEX channel is required for each user.

These channel files exist in /dev/nvidia-caps-imex-channels/channel* and can be created using standard OS native calls like mknod on Linux. For example: To create channel0 with the major number from /proc/devices users can execute the following command: `mknod /dev/nvidia-caps-imex-channels/channel0 c <major number>=""> 0`

If CUmemAllocationProp::allocFlags::usage contains [CU_MEM_CREATE_USAGE_TILE_POOL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb445d91d9773d728b1a9962184c05799>) flag then the memory allocation is intended only to be used as backing tile pool for sparse CUDA arrays and sparse CUDA mipmapped arrays. (see [cuMemMapArrayAsync](<group__CUDA__VA.html#group__CUDA__VA_1g5dc41a62a9feb68f2e943b438c83e5ab> "Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.")).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate."), [cuMemExportToShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56> "Exports an allocation to a requested shareable handle type."), [cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemExportToShareableHandle ( void*Â shareableHandle, CUmemGenericAllocationHandleÂ handle, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â handleType, unsigned long longÂ flags )


Exports an allocation to a requested shareable handle type.

######  Parameters

`shareableHandle`
    \- Pointer to the location in which to store the requested handle type
`handle`
    \- CUDA handle for the memory allocation
`handleType`
    \- Type of shareable handle requested (defines type and size of the `shareableHandle` output parameter)
`flags`
    \- Reserved, must be zero

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Given a CUDA memory handle, create a shareable memory allocation handle that can be used to share the memory with other processes. The recipient process can convert the shareable handle back into a CUDA memory handle using [cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type.") and map it with [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."). The implementation of what this handle is and how it can be transferred is defined by the requested handle type in `handleType`

Once all shareable handles are closed and the allocation is released, the allocated memory referenced will be released back to the OS and uses of the CUDA handle afterward will lead to undefined behavior.

This API can also be used in conjunction with other APIs (e.g. Vulkan, OpenGL) that support importing memory from the shareable type

**See also:**

[cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetAccess ( unsigned long long*Â flags, const [CUmemLocation](<structCUmemLocation__v1.html#structCUmemLocation__v1>)*Â location, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr )


Get the access `flags` set for the given `location` and `ptr`.

######  Parameters

`flags`
    \- Flags set for this location
`location`
    \- Location in which to check the flags for
`ptr`
    \- Address in which to check the access flags for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

**See also:**

[cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetAllocationGranularity ( size_t*Â granularity, const [CUmemAllocationProp](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1>)*Â prop, [CUmemAllocationGranularity_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3a202e4d32ae296db1af7efe75ce365d>)Â option )


Calculates either the minimal or recommended granularity.

######  Parameters

`granularity`
    Returned granularity.
`prop`
    Property for which to determine the granularity for
`option`
    Determines which granularity to return

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Calculates either the minimal or recommended granularity for a given allocation specification and returns it in granularity. This granularity can be used as a multiple for alignment, size, or address mapping.

**See also:**

[cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetAllocationPropertiesFromHandle ( [CUmemAllocationProp](<structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1>)*Â prop, CUmemGenericAllocationHandleÂ handle )


Retrieve the contents of the property structure defining properties for this handle.

######  Parameters

`prop`
    \- Pointer to a properties structure which will hold the information about this handle
`handle`
    \- Handle which to perform the query on

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

**See also:**

[cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemImportFromShareableHandle ( CUmemGenericAllocationHandle*Â handle, void*Â osHandle, [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>)Â shHandleType )


Imports an allocation from a requested shareable handle type.

######  Parameters

`handle`
    \- CUDA Memory handle for the memory allocation.
`osHandle`
    \- Shareable Handle representing the memory allocation that is to be imported.
`shHandleType`
    \- handle type of the exported handle [CUmemAllocationHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g450a23153d86fce0afe30e25d63caef9>).

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

If the current process cannot support the memory described by this shareable handle, this API will error as [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>).

If `shHandleType` is [CU_MEM_HANDLE_TYPE_FABRIC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg450a23153d86fce0afe30e25d63caef9e38a182adb450da6c1a3f29cd5dca032>) and the importer process has not been granted access to the same IMEX channel as the exporter process, this API will error as [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>).

Note:

Importing shareable handles exported from some graphics APIs(VUlkan, OpenGL, etc) created on devices under an SLI group may not be supported, and thus this API will return CUDA_ERROR_NOT_SUPPORTED. There is no guarantee that the contents of `handle` will be the same CUDA memory handle for the same given OS shareable handle, or the same underlying allocation.

**See also:**

[cuMemExportToShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56> "Exports an allocation to a requested shareable handle type."), [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."), [cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemMap ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size, size_tÂ offset, CUmemGenericAllocationHandleÂ handle, unsigned long longÂ flags )


Maps an allocation handle to a reserved virtual address range.

######  Parameters

`ptr`
    \- Address where memory will be mapped.
`size`
    \- Size of the memory mapping.
`offset`
    \- Offset into the memory represented by

  * `handle` from which to start mapping
  * Note: currently must be zero.


`handle`
    \- Handle to a shareable memory
`flags`
    \- flags for future use, must be zero now.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>)

###### Description

Maps bytes of memory represented by `handle` starting from byte `offset` to `size` to address range [`addr`, `addr` \+ `size`]. This range must be an address reservation previously reserved with [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation."), and `offset` \+ `size` must be less than the size of the memory allocation. Both `ptr`, `size`, and `offset` must be a multiple of the value given via [cuMemGetAllocationGranularity](<group__CUDA__VA.html#group__CUDA__VA_1g30ee906c2cf66a0347b3dfec3d7eb31a> "Calculates either the minimal or recommended granularity.") with the [CU_MEM_ALLOC_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3a202e4d32ae296db1af7efe75ce365dc74872d07341bb1ac24ccc4a1c9c2f56>) flag. If `handle` represents a multicast object, `ptr`, `size` and `offset` must be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag CU_MULTICAST_MINIMUM_GRANULARITY. For best performance however, it is recommended that `ptr`, `size` and `offset` be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag CU_MULTICAST_RECOMMENDED_GRANULARITY.

When `handle` represents a multicast object, this call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that the system configuration is in a valid state and all required driver daemons are running properly.

Please note calling [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") does not make the address accessible, the caller needs to update accessibility of a contiguous mapped VA range by calling [cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range.").

Once a recipient process obtains a shareable memory handle from [cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type."), the process must use [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") to map the memory into its address ranges before setting accessibility with [cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range.").

[cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") can only create mappings on VA range reservations that are not currently mapped.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemUnmap](<group__CUDA__VA.html#group__CUDA__VA_1gfb50aac00c848fd7087e858f59bf7e2a> "Unmap the backing memory of a given address range."), [cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation."), [cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemMapArrayAsync ( [CUarrayMapInfo](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1>)*Â mapInfoList, unsigned int Â count, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.

######  Parameters

`mapInfoList`
    \- List of CUarrayMapInfo
`count`
    \- Count of CUarrayMapInfo in `mapInfoList`
`hStream`
    \- Stream identifier for the stream to use for map or unmap operations

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Performs map or unmap operations on subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays. Each operation is specified by a CUarrayMapInfo entry in the `mapInfoList` array of size `count`. The structure CUarrayMapInfo is defined as follow:


    â     typedef struct CUarrayMapInfo_st {
                  [CUresourcetype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9f0a76c9f6be437e75c8310aea5280f6>) resourceType;
                  union {
                      [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>) mipmap;
                      [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) array;
                  } resource;

                  [CUarraySparseSubresourceType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb7f440dccee1200001e4b14f53785d0a>) subresourceType;
                  union {
                      struct {
                          unsigned int level;
                          unsigned int layer;
                          unsigned int offsetX;
                          unsigned int offsetY;
                          unsigned int offsetZ;
                          unsigned int extentWidth;
                          unsigned int extentHeight;
                          unsigned int extentDepth;
                      } sparseLevel;
                      struct {
                          unsigned int layer;
                          unsigned long long offset;
                          unsigned long long size;
                      } miptail;
                  } subresource;

                  [CUmemOperationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge7469bd2e035fc9c937e84490fdcd349>) memOperationType;

                  [CUmemHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g31a65081da1729d11a1d6f5a433d93b0>) memHandleType;
                  union {
                      CUmemGenericAllocationHandle memHandle;
                  } memHandle;

                  unsigned long long offset;
                  unsigned int deviceBitMask;
                  unsigned int flags;
                  unsigned int reserved[2];
              } [CUarrayMapInfo](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1>);

where [CUarrayMapInfo::resourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_18cf3c7ba97c834ab8b0fcfb50fec578c>) specifies the type of resource to be operated on. If [CUarrayMapInfo::resourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_18cf3c7ba97c834ab8b0fcfb50fec578c>) is set to [CUresourcetype::CU_RESOURCE_TYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f68171f299e8447a926051e13d613d77b1>) then CUarrayMapInfo::resource::array must be set to a valid sparse CUDA array handle. The CUDA array must be either a 2D, 2D layered or 3D CUDA array and must have been allocated using [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array.") or [cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array.") with the flag [CUDA_ARRAY3D_SPARSE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8e13c9d3ef98d1f3dce95901a115abc2>) or [CUDA_ARRAY3D_DEFERRED_MAPPING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g854c29dbc47d04a4e42863cb87487d55>). For CUDA arrays obtained using [cuMipmappedArrayGetLevel](<group__CUDA__MEM.html#group__CUDA__MEM_1g82f276659f05be14820e99346b0f86b7> "Gets a mipmap level of a CUDA mipmapped array."), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned. If [CUarrayMapInfo::resourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_18cf3c7ba97c834ab8b0fcfb50fec578c>) is set to [CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f642868e220af0309016ec733e37db7f24>) then CUarrayMapInfo::resource::mipmap must be set to a valid sparse CUDA mipmapped array handle. The CUDA mipmapped array must be either a 2D, 2D layered or 3D CUDA mipmapped array and must have been allocated using [cuMipmappedArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1ga5d2e311c7f9b0bc6d130af824a40bd3> "Creates a CUDA mipmapped array.") with the flag [CUDA_ARRAY3D_SPARSE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8e13c9d3ef98d1f3dce95901a115abc2>) or [CUDA_ARRAY3D_DEFERRED_MAPPING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g854c29dbc47d04a4e42863cb87487d55>).

[CUarrayMapInfo::subresourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_17a2fe272bd8e44af2386f1c20c3d3c68>) specifies the type of subresource within the resource. CUarraySparseSubresourceType_enum is defined as:


    â    typedef enum CUarraySparseSubresourceType_enum {
                  CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
                  CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
              } [CUarraySparseSubresourceType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb7f440dccee1200001e4b14f53785d0a>);

where CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL indicates a sparse-miplevel which spans at least one tile in every dimension. The remaining miplevels which are too small to span at least one tile in any dimension constitute the mip tail region as indicated by CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL subresource type.

If [CUarrayMapInfo::subresourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_17a2fe272bd8e44af2386f1c20c3d3c68>) is set to CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL then CUarrayMapInfo::subresource::sparseLevel struct must contain valid array subregion offsets and extents. The CUarrayMapInfo::subresource::sparseLevel::offsetX, CUarrayMapInfo::subresource::sparseLevel::offsetY and CUarrayMapInfo::subresource::sparseLevel::offsetZ must specify valid X, Y and Z offsets respectively. The CUarrayMapInfo::subresource::sparseLevel::extentWidth, CUarrayMapInfo::subresource::sparseLevel::extentHeight and CUarrayMapInfo::subresource::sparseLevel::extentDepth must specify valid width, height and depth extents respectively. These offsets and extents must be aligned to the corresponding tile dimension. For CUDA mipmapped arrays CUarrayMapInfo::subresource::sparseLevel::level must specify a valid mip level index. Otherwise, must be zero. For layered CUDA arrays and layered CUDA mipmapped arrays CUarrayMapInfo::subresource::sparseLevel::layer must specify a valid layer index. Otherwise, must be zero. CUarrayMapInfo::subresource::sparseLevel::offsetZ must be zero and CUarrayMapInfo::subresource::sparseLevel::extentDepth must be set to 1 for 2D and 2D layered CUDA arrays and CUDA mipmapped arrays. Tile extents can be obtained by calling [cuArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1gf74df88a07404ee051f0e5b36647d8c7> "Returns the layout properties of a sparse CUDA array.") and [cuMipmappedArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1g55a16bd1780acb3cc94e8b88d5fe5e19> "Returns the layout properties of a sparse CUDA mipmapped array.")

If [CUarrayMapInfo::subresourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_17a2fe272bd8e44af2386f1c20c3d3c68>) is set to CUarraySparseSubresourceType::CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL then CUarrayMapInfo::subresource::miptail struct must contain valid mip tail offset in CUarrayMapInfo::subresource::miptail::offset and size in CUarrayMapInfo::subresource::miptail::size. Both, mip tail offset and mip tail size must be aligned to the tile size. For layered CUDA mipmapped arrays which don't have the flag [CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0dcf4ba7e64caa5c1aa4e88caa7f659a>) set in [CUDA_ARRAY_SPARSE_PROPERTIES::flags](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_10e842bb64091fa47809112c700cb5f0a>) as returned by [cuMipmappedArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1g55a16bd1780acb3cc94e8b88d5fe5e19> "Returns the layout properties of a sparse CUDA mipmapped array."), CUarrayMapInfo::subresource::miptail::layer must specify a valid layer index. Otherwise, must be zero.

If CUarrayMapInfo::resource::array or CUarrayMapInfo::resource::mipmap was created with [CUDA_ARRAY3D_DEFERRED_MAPPING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g854c29dbc47d04a4e42863cb87487d55>) flag set the [CUarrayMapInfo::subresourceType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_17a2fe272bd8e44af2386f1c20c3d3c68>) and the contents of CUarrayMapInfo::subresource will be ignored.

[CUarrayMapInfo::memOperationType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1dd139e9655407264f2aeb812cec0f19e>) specifies the type of operation. [CUmemOperationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge7469bd2e035fc9c937e84490fdcd349>) is defined as:


    â    typedef enum CUmemOperationType_enum {
                  CU_MEM_OPERATION_TYPE_MAP = 1,
                  CU_MEM_OPERATION_TYPE_UNMAP = 2
              } [CUmemOperationType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge7469bd2e035fc9c937e84490fdcd349>);

If [CUarrayMapInfo::memOperationType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1dd139e9655407264f2aeb812cec0f19e>) is set to CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP then the subresource will be mapped onto the tile pool memory specified by CUarrayMapInfo::memHandle at offset [CUarrayMapInfo::offset](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_14aee5e358272af897aeaf8b44fd15bdb>). The tile pool allocation has to be created by specifying the [CU_MEM_CREATE_USAGE_TILE_POOL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb445d91d9773d728b1a9962184c05799>) flag when calling [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."). Also, [CUarrayMapInfo::memHandleType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1ed15e0aa10304948c68946fe8a5da161>) must be set to CUmemHandleType::CU_MEM_HANDLE_TYPE_GENERIC.

If [CUarrayMapInfo::memOperationType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1dd139e9655407264f2aeb812cec0f19e>) is set to CUmemOperationType::CU_MEM_OPERATION_TYPE_UNMAP then an unmapping operation is performed. CUarrayMapInfo::memHandle must be NULL.

[CUarrayMapInfo::deviceBitMask](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1f2d14f719018cc1daa786f7fd0652c2c>) specifies the list of devices that must map or unmap physical memory. Currently, this mask must have exactly one bit set, and the corresponding device must match the device associated with the stream. If [CUarrayMapInfo::memOperationType](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1dd139e9655407264f2aeb812cec0f19e>) is set to CUmemOperationType::CU_MEM_OPERATION_TYPE_MAP, the device must also match the device associated with the tile pool memory allocation as specified by CUarrayMapInfo::memHandle.

[CUarrayMapInfo::flags](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_12e36982c16390693a56807d0b8e6380f>) and [CUarrayMapInfo::reserved](<structCUarrayMapInfo__v1.html#structCUarrayMapInfo__v1_1cf7014dc4a157928de12563d0181ceba>)[] are unused and must be set to zero.

**See also:**

[cuMipmappedArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1ga5d2e311c7f9b0bc6d130af824a40bd3> "Creates a CUDA mipmapped array."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1gf74df88a07404ee051f0e5b36647d8c7> "Returns the layout properties of a sparse CUDA array."), [cuMipmappedArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1g55a16bd1780acb3cc94e8b88d5fe5e19> "Returns the layout properties of a sparse CUDA mipmapped array.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemRelease ( CUmemGenericAllocationHandleÂ handle )


Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.

######  Parameters

`handle`
    Value of handle which was returned previously by cuMemCreate.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Frees the memory that was allocated on a device through cuMemCreate.

The memory allocation will be freed when all outstanding mappings to the memory are unmapped and when all outstanding references to the handle (including it's shareable counterparts) are also released. The generic memory handle can be freed when there are still outstanding mappings made with this handle. Each time a recipient process imports a shareable handle, it needs to pair it with [cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.") for the handle to be freed. If `handle` is not a valid handle the behavior is undefined.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemRetainAllocationHandle ( CUmemGenericAllocationHandle*Â handle, void*Â addr )


Given an address `addr`, returns the allocation handle of the backing memory allocation.

######  Parameters

`handle`
    CUDA Memory handle for the backing memory allocation.
`addr`
    Memory address to query, that has been mapped previously.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

The handle is guaranteed to be the same handle value used to map the memory. If the address requested is not mapped, the function will fail. The returned handle must be released with corresponding number of calls to [cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.").

Note:

The address `addr`, can be any address in a range previously mapped by [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."), and not necessarily the start address.

**See also:**

[cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate."), [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemSetAccess ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size, const [CUmemAccessDesc](<structCUmemAccessDesc__v1.html#structCUmemAccessDesc__v1>)*Â desc, size_tÂ count )


Set the access flags for each location specified in `desc` for the given virtual address range.

######  Parameters

`ptr`
    \- Starting address for the virtual address range
`size`
    \- Length of the virtual address range
`desc`
    \- Array of CUmemAccessDesc that describe how to change the

  * mapping for each location specified


`count`
    \- Number of CUmemAccessDesc in `desc`

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Given the virtual address range via `ptr` and `size`, and the locations in the array given by `desc` and `count`, set the access flags for the target locations. The range must be a fully mapped address range containing all allocations created by [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") / [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."). Users cannot specify [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>) accessibility for allocations created on with other location types. Note: When CUmemAccessDesc::CUmemLocation::type is [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>), CUmemAccessDesc::CUmemLocation::id is ignored. When setting the access flags for a virtual address range mapping a multicast object, `ptr` and `size` must be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag CU_MULTICAST_MINIMUM_GRANULARITY. For best performance however, it is recommended that `ptr` and `size` be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag CU_MULTICAST_RECOMMENDED_GRANULARITY.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), :[cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemUnmap ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â ptr, size_tÂ size )


Unmap the backing memory of a given address range.

######  Parameters

`ptr`
    \- Starting address for the virtual address range to unmap
`size`
    \- Size of the virtual address range to unmap

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

The range must be the entire contiguous address range that was mapped to. In other words, [cuMemUnmap](<group__CUDA__VA.html#group__CUDA__VA_1gfb50aac00c848fd7087e858f59bf7e2a> "Unmap the backing memory of a given address range.") cannot unmap a sub-range of an address range mapped by [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.") / [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."). Any backing memory allocations will be freed if there are no existing mappings and there are no unreleased memory handles.

When [cuMemUnmap](<group__CUDA__VA.html#group__CUDA__VA_1gfb50aac00c848fd7087e858f59bf7e2a> "Unmap the backing memory of a given address range.") returns successfully the address range is converted to an address reservation and can be used for a future calls to [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."). Any new mapping to this virtual address will need to have access granted through [cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range."), as all mappings start with no accessibility setup.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation.")

* * *
