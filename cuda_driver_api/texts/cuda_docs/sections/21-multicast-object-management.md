# Multicast Object Management

## 6.16.Â Multicast Object Management

This section describes the CUDA multicast object operations exposed by the low-level CUDA driver application programming interface.

**overview**

A multicast object created via [cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties.") enables certain memory operations to be broadcast to a team of devices. Devices can be added to a multicast object via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). Memory can be bound on each participating device via [cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object."), or [cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object."). Multicast objects can be mapped into a device's virtual address space using the virtual memmory management APIs (see [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") and [cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range.")).

**Supported Platforms**

Support for multicast on a specific device can be queried using the device attribute [CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3ce470c9ff9166a3a8740bef623f5d299>)

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastAddDevice](<#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d>) ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Associate a device to a multicast object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastBindAddr](<#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f>) ( CUmemGenericAllocationHandleÂ mcHandle, size_tÂ mcOffset, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â memptr, size_tÂ size, unsigned long longÂ flags )
     Bind a memory allocation represented by a virtual address to a multicast object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastBindAddr_v2](<#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed>) ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, size_tÂ mcOffset, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â memptr, size_tÂ size, unsigned long longÂ flags )
     Bind a memory allocation represented by a virtual address to a multicast object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastBindMem](<#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb>) ( CUmemGenericAllocationHandleÂ mcHandle, size_tÂ mcOffset, CUmemGenericAllocationHandleÂ memHandle, size_tÂ memOffset, size_tÂ size, unsigned long longÂ flags )
     Bind a memory allocation represented by a handle to a multicast object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastBindMem_v2](<#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5>) ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, size_tÂ mcOffset, CUmemGenericAllocationHandleÂ memHandle, size_tÂ memOffset, size_tÂ size, unsigned long longÂ flags )
     Bind a memory allocation represented by a handle to a multicast object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastCreate](<#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242>) ( CUmemGenericAllocationHandle*Â mcHandle, const [CUmulticastObjectProp](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1>)*Â prop )
     Create a generic allocation handle representing a multicast object described by the given properties.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastGetGranularity](<#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547>) ( size_t*Â granularity, const [CUmulticastObjectProp](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1>)*Â prop, [CUmulticastGranularity_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gcdeff171670a788001418262a0f88378>)Â option )
     Calculates either the minimal or recommended granularity for multicast object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMulticastUnbind](<#group__CUDA__MULTICAST_1g424b4563760cc5dab52ee9a8d28656ac>) ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, size_tÂ mcOffset, size_tÂ size )
     Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastAddDevice ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Associate a device to a multicast object.

######  Parameters

`mcHandle`
    Handle representing a multicast object.
`dev`
    Device that will be associated to the multicast object.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Associates a device to a multicast object. The added device will be a part of the multicast team of size specified by [CUmulticastObjectProp::numDevices](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1_1e5d44c9262847a6c74e4ae37acdc7478>) during [cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."). The association of the device to the multicast object is permanent during the life time of the multicast object. All devices must be added to the multicast team before any memory can be bound to any device in the team. Any calls to [cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object."), or [cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object.") will block until all devices have been added. Similarly all devices must be added to the multicast team before a virtual address range can be mapped to the multicast object. A call to [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") will block until all devices have been added.

**See also:**

[cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."), [cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastBindAddr ( CUmemGenericAllocationHandleÂ mcHandle, size_tÂ mcOffset, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â memptr, size_tÂ size, unsigned long longÂ flags )


Bind a memory allocation represented by a virtual address to a multicast object.

######  Parameters

`mcHandle`
    Handle representing a multicast object.
`mcOffset`
    Offset into multicast va range for attachment.
`memptr`
    Virtual address of the memory allocation.
`size`
    Size of memory that will be bound to the multicast object.
`flags`
    Flags for future use, must be zero now.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_SYSTEM_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9659c7730ee79fae8262043448f2ce1e3>), [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>)

###### Description

Binds a memory allocation specified by its mapped address `memptr` to a multicast object represented by `mcHandle`. The memory must have been allocated via [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.") or [cudaMallocAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb>). The intended `size` of the bind, the offset in the multicast range `mcOffset` and `memptr` must be a multiple of the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378790ecc4b1f2c2f6e2c9bc21b230872a9>). For best performance however, `size`, `mcOffset` and `memptr` should be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_RECOMMENDED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378dfff741c8fa1dd2f96e584939b0c53ce>).

The `size` cannot be larger than the size of the allocated memory. Similarly the `size` \+ `mcOffset` cannot be larger than the total size of the multicast object. The memory allocation must have beeen created on one of the devices that was added to the multicast team via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). Externally shareable as well as imported multicast objects can be bound only to externally shareable memory. Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are insufficient resources required to perform the bind. This call may also return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or running.

This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that the system configuration is in a valid state and all required driver daemons are running properly.

**See also:**

[cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."), [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.")

[cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastBindAddr_v2 ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, size_tÂ mcOffset, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â memptr, size_tÂ size, unsigned long longÂ flags )


Bind a memory allocation represented by a virtual address to a multicast object.

######  Parameters

`mcHandle`
    Handle representing a multicast object.
`dev`
    The device that for which the multicast memory binding will be applicable.
`mcOffset`
    Offset into multicast va range for attachment.
`memptr`
    Virtual address of the memory allocation.
`size`
    Size of memory that will be bound to the multicast object.
`flags`
    Flags for future use, must be zero now.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_SYSTEM_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9659c7730ee79fae8262043448f2ce1e3>), [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>)

###### Description

Binds a memory allocation specified by its mapped address `memptr` to a multicast object represented by `mcHandle`. The binding will be applicable for the device `dev`. The memory must have been allocated via [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.") or [cudaMallocAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb>). The intended `size` of the bind, the offset in the multicast range `mcOffset` and `memptr` must be a multiple of the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378790ecc4b1f2c2f6e2c9bc21b230872a9>). For best performance however, `size`, `mcOffset` and `memptr` should be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_RECOMMENDED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378dfff741c8fa1dd2f96e584939b0c53ce>).

The `size` cannot be larger than the size of the allocated memory. Similarly the `size` \+ `mcOffset` cannot be larger than the total size of the multicast object. For device memory, i.e., type [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), the memory allocation must have been created on the device specified by `dev`. For host NUMA memory, i.e., type [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>), the memory allocation must have been created on the CPU NUMA node closest to `dev`. That is, the value returned when querying [CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a32583f8184b09441b3c83b1bee7849556>) for `dev`, must be the CPU NUMA node where the memory was allocated. In both cases, the device named by `dev` must have been added to the multicast team via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). Externally shareable as well as imported multicast objects can be bound only to externally shareable memory. Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are insufficient resources required to perform the bind. This call may also return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or running.

This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that the system configuration is in a valid state and all required driver daemons are running properly.

**See also:**

[cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."), [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastBindMem ( CUmemGenericAllocationHandleÂ mcHandle, size_tÂ mcOffset, CUmemGenericAllocationHandleÂ memHandle, size_tÂ memOffset, size_tÂ size, unsigned long longÂ flags )


Bind a memory allocation represented by a handle to a multicast object.

######  Parameters

`mcHandle`
    Handle representing a multicast object.
`mcOffset`
    Offset into the multicast object for attachment.
`memHandle`
    Handle representing a memory allocation.
`memOffset`
    Offset into the memory for attachment.
`size`
    Size of the memory that will be bound to the multicast object.
`flags`
    Flags for future use, must be zero for now.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_SYSTEM_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9659c7730ee79fae8262043448f2ce1e3>), [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>)

###### Description

Binds a memory allocation specified by `memHandle` and created via [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.") to a multicast object represented by `mcHandle` and created via [cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."). The intended `size` of the bind, the offset in the multicast range `mcOffset` as well as the offset in the memory `memOffset` must be a multiple of the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378790ecc4b1f2c2f6e2c9bc21b230872a9>). For best performance however, `size`, `mcOffset` and `memOffset` should be aligned to the granularity of the memory allocation(see ::cuMemGetAllocationGranularity) or to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_RECOMMENDED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378dfff741c8fa1dd2f96e584939b0c53ce>).

The `size` \+ `memOffset` cannot be larger than the size of the allocated memory. Similarly the `size` \+ `mcOffset` cannot be larger than the size of the multicast object. The memory allocation must have beeen created on one of the devices that was added to the multicast team via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). Externally shareable as well as imported multicast objects can be bound only to externally shareable memory. Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are insufficient resources required to perform the bind. This call may also return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or running.

This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that the system configuration is in a valid state and all required driver daemons are running properly.

**See also:**

[cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."), [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.")

[cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastBindMem_v2 ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, size_tÂ mcOffset, CUmemGenericAllocationHandleÂ memHandle, size_tÂ memOffset, size_tÂ size, unsigned long longÂ flags )


Bind a memory allocation represented by a handle to a multicast object.

######  Parameters

`mcHandle`
    Handle representing a multicast object.
`dev`
    The device that for which the multicast memory binding will be applicable.
`mcOffset`
    Offset into the multicast object for attachment.
`memHandle`
    Handle representing a memory allocation.
`memOffset`
    Offset into the memory for attachment.
`size`
    Size of the memory that will be bound to the multicast object.
`flags`
    Flags for future use, must be zero for now.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_SYSTEM_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9659c7730ee79fae8262043448f2ce1e3>), [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>)

###### Description

Binds a memory allocation specified by `memHandle` and created via [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.") to a multicast object represented by `mcHandle` and created via [cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."). The binding will be applicable for the device `dev`. The intended `size` of the bind, the offset in the multicast range `mcOffset` as well as the offset in the memory `memOffset` must be a multiple of the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378790ecc4b1f2c2f6e2c9bc21b230872a9>). For best performance however, `size`, `mcOffset` and `memOffset` should be aligned to the granularity of the memory allocation(see ::cuMemGetAllocationGranularity) or to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_RECOMMENDED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378dfff741c8fa1dd2f96e584939b0c53ce>).

The `size` \+ `memOffset` cannot be larger than the size of the allocated memory. Similarly the `size` \+ `mcOffset` cannot be larger than the size of the multicast object. The memory allocation must have beeen created on one of the devices that was added to the multicast team via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). For device memory, i.e., type [CU_MEM_LOCATION_TYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882ea4409aa6b414995d628a320eafbbbb6e>), the memory allocation must have been created on the device specified by `dev`. For host NUMA memory, i.e., type [CU_MEM_LOCATION_TYPE_HOST_NUMA](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75cfd5b9fa5c1c6ee2be2547bfbe882eb61a1d3409ed83a43b5706cc004ac861>), the memory allocation must have been created on the CPU NUMA node closest to `dev`. That is, the value returned when querying [CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a32583f8184b09441b3c83b1bee7849556>) for `dev`, must be the CPU NUMA node where the memory was allocated. In both cases, the device named by `dev` must have been added to the multicast team via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). Externally shareable as well as imported multicast objects can be bound only to externally shareable memory. Note that this call will return CUDA_ERROR_OUT_OF_MEMORY if there are insufficient resources required to perform the bind. This call may also return CUDA_ERROR_SYSTEM_NOT_READY if the necessary system software is not initialized or running.

This call may return CUDA_ERROR_ILLEGAL_STATE if the system configuration is in an illegal state. In such cases, to continue using multicast, verify that the system configuration is in a valid state and all required driver daemons are running properly.

**See also:**

[cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."), [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."), [cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastCreate ( CUmemGenericAllocationHandle*Â mcHandle, const [CUmulticastObjectProp](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1>)*Â prop )


Create a generic allocation handle representing a multicast object described by the given properties.

######  Parameters

`mcHandle`
    Value of handle returned.
`prop`
    Properties of the multicast object to create.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

This creates a multicast object as described by `prop`. The number of participating devices is specified by [CUmulticastObjectProp::numDevices](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1_1e5d44c9262847a6c74e4ae37acdc7478>). Devices can be added to the multicast object via [cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."). All participating devices must be added to the multicast object before memory can be bound to it. Memory is bound to the multicast object via [cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object."), or [cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object."). and can be unbound via [cuMulticastUnbind](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g424b4563760cc5dab52ee9a8d28656ac> "Unbind any memory allocations bound to a multicast object at a given offset and upto a given size."). The total amount of memory that can be bound per device is specified by :[CUmulticastObjectProp::size](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1_1a45dfd715e2e442fcc7e43f5ce2f8a46>). This size must be a multiple of the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378790ecc4b1f2c2f6e2c9bc21b230872a9>). For best performance however, the size should be aligned to the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") with the flag [CU_MULTICAST_GRANULARITY_RECOMMENDED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378dfff741c8fa1dd2f96e584939b0c53ce>).

After all participating devices have been added, multicast objects can also be mapped to a device's virtual address space using the virtual memory management APIs (see [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range.") and [cuMemSetAccess](<group__CUDA__VA.html#group__CUDA__VA_1g1b6b12b10e8324bf462ecab4e7ef30e1> "Set the access flags for each location specified in desc for the given virtual address range.")). Multicast objects can also be shared with other processes by requesting a shareable handle via [cuMemExportToShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56> "Exports an allocation to a requested shareable handle type."). Note that the desired types of shareable handles must be specified in the bitmask [CUmulticastObjectProp::handleTypes](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1_1705cc17eda91d960e96982f2fda52d55>). Multicast objects can be released using the virtual memory management API [cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate.").

**See also:**

[cuMulticastAddDevice](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g2df07b38ff506c519e9f799c5ddf7e5d> "Associate a device to a multicast object."), [cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object."), [cuMulticastUnbind](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g424b4563760cc5dab52ee9a8d28656ac> "Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.")

[cuMemCreate](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemRelease](<group__CUDA__VA.html#group__CUDA__VA_1g3014f0759f43a8d82db951b8e4b91d68> "Release a memory handle representing a memory allocation which was previously allocated through cuMemCreate."), [cuMemExportToShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g633f273b155815f23c1d70e7d9384c56> "Exports an allocation to a requested shareable handle type."), [cuMemImportFromShareableHandle](<group__CUDA__VA.html#group__CUDA__VA_1g1577822cc83ea896b4892f2d69630463> "Imports an allocation from a requested shareable handle type.")

[cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object."), [cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastGetGranularity ( size_t*Â granularity, const [CUmulticastObjectProp](<structCUmulticastObjectProp__v1.html#structCUmulticastObjectProp__v1>)*Â prop, [CUmulticastGranularity_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gcdeff171670a788001418262a0f88378>)Â option )


Calculates either the minimal or recommended granularity for multicast object.

######  Parameters

`granularity`
    Returned granularity.
`prop`
    Properties of the multicast object.
`option`
    Determines which granularity to return.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Calculates either the minimal or recommended granularity for a given set of multicast object properties and returns it in granularity. This granularity can be used as a multiple for size, bind offsets and address mappings of the multicast object.

**See also:**

[cuMulticastCreate](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gd4413361f466d4ed86a93d8c309c0242> "Create a generic allocation handle representing a multicast object described by the given properties."), [cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object."), [cuMulticastUnbind](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g424b4563760cc5dab52ee9a8d28656ac> "Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.")

[cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMulticastUnbind ( CUmemGenericAllocationHandleÂ mcHandle, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, size_tÂ mcOffset, size_tÂ size )


Unbind any memory allocations bound to a multicast object at a given offset and upto a given size.

######  Parameters

`mcHandle`
    Handle representing a multicast object.
`dev`
    Device that hosts the memory allocation.
`mcOffset`
    Offset into the multicast object.
`size`
    Desired size to unbind.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Unbinds any memory allocations hosted on `dev` and bound to a multicast object at `mcOffset` and upto a given `size`. The intended `size` of the unbind and the offset in the multicast range ( `mcOffset` ) must be a multiple of the value returned by [cuMulticastGetGranularity](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g067a02ad98f4e01f149011f523fec547> "Calculates either the minimal or recommended granularity for multicast object.") flag [CU_MULTICAST_GRANULARITY_MINIMUM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggcdeff171670a788001418262a0f88378790ecc4b1f2c2f6e2c9bc21b230872a9>). The `size` \+ `mcOffset` cannot be larger than the total size of the multicast object.

Note:

Warning: The `mcOffset` and the `size` must match the corresponding values specified during the bind call. Any other values may result in undefined behavior.

**See also:**

[cuMulticastBindMem](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1gcadf88b616a3f766f6279288e435a4bb> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g07c8cf1d3bf1d04a2d1867f09647b03f> "Bind a memory allocation represented by a virtual address to a multicast object.")

[cuMulticastBindMem_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0e0ed2e81af121bdcbb54e1a9c4e63a5> "Bind a memory allocation represented by a handle to a multicast object."), [cuMulticastBindAddr_v2](<group__CUDA__MULTICAST.html#group__CUDA__MULTICAST_1g0eb82d8911bf179ae239d522da049bed> "Bind a memory allocation represented by a virtual address to a multicast object.")

* * *
