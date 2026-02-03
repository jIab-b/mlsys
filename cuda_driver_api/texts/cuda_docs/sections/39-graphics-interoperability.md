# Graphics Interoperability

## 6.32.Â Graphics Interoperability

This section describes the graphics interoperability functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsMapResources](<#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0>) ( unsigned int Â count, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â resources, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Map graphics resources for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsResourceGetMappedMipmappedArray](<#group__CUDA__GRAPHICS_1g37680bbe89c7fe5c613563eaab9d14c1>) ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â pMipmappedArray, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource )
     Get a mipmapped array through which to access a mapped graphics resource.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsResourceGetMappedPointer](<#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pDevPtr, size_t*Â pSize, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource )
     Get a device pointer through which to access a mapped graphics resource.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsResourceSetMapFlags](<#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource, unsigned int Â flags )
     Set usage flags for mapping a graphics resource.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsSubResourceGetMappedArray](<#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pArray, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource, unsigned int Â arrayIndex, unsigned int Â mipLevel )
     Get an array through which to access a subresource of a mapped graphics resource.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsUnmapResources](<#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32>) ( unsigned int Â count, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â resources, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Unmap graphics resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsUnregisterResource](<#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource )
     Unregisters a graphics resource for access by CUDA.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsMapResources ( unsigned int Â count, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â resources, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Map graphics resources for access by CUDA.

######  Parameters

`count`
    \- Number of resources to map
`resources`
    \- Resources to map for CUDA usage
`hStream`
    \- Stream with which to synchronize

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Maps the `count` graphics resources in `resources` for access by CUDA.

The resources in `resources` may be accessed by CUDA until they are unmapped. The graphics API from which `resources` were registered should not access any resources while they are mapped by CUDA. If an application does so, the results are undefined.

This function provides the synchronization guarantee that any graphics calls issued before [cuGraphicsMapResources()](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA.") will complete before any subsequent CUDA work issued in `stream` begins.

If `resources` includes any duplicate entries then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If any of `resources` are presently mapped for access by CUDA then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources."), [cudaGraphicsMapResources](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gad8fbe74d02adefb8e7efb4971ee6322>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsResourceGetMappedMipmappedArray ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â pMipmappedArray, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource )


Get a mipmapped array through which to access a mapped graphics resource.

######  Parameters

`pMipmappedArray`
    \- Returned mipmapped array through which `resource` may be accessed
`resource`
    \- Mapped resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>), [CUDA_ERROR_NOT_MAPPED_AS_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9037f39ebe1bbd16d1b162f4456d507c5>)

###### Description

Returns in `*pMipmappedArray` a mipmapped array through which the mapped graphics resource `resource`. The value set in `*pMipmappedArray` may change every time that `resource` is mapped.

If `resource` is not a texture then it cannot be accessed via a mipmapped array and [CUDA_ERROR_NOT_MAPPED_AS_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9037f39ebe1bbd16d1b162f4456d507c5>) is returned. If `resource` is not mapped then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cudaGraphicsResourceGetMappedMipmappedArray](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g84c3772d2ed06cda8c92bc43cdc893d0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsResourceGetMappedPointer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pDevPtr, size_t*Â pSize, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource )


Get a device pointer through which to access a mapped graphics resource.

######  Parameters

`pDevPtr`
    \- Returned pointer through which `resource` may be accessed
`pSize`
    \- Returned size of the buffer accessible starting at `*pPointer`
`resource`
    \- Mapped resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>), [CUDA_ERROR_NOT_MAPPED_AS_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a7ceef277b14abcb140c4ffa72a37473>)

###### Description

Returns in `*pDevPtr` a pointer through which the mapped graphics resource `resource` may be accessed. Returns in `pSize` the size of the memory in bytes which may be accessed from that pointer. The value set in `pPointer` may change every time that `resource` is mapped.

If `resource` is not a buffer then it cannot be accessed via a pointer and [CUDA_ERROR_NOT_MAPPED_AS_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a7ceef277b14abcb140c4ffa72a37473>) is returned. If `resource` is not mapped then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned. *

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cudaGraphicsResourceGetMappedPointer](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1ga36881081c8deb4df25c256158e1ac99>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsResourceSetMapFlags ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource, unsigned int Â flags )


Set usage flags for mapping a graphics resource.

######  Parameters

`resource`
    \- Registered resource to set flags for
`flags`
    \- Parameters for resource mapping

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>)

###### Description

Set `flags` for mapping the graphics resource `resource`.

Changes to `flags` will take effect the next time `resource` is mapped. The `flags` argument may be any of the following:

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA kernels. This is the default value.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which access this resource will not write to this resource.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels which access this resource will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


If `resource` is presently mapped for access by CUDA then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned. If `flags` is not one of the above values then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cudaGraphicsResourceSetMapFlags](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g5f94a0043909fddc100ab5f0c2476b9f>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsSubResourceGetMappedArray ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pArray, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource, unsigned int Â arrayIndex, unsigned int Â mipLevel )


Get an array through which to access a subresource of a mapped graphics resource.

######  Parameters

`pArray`
    \- Returned array through which a subresource of `resource` may be accessed
`resource`
    \- Mapped resource to access
`arrayIndex`
    \- Array index for array textures or cubemap face index as defined by [CUarray_cubemap_face](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g012fda14b50e7db8798a340627c4c330>) for cubemap textures for the subresource to access
`mipLevel`
    \- Mipmap level for the subresource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>), [CUDA_ERROR_NOT_MAPPED_AS_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9037f39ebe1bbd16d1b162f4456d507c5>)

###### Description

Returns in `*pArray` an array through which the subresource of the mapped graphics resource `resource` which corresponds to array index `arrayIndex` and mipmap level `mipLevel` may be accessed. The value set in `*pArray` may change every time that `resource` is mapped.

If `resource` is not a texture then it cannot be accessed via an array and [CUDA_ERROR_NOT_MAPPED_AS_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9037f39ebe1bbd16d1b162f4456d507c5>) is returned. If `arrayIndex` is not a valid array index for `resource` then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned. If `mipLevel` is not a valid mipmap level for `resource` then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned. If `resource` is not mapped then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cudaGraphicsSubResourceGetMappedArray](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g0dd6b5f024dfdcff5c28a08ef9958031>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsUnmapResources ( unsigned int Â count, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â resources, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Unmap graphics resources.

######  Parameters

`count`
    \- Number of resources to unmap
`resources`
    \- Resources to unmap
`hStream`
    \- Stream with which to synchronize

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Unmaps the `count` graphics resources in `resources`.

Once unmapped, the resources in `resources` may not be accessed by CUDA until they are mapped again.

This function provides the synchronization guarantee that any CUDA work issued in `stream` before [cuGraphicsUnmapResources()](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources.") will complete before any subsequently issued graphics work begins.

If `resources` includes any duplicate entries then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If any of `resources` are not presently mapped for access by CUDA then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cudaGraphicsUnmapResources](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1g11988ab4431b11ddb7cbde7aedb60491>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsUnregisterResource ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource )


Unregisters a graphics resource for access by CUDA.

######  Parameters

`resource`
    \- Resource to unregister

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Unregisters the graphics resource `resource` so it is not accessible by CUDA unless registered again.

If `resource` is invalid then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsD3D9RegisterResource](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g391835e0d3c5a34bdba99840157194bf> "Register a Direct3D 9 resource for access by CUDA."), [cuGraphicsD3D10RegisterResource](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g87fb2a189c27c4b63538d23f53b2c8e6> "Register a Direct3D 10 resource for access by CUDA."), [cuGraphicsD3D11RegisterResource](<group__CUDA__D3D11.html#group__CUDA__D3D11_1g4c02792aa87c3acc255b9de15b0509da> "Register a Direct3D 11 resource for access by CUDA."), [cuGraphicsGLRegisterBuffer](<group__CUDA__GL.html#group__CUDA__GL_1gd530f66cc9ab43a31a98527e75f343a0> "Registers an OpenGL buffer object."), [cuGraphicsGLRegisterImage](<group__CUDA__GL.html#group__CUDA__GL_1g52c3a36c4c92611b6fcf0662b2f74e40> "Register an OpenGL texture or renderbuffer object."), [cudaGraphicsUnregisterResource](<../cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gc65d1f2900086747de1e57301d709940>)

* * *
