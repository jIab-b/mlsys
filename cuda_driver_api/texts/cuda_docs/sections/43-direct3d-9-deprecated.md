# Direct3D 9 (Deprecated)

## 6.41.1.Â Direct3D 9 Interoperability [DEPRECATED]

## [[Direct3D 9 Interoperability](<group__CUDA__D3D9.html#group__CUDA__D3D9>)]

This section describes deprecated Direct3D 9 interoperability functionality.

### Enumerations

enumÂ [CUd3d9map_flags](<#group__CUDA__D3D9__DEPRECATED_1ge689aa9141452e4048257aabe606d6bc>)

enumÂ [CUd3d9register_flags](<#group__CUDA__D3D9__DEPRECATED_1g8300da18582a2d6b981e74b2348e4f77>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9MapResources](<#group__CUDA__D3D9__DEPRECATED_1g092c45dd723d9881c7c95b2fdbecb5d8>) ( unsigned int Â count, IDirect3DResource9**Â ppResource )
     Map Direct3D resources for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9RegisterResource](<#group__CUDA__D3D9__DEPRECATED_1g2797f46baff62444656115eae9c8e1de>) ( IDirect3DResource9*Â pResource, unsigned int Â Flags )
     Register a Direct3D resource for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9ResourceGetMappedArray](<#group__CUDA__D3D9__DEPRECATED_1g23597dfa283062869d6e62e0c3c03d91>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pArray, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )
     Get an array through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9ResourceGetMappedPitch](<#group__CUDA__D3D9__DEPRECATED_1g22273146dd5681ccfc473bd9ef82f9f2>) ( size_t*Â pPitch, size_t*Â pPitchSlice, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )
     Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9ResourceGetMappedPointer](<#group__CUDA__D3D9__DEPRECATED_1g7fb080fc9497d1f25fe1e3a226c56bdc>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pDevPtr, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )
     Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9ResourceGetMappedSize](<#group__CUDA__D3D9__DEPRECATED_1g84dbbbfc3cbb9260e22fea8f632c40ac>) ( size_t*Â pSize, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )
     Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9ResourceGetSurfaceDimensions](<#group__CUDA__D3D9__DEPRECATED_1gdf64c251f6fdc73869b7ccdf34befb1c>) ( size_t*Â pWidth, size_t*Â pHeight, size_t*Â pDepth, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )
     Get the dimensions of a registered surface.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9ResourceSetMapFlags](<#group__CUDA__D3D9__DEPRECATED_1gbb78cc5147eaed89e0a9b86dbf2d7408>) ( IDirect3DResource9*Â pResource, unsigned int Â Flags )
     Set usage flags for mapping a Direct3D resource.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9UnmapResources](<#group__CUDA__D3D9__DEPRECATED_1gb985949a449e21274dd346c538619afe>) ( unsigned int Â count, IDirect3DResource9**Â ppResource )
     Unmaps Direct3D resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9UnregisterResource](<#group__CUDA__D3D9__DEPRECATED_1g4889264b5a904f2dcb0f739ee67c4ce6>) ( IDirect3DResource9*Â pResource )
     Unregister a Direct3D resource.

### Enumerations

enum CUd3d9map_flags


Flags to map or unmap a resource

######  Values

CU_D3D9_MAPRESOURCE_FLAGS_NONE = 0x00

CU_D3D9_MAPRESOURCE_FLAGS_READONLY = 0x01

CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02


enum CUd3d9register_flags


Flags to register a resource

######  Values

CU_D3D9_REGISTER_FLAGS_NONE = 0x00

CU_D3D9_REGISTER_FLAGS_ARRAY = 0x01


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9MapResources ( unsigned int Â count, IDirect3DResource9**Â ppResource )


Map Direct3D resources for access by CUDA.

######  Parameters

`count`
    \- Number of resources in ppResource
`ppResource`
    \- Resources to map for CUDA usage

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000082>)

This function is deprecated as of CUDA 3.0.

###### Description

Maps the `count` Direct3D resources in `ppResource` for access by CUDA.

The resources in `ppResource` may be accessed in CUDA kernels until they are unmapped. Direct3D should not access any resources while they are mapped by CUDA. If an application does so the results are undefined.

This function provides the synchronization guarantee that any Direct3D calls issued before [cuD3D9MapResources()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g092c45dd723d9881c7c95b2fdbecb5d8> "Map Direct3D resources for access by CUDA.") will complete before any CUDA kernels issued after [cuD3D9MapResources()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g092c45dd723d9881c7c95b2fdbecb5d8> "Map Direct3D resources for access by CUDA.") begin.

If any of `ppResource` have not been registered for use with CUDA or if `ppResource` contains any duplicate entries, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If any of `ppResource` are presently mapped for access by CUDA, then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9RegisterResource ( IDirect3DResource9*Â pResource, unsigned int Â Flags )


Register a Direct3D resource for access by CUDA.

######  Parameters

`pResource`
    \- Resource to register for CUDA access
`Flags`
    \- Flags for resource registration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000080>)

This function is deprecated as of CUDA 3.0.

###### Description

Registers the Direct3D resource `pResource` for access by CUDA.

If this call is successful, then the application will be able to map and unmap this resource until it is unregistered through [cuD3D9UnregisterResource()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g4889264b5a904f2dcb0f739ee67c4ce6> "Unregister a Direct3D resource."). Also on success, this call will increase the internal reference count on `pResource`. This reference count will be decremented when this resource is unregistered through [cuD3D9UnregisterResource()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g4889264b5a904f2dcb0f739ee67c4ce6> "Unregister a Direct3D resource.").

This call is potentially high-overhead and should not be called every frame in interactive applications.

The type of `pResource` must be one of the following.

  * IDirect3DVertexBuffer9: Cannot be used with `Flags` set to CU_D3D9_REGISTER_FLAGS_ARRAY.

  * IDirect3DIndexBuffer9: Cannot be used with `Flags` set to CU_D3D9_REGISTER_FLAGS_ARRAY.

  * IDirect3DSurface9: Only stand-alone objects of type IDirect3DSurface9 may be explicitly shared. In particular, individual mipmap levels and faces of cube maps may not be registered directly. To access individual surfaces associated with a texture, one must register the base texture object. For restrictions on the `Flags` parameter, see type IDirect3DBaseTexture9.

  * IDirect3DBaseTexture9: When a texture is registered, all surfaces associated with the all mipmap levels of all faces of the texture will be accessible to CUDA.


The `Flags` argument specifies the mechanism through which CUDA will access the Direct3D resource. The following values are allowed.

  * CU_D3D9_REGISTER_FLAGS_NONE: Specifies that CUDA will access this resource through a [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>). The pointer, size, and (for textures), pitch for each subresource of this allocation may be queried through [cuD3D9ResourceGetMappedPointer()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g7fb080fc9497d1f25fe1e3a226c56bdc> "Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA."), [cuD3D9ResourceGetMappedSize()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g84dbbbfc3cbb9260e22fea8f632c40ac> "Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA."), and [cuD3D9ResourceGetMappedPitch()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g22273146dd5681ccfc473bd9ef82f9f2> "Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.") respectively. This option is valid for all resource types.

  * CU_D3D9_REGISTER_FLAGS_ARRAY: Specifies that CUDA will access this resource through a [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) queried on a sub-resource basis through [cuD3D9ResourceGetMappedArray()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g23597dfa283062869d6e62e0c3c03d91> "Get an array through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA."). This option is only valid for resources of type IDirect3DSurface9 and subtypes of IDirect3DBaseTexture9.


Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some limitations.

  * The primary rendertarget may not be registered with CUDA.

  * Resources allocated as shared may not be registered with CUDA.

  * Any resources allocated in D3DPOOL_SYSTEMMEM or D3DPOOL_MANAGED may not be registered with CUDA.

  * Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data cannot be shared.

  * Surfaces of depth or stencil formats cannot be shared.


If Direct3D interoperability is not initialized on this context, then [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned. If `pResource` is of incorrect type (e.g. is a non-stand-alone IDirect3DSurface9) or is already registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` cannot be registered then [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsD3D9RegisterResource](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g391835e0d3c5a34bdba99840157194bf> "Register a Direct3D 9 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9ResourceGetMappedArray ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pArray, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )


Get an array through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pArray`
    \- Returned array corresponding to subresource
`pResource`
    \- Mapped resource to access
`Face`
    \- Face of resource to access
`Level`
    \- Level of resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000086>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pArray` an array through which the subresource of the mapped Direct3D resource `pResource` which corresponds to `Face` and `Level` may be accessed. The value set in `pArray` may change every time that `pResource` is mapped.

If `pResource` is not registered then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D9_REGISTER_FLAGS_ARRAY then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

For usage requirements of `Face` and `Level` parameters, see [cuD3D9ResourceGetMappedPointer()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g7fb080fc9497d1f25fe1e3a226c56bdc> "Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9ResourceGetMappedPitch ( size_t*Â pPitch, size_t*Â pPitchSlice, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )


Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pPitch`
    \- Returned pitch of subresource
`pPitchSlice`
    \- Returned Z-slice pitch of subresource
`pResource`
    \- Mapped resource to access
`Face`
    \- Face of resource to access
`Level`
    \- Level of resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000089>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pPitch` and `*pPitchSlice` the pitch and Z-slice pitch of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `Face` and `Level`. The values set in `pPitch` and `pPitchSlice` may change every time that `pResource` is mapped.

The pitch and Z-slice pitch values may be used to compute the location of a sample on a surface as follows.

For a 2D surface, the byte offset of the sample at position **x** , **y** from the base pointer of the surface is:

**y** * **pitch** \+ (**bytes per pixel**) * **x**

For a 3D surface, the byte offset of the sample at position **x** , **y** , **z** from the base pointer of the surface is:

**z*** **slicePitch** \+ **y** * **pitch** \+ (**bytes per pixel**) * **x**

Both parameters `pPitch` and `pPitchSlice` are optional and may be set to NULL.

If `pResource` is not of type IDirect3DBaseTexture9 or one of its sub-types or if `pResource` has not been registered for use with CUDA, then [cudaErrorInvalidResourceHandle](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038c4673247aee4d1ab8d07871f376e0273>) is returned. If `pResource` was not registered with usage flags CU_D3D9_REGISTER_FLAGS_NONE, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped for access by CUDA then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

For usage requirements of `Face` and `Level` parameters, see [cuD3D9ResourceGetMappedPointer()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g7fb080fc9497d1f25fe1e3a226c56bdc> "Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9ResourceGetMappedPointer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pDevPtr, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )


Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pDevPtr`
    \- Returned pointer corresponding to subresource
`pResource`
    \- Mapped resource to access
`Face`
    \- Face of resource to access
`Level`
    \- Level of resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000087>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pDevPtr` the base pointer of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `Face` and `Level`. The value set in `pDevPtr` may change every time that `pResource` is mapped.

If `pResource` is not registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D9_REGISTER_FLAGS_NONE, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

If `pResource` is of type IDirect3DCubeTexture9, then `Face` must one of the values enumerated by type D3DCUBEMAP_FACES. For all other types `Face` must be 0. If `Face` is invalid, then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

If `pResource` is of type IDirect3DBaseTexture9, then `Level` must correspond to a valid mipmap level. At present only mipmap level 0 is supported. For all other types `Level` must be 0. If `Level` is invalid, then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9ResourceGetMappedSize ( size_t*Â pSize, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )


Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pSize`
    \- Returned size of subresource
`pResource`
    \- Mapped resource to access
`Face`
    \- Face of resource to access
`Level`
    \- Level of resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000088>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pSize` the size of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `Face` and `Level`. The value set in `pSize` may change every time that `pResource` is mapped.

If `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D9_REGISTER_FLAGS_NONE, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped for access by CUDA, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

For usage requirements of `Face` and `Level` parameters, see [cuD3D9ResourceGetMappedPointer](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g7fb080fc9497d1f25fe1e3a226c56bdc> "Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9ResourceGetSurfaceDimensions ( size_t*Â pWidth, size_t*Â pHeight, size_t*Â pDepth, IDirect3DResource9*Â pResource, unsigned int Â Face, unsigned int Â Level )


Get the dimensions of a registered surface.

######  Parameters

`pWidth`
    \- Returned width of surface
`pHeight`
    \- Returned height of surface
`pDepth`
    \- Returned depth of surface
`pResource`
    \- Registered resource to access
`Face`
    \- Face of resource to access
`Level`
    \- Level of resource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000085>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pWidth`, `*pHeight`, and `*pDepth` the dimensions of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `Face` and `Level`.

Because anti-aliased surfaces may have multiple samples per pixel, it is possible that the dimensions of a resource will be an integer factor larger than the dimensions reported by the Direct3D runtime.

The parameters `pWidth`, `pHeight`, and `pDepth` are optional. For 2D surfaces, the value returned in `*pDepth` will be 0.

If `pResource` is not of type IDirect3DBaseTexture9 or IDirect3DSurface9 or if `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned.

For usage requirements of `Face` and `Level` parameters, see [cuD3D9ResourceGetMappedPointer()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1g7fb080fc9497d1f25fe1e3a226c56bdc> "Get the pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9ResourceSetMapFlags ( IDirect3DResource9*Â pResource, unsigned int Â Flags )


Set usage flags for mapping a Direct3D resource.

######  Parameters

`pResource`
    \- Registered resource to set flags for
`Flags`
    \- Parameters for resource mapping

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000084>)

This function is deprecated as of Cuda 3.0.

###### Description

Set `Flags` for mapping the Direct3D resource `pResource`.

Changes to `Flags` will take effect the next time `pResource` is mapped. The `Flags` argument may be any of the following:

  * CU_D3D9_MAPRESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA kernels. This is the default value.

  * CU_D3D9_MAPRESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which access this resource will not write to this resource.

  * CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels which access this resource will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


If `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is presently mapped for access by CUDA, then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9UnmapResources ( unsigned int Â count, IDirect3DResource9**Â ppResource )


Unmaps Direct3D resources.

######  Parameters

`count`
    \- Number of resources to unmap for CUDA
`ppResource`
    \- Resources to unmap for CUDA

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000083>)

This function is deprecated as of CUDA 3.0.

###### Description

Unmaps the `count` Direct3D resources in `ppResource`.

This function provides the synchronization guarantee that any CUDA kernels issued before [cuD3D9UnmapResources()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1gb985949a449e21274dd346c538619afe> "Unmaps Direct3D resources.") will complete before any Direct3D calls issued after [cuD3D9UnmapResources()](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED_1gb985949a449e21274dd346c538619afe> "Unmaps Direct3D resources.") begin.

If any of `ppResource` have not been registered for use with CUDA or if `ppResource` contains any duplicate entries, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If any of `ppResource` are not presently mapped for access by CUDA, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9UnregisterResource ( IDirect3DResource9*Â pResource )


Unregister a Direct3D resource.

######  Parameters

`pResource`
    \- Resource to unregister

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000081>)

This function is deprecated as of CUDA 3.0.

###### Description

Unregisters the Direct3D resource `pResource` so it is not accessible by CUDA unless registered again.

If `pResource` is not registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA.")

* * *
