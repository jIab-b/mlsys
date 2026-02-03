# Direct3D 10 (Deprecated)

## 6.42.1.Â Direct3D 10 Interoperability [DEPRECATED]

## [[Direct3D 10 Interoperability](<group__CUDA__D3D10.html#group__CUDA__D3D10>)]

This section describes deprecated Direct3D 10 interoperability functionality.

### Enumerations

enumÂ [CUD3D10map_flags](<#group__CUDA__D3D10__DEPRECATED_1g74452bd2ef8d2515e627a4b54ca44394>)

enumÂ [CUD3D10register_flags](<#group__CUDA__D3D10__DEPRECATED_1gbad99be7f2d194fda482bae2c34286ad>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10CtxCreate](<#group__CUDA__D3D10__DEPRECATED_1gceb8ac9cabf2dae3f6185be772e36e95>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, unsigned int Â Flags, ID3D10Device*Â pD3DDevice )
     Create a CUDA context for interoperability with Direct3D 10.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10CtxCreateOnDevice](<#group__CUDA__D3D10__DEPRECATED_1gcac8d5f82332089f34fee44831477ae4>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, ID3D10Device*Â pD3DDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â cudaDevice )
     Create a CUDA context for interoperability with Direct3D 10.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10GetDirect3DDevice](<#group__CUDA__D3D10__DEPRECATED_1ga3b842b3a35adab2fec4febb44ed6251>) ( ID3D10Device**Â ppD3DDevice )
     Get the Direct3D 10 device against which the current CUDA context was created.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10MapResources](<#group__CUDA__D3D10__DEPRECATED_1g5c8ecc921f0830b3163a0f32ccd7511d>) ( unsigned int Â count, ID3D10Resource**Â ppResources )
     Map Direct3D resources for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10RegisterResource](<#group__CUDA__D3D10__DEPRECATED_1g476a6f370797a72d0238898e5d3e93ce>) ( ID3D10Resource*Â pResource, unsigned int Â Flags )
     Register a Direct3D resource for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10ResourceGetMappedArray](<#group__CUDA__D3D10__DEPRECATED_1ge10b1c832c2f8ac54cf72aa1dca8ad0f>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pArray, ID3D10Resource*Â pResource, unsigned int Â SubResource )
     Get an array through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10ResourceGetMappedPitch](<#group__CUDA__D3D10__DEPRECATED_1ga5c9af1165e0f783a123f4ae8ceb3379>) ( size_t*Â pPitch, size_t*Â pPitchSlice, ID3D10Resource*Â pResource, unsigned int Â SubResource )
     Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10ResourceGetMappedPointer](<#group__CUDA__D3D10__DEPRECATED_1ged2d8b89638fb2355e1ba2d7b92e0ff1>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pDevPtr, ID3D10Resource*Â pResource, unsigned int Â SubResource )
     Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10ResourceGetMappedSize](<#group__CUDA__D3D10__DEPRECATED_1g1c0069e431c8f95fd85fd3379cf7cb0e>) ( size_t*Â pSize, ID3D10Resource*Â pResource, unsigned int Â SubResource )
     Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10ResourceGetSurfaceDimensions](<#group__CUDA__D3D10__DEPRECATED_1g7795faffd5e58e04f277263d310278fe>) ( size_t*Â pWidth, size_t*Â pHeight, size_t*Â pDepth, ID3D10Resource*Â pResource, unsigned int Â SubResource )
     Get the dimensions of a registered surface.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10ResourceSetMapFlags](<#group__CUDA__D3D10__DEPRECATED_1gcbfee49e43deebbcde1deea91d8e48fa>) ( ID3D10Resource*Â pResource, unsigned int Â Flags )
     Set usage flags for mapping a Direct3D resource.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10UnmapResources](<#group__CUDA__D3D10__DEPRECATED_1gbe4c93d0d53f16e843c035b2dd144a46>) ( unsigned int Â count, ID3D10Resource**Â ppResources )
     Unmap Direct3D resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10UnregisterResource](<#group__CUDA__D3D10__DEPRECATED_1gb4bb733df68b54424ac0b575e113e4ca>) ( ID3D10Resource*Â pResource )
     Unregister a Direct3D resource.

### Enumerations

enum CUD3D10map_flags


Flags to map or unmap a resource

######  Values

CU_D3D10_MAPRESOURCE_FLAGS_NONE = 0x00

CU_D3D10_MAPRESOURCE_FLAGS_READONLY = 0x01

CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02


enum CUD3D10register_flags


Flags to register a resource

######  Values

CU_D3D10_REGISTER_FLAGS_NONE = 0x00

CU_D3D10_REGISTER_FLAGS_ARRAY = 0x01


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10CtxCreate ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, unsigned int Â Flags, ID3D10Device*Â pD3DDevice )


Create a CUDA context for interoperability with Direct3D 10.

######  Parameters

`pCtx`
    \- Returned newly created CUDA context
`pCudaDevice`
    \- Returned pointer to the device on which the context was created
`Flags`
    \- Context creation flags (see [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") for details)
`pD3DDevice`
    \- Direct3D device to create interoperability context with

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000090>)

This function is deprecated as of CUDA 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with a D3D10 device in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D10GetDevice](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g98e0c9dcac9771d45112053045e0c34f> "Gets the CUDA device corresponding to a display adapter."), [cuGraphicsD3D10RegisterResource](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g87fb2a189c27c4b63538d23f53b2c8e6> "Register a Direct3D 10 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10CtxCreateOnDevice ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, ID3D10Device*Â pD3DDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â cudaDevice )


Create a CUDA context for interoperability with Direct3D 10.

######  Parameters

`pCtx`
    \- Returned newly created CUDA context
`flags`
    \- Context creation flags (see [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") for details)
`pD3DDevice`
    \- Direct3D device to create interoperability context with
`cudaDevice`
    \- The CUDA device on which to create the context. This device must be among the devices returned when querying CU_D3D10_DEVICES_ALL from [cuD3D10GetDevices](<group__CUDA__D3D10.html#group__CUDA__D3D10_1gdcc33dea972d5b834f45a0acefe5fe77> "Gets the CUDA devices corresponding to a Direct3D 10 device.").

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000091>)

This function is deprecated as of CUDA 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with a D3D10 device in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D10GetDevices](<group__CUDA__D3D10.html#group__CUDA__D3D10_1gdcc33dea972d5b834f45a0acefe5fe77> "Gets the CUDA devices corresponding to a Direct3D 10 device."), [cuGraphicsD3D10RegisterResource](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g87fb2a189c27c4b63538d23f53b2c8e6> "Register a Direct3D 10 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10GetDirect3DDevice ( ID3D10Device**Â ppD3DDevice )


Get the Direct3D 10 device against which the current CUDA context was created.

######  Parameters

`ppD3DDevice`
    \- Returned Direct3D device corresponding to CUDA context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000092>)

This function is deprecated as of CUDA 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with a D3D10 device in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D10GetDevice](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g98e0c9dcac9771d45112053045e0c34f> "Gets the CUDA device corresponding to a display adapter.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10MapResources ( unsigned int Â count, ID3D10Resource**Â ppResources )


Map Direct3D resources for access by CUDA.

######  Parameters

`count`
    \- Number of resources to map for CUDA
`ppResources`
    \- Resources to map for CUDA

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000095>)

This function is deprecated as of CUDA 3.0.

###### Description

Maps the `count` Direct3D resources in `ppResources` for access by CUDA.

The resources in `ppResources` may be accessed in CUDA kernels until they are unmapped. Direct3D should not access any resources while they are mapped by CUDA. If an application does so, the results are undefined.

This function provides the synchronization guarantee that any Direct3D calls issued before [cuD3D10MapResources()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1g5c8ecc921f0830b3163a0f32ccd7511d> "Map Direct3D resources for access by CUDA.") will complete before any CUDA kernels issued after [cuD3D10MapResources()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1g5c8ecc921f0830b3163a0f32ccd7511d> "Map Direct3D resources for access by CUDA.") begin.

If any of `ppResources` have not been registered for use with CUDA or if `ppResources` contains any duplicate entries, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If any of `ppResources` are presently mapped for access by CUDA, then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10RegisterResource ( ID3D10Resource*Â pResource, unsigned int Â Flags )


Register a Direct3D resource for access by CUDA.

######  Parameters

`pResource`
    \- Resource to register
`Flags`
    \- Parameters for resource registration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000093>)

This function is deprecated as of CUDA 3.0.

###### Description

Registers the Direct3D resource `pResource` for access by CUDA.

If this call is successful, then the application will be able to map and unmap this resource until it is unregistered through [cuD3D10UnregisterResource()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1gb4bb733df68b54424ac0b575e113e4ca> "Unregister a Direct3D resource."). Also on success, this call will increase the internal reference count on `pResource`. This reference count will be decremented when this resource is unregistered through [cuD3D10UnregisterResource()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1gb4bb733df68b54424ac0b575e113e4ca> "Unregister a Direct3D resource.").

This call is potentially high-overhead and should not be called every frame in interactive applications.

The type of `pResource` must be one of the following.

  * ID3D10Buffer: Cannot be used with `Flags` set to CU_D3D10_REGISTER_FLAGS_ARRAY.

  * ID3D10Texture1D: No restrictions.

  * ID3D10Texture2D: No restrictions.

  * ID3D10Texture3D: No restrictions.


The `Flags` argument specifies the mechanism through which CUDA will access the Direct3D resource. The following values are allowed.

  * CU_D3D10_REGISTER_FLAGS_NONE: Specifies that CUDA will access this resource through a [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>). The pointer, size, and (for textures), pitch for each subresource of this allocation may be queried through [cuD3D10ResourceGetMappedPointer()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ged2d8b89638fb2355e1ba2d7b92e0ff1> "Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA."), [cuD3D10ResourceGetMappedSize()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1g1c0069e431c8f95fd85fd3379cf7cb0e> "Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA."), and [cuD3D10ResourceGetMappedPitch()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ga5c9af1165e0f783a123f4ae8ceb3379> "Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.") respectively. This option is valid for all resource types.

  * CU_D3D10_REGISTER_FLAGS_ARRAY: Specifies that CUDA will access this resource through a [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) queried on a sub-resource basis through [cuD3D10ResourceGetMappedArray()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ge10b1c832c2f8ac54cf72aa1dca8ad0f> "Get an array through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA."). This option is only valid for resources of type ID3D10Texture1D, ID3D10Texture2D, and ID3D10Texture3D.


Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some limitations.

  * The primary rendertarget may not be registered with CUDA.

  * Resources allocated as shared may not be registered with CUDA.

  * Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data cannot be shared.

  * Surfaces of depth or stencil formats cannot be shared.


If Direct3D interoperability is not initialized on this context then [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned. If `pResource` is of incorrect type or is already registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` cannot be registered, then [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsD3D10RegisterResource](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g87fb2a189c27c4b63538d23f53b2c8e6> "Register a Direct3D 10 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10ResourceGetMappedArray ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pArray, ID3D10Resource*Â pResource, unsigned int Â SubResource )


Get an array through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pArray`
    \- Returned array corresponding to subresource
`pResource`
    \- Mapped resource to access
`SubResource`
    \- Subresource of pResource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000098>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pArray` an array through which the subresource of the mapped Direct3D resource `pResource`, which corresponds to `SubResource` may be accessed. The value set in `pArray` may change every time that `pResource` is mapped.

If `pResource` is not registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D10_REGISTER_FLAGS_ARRAY, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

For usage requirements of the `SubResource` parameter, see [cuD3D10ResourceGetMappedPointer()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ged2d8b89638fb2355e1ba2d7b92e0ff1> "Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10ResourceGetMappedPitch ( size_t*Â pPitch, size_t*Â pPitchSlice, ID3D10Resource*Â pResource, unsigned int Â SubResource )


Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pPitch`
    \- Returned pitch of subresource
`pPitchSlice`
    \- Returned Z-slice pitch of subresource
`pResource`
    \- Mapped resource to access
`SubResource`
    \- Subresource of pResource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000101>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pPitch` and `*pPitchSlice` the pitch and Z-slice pitch of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `SubResource`. The values set in `pPitch` and `pPitchSlice` may change every time that `pResource` is mapped.

The pitch and Z-slice pitch values may be used to compute the location of a sample on a surface as follows.

For a 2D surface, the byte offset of the sample at position **x** , **y** from the base pointer of the surface is:

**y** * **pitch** \+ (**bytes per pixel**) * **x**

For a 3D surface, the byte offset of the sample at position **x** , **y** , **z** from the base pointer of the surface is:

**z*** **slicePitch** \+ **y** * **pitch** \+ (**bytes per pixel**) * **x**

Both parameters `pPitch` and `pPitchSlice` are optional and may be set to NULL.

If `pResource` is not of type IDirect3DBaseTexture10 or one of its sub-types or if `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D10_REGISTER_FLAGS_NONE, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped for access by CUDA, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

For usage requirements of the `SubResource` parameter, see [cuD3D10ResourceGetMappedPointer()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ged2d8b89638fb2355e1ba2d7b92e0ff1> "Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10ResourceGetMappedPointer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pDevPtr, ID3D10Resource*Â pResource, unsigned int Â SubResource )


Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pDevPtr`
    \- Returned pointer corresponding to subresource
`pResource`
    \- Mapped resource to access
`SubResource`
    \- Subresource of pResource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000099>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pDevPtr` the base pointer of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `SubResource`. The value set in `pDevPtr` may change every time that `pResource` is mapped.

If `pResource` is not registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D10_REGISTER_FLAGS_NONE, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

If `pResource` is of type ID3D10Buffer, then `SubResource` must be 0. If `pResource` is of any other type, then the value of `SubResource` must come from the subresource calculation in D3D10CalcSubResource().

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10ResourceGetMappedSize ( size_t*Â pSize, ID3D10Resource*Â pResource, unsigned int Â SubResource )


Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA.

######  Parameters

`pSize`
    \- Returned size of subresource
`pResource`
    \- Mapped resource to access
`SubResource`
    \- Subresource of pResource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000100>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pSize` the size of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `SubResource`. The value set in `pSize` may change every time that `pResource` is mapped.

If `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` was not registered with usage flags CU_D3D10_REGISTER_FLAGS_NONE, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is not mapped for access by CUDA, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

For usage requirements of the `SubResource` parameter, see [cuD3D10ResourceGetMappedPointer()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ged2d8b89638fb2355e1ba2d7b92e0ff1> "Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10ResourceGetSurfaceDimensions ( size_t*Â pWidth, size_t*Â pHeight, size_t*Â pDepth, ID3D10Resource*Â pResource, unsigned int Â SubResource )


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
`SubResource`
    \- Subresource of pResource to access

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000102>)

This function is deprecated as of CUDA 3.0.

###### Description

Returns in `*pWidth`, `*pHeight`, and `*pDepth` the dimensions of the subresource of the mapped Direct3D resource `pResource`, which corresponds to `SubResource`.

Because anti-aliased surfaces may have multiple samples per pixel, it is possible that the dimensions of a resource will be an integer factor larger than the dimensions reported by the Direct3D runtime.

The parameters `pWidth`, `pHeight`, and `pDepth` are optional. For 2D surfaces, the value returned in `*pDepth` will be 0.

If `pResource` is not of type IDirect3DBaseTexture10 or IDirect3DSurface10 or if `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned.

For usage requirements of the `SubResource` parameter, see [cuD3D10ResourceGetMappedPointer()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1ged2d8b89638fb2355e1ba2d7b92e0ff1> "Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for access by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10ResourceSetMapFlags ( ID3D10Resource*Â pResource, unsigned int Â Flags )


Set usage flags for mapping a Direct3D resource.

######  Parameters

`pResource`
    \- Registered resource to set flags for
`Flags`
    \- Parameters for resource mapping

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000097>)

This function is deprecated as of CUDA 3.0.

###### Description

Set flags for mapping the Direct3D resource `pResource`.

Changes to flags will take effect the next time `pResource` is mapped. The `Flags` argument may be any of the following.

  * CU_D3D10_MAPRESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA kernels. This is the default value.

  * CU_D3D10_MAPRESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which access this resource will not write to this resource.

  * CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels which access this resource will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


If `pResource` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pResource` is presently mapped for access by CUDA then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10UnmapResources ( unsigned int Â count, ID3D10Resource**Â ppResources )


Unmap Direct3D resources.

######  Parameters

`count`
    \- Number of resources to unmap for CUDA
`ppResources`
    \- Resources to unmap for CUDA

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000096>)

This function is deprecated as of CUDA 3.0.

###### Description

Unmaps the `count` Direct3D resources in `ppResources`.

This function provides the synchronization guarantee that any CUDA kernels issued before [cuD3D10UnmapResources()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1gbe4c93d0d53f16e843c035b2dd144a46> "Unmap Direct3D resources.") will complete before any Direct3D calls issued after [cuD3D10UnmapResources()](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED_1gbe4c93d0d53f16e843c035b2dd144a46> "Unmap Direct3D resources.") begin.

If any of `ppResources` have not been registered for use with CUDA or if `ppResources` contains any duplicate entries, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If any of `ppResources` are not presently mapped for access by CUDA, then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10UnregisterResource ( ID3D10Resource*Â pResource )


Unregister a Direct3D resource.

######  Parameters

`pResource`
    \- Resources to unregister

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000094>)

This function is deprecated as of CUDA 3.0.

###### Description

Unregisters the Direct3D resource `pResource` so it is not accessible by CUDA unless registered again.

If `pResource` is not registered, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA.")

* * *
