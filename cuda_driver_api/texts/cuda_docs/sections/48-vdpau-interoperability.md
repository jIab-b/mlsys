# VDPAU Interoperability

## 6.44.Â VDPAU Interoperability

This section describes the VDPAU interoperability functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsVDPAURegisterOutputSurface](<#group__CUDA__VDPAU_1g54874c7f771e51f27292a562c92cee28>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, VdpOutputSurfaceÂ vdpSurface, unsigned int Â flags )
     Registers a VDPAU VdpOutputSurface object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsVDPAURegisterVideoSurface](<#group__CUDA__VDPAU_1ga5e00ff2d3ff2f8b680a69f3bc5cd891>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, VdpVideoSurfaceÂ vdpSurface, unsigned int Â flags )
     Registers a VDPAU VdpVideoSurface object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuVDPAUCtxCreate](<#group__CUDA__VDPAU_1gbfca4396947b5f194901c279cc6973d4>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, VdpDeviceÂ vdpDevice, VdpGetProcAddress*Â vdpGetProcAddress )
     Create a CUDA context for interoperability with VDPAU.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuVDPAUGetDevice](<#group__CUDA__VDPAU_1g0cce87525545da2cf1e84e007d5fe230>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pDevice, VdpDeviceÂ vdpDevice, VdpGetProcAddress*Â vdpGetProcAddress )
     Gets the CUDA device associated with a VDPAU device.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsVDPAURegisterOutputSurface ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, VdpOutputSurfaceÂ vdpSurface, unsigned int Â flags )


Registers a VDPAU VdpOutputSurface object.

######  Parameters

`pCudaResource`
    \- Pointer to the returned object handle
`vdpSurface`
    \- The VdpOutputSurface to be registered
`flags`
    \- Map flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Registers the VdpOutputSurface specified by `vdpSurface` for access by CUDA. A handle to the registered object is returned as `pCudaResource`. The surface's intended usage is specified using `flags`, as follows:

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA. This is the default value.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA will not write to this resource.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


The VdpOutputSurface is presented as an array of subresources that may be accessed using pointers returned by [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."). The exact number of valid `arrayIndex` values depends on the VDPAU surface format. The mapping is shown in the table below. `mipLevel` must be 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuVDPAUCtxCreate](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1gbfca4396947b5f194901c279cc6973d4> "Create a CUDA context for interoperability with VDPAU."), [cuGraphicsVDPAURegisterVideoSurface](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1ga5e00ff2d3ff2f8b680a69f3bc5cd891> "Registers a VDPAU VdpVideoSurface object."), [cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuVDPAUGetDevice](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1g0cce87525545da2cf1e84e007d5fe230> "Gets the CUDA device associated with a VDPAU device."), [cudaGraphicsVDPAURegisterOutputSurface](<../cuda-runtime-api/group__CUDART__VDPAU.html#group__CUDART__VDPAU_1gda9802a968253275d1d79f54debf5f6e>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsVDPAURegisterVideoSurface ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, VdpVideoSurfaceÂ vdpSurface, unsigned int Â flags )


Registers a VDPAU VdpVideoSurface object.

######  Parameters

`pCudaResource`
    \- Pointer to the returned object handle
`vdpSurface`
    \- The VdpVideoSurface to be registered
`flags`
    \- Map flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Registers the VdpVideoSurface specified by `vdpSurface` for access by CUDA. A handle to the registered object is returned as `pCudaResource`. The surface's intended usage is specified using `flags`, as follows:

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA. This is the default value.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA will not write to this resource.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


The VdpVideoSurface is presented as an array of subresources that may be accessed using pointers returned by [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."). The exact number of valid `arrayIndex` values depends on the VDPAU surface format. The mapping is shown in the table below. `mipLevel` must be 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuVDPAUCtxCreate](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1gbfca4396947b5f194901c279cc6973d4> "Create a CUDA context for interoperability with VDPAU."), [cuGraphicsVDPAURegisterOutputSurface](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1g54874c7f771e51f27292a562c92cee28> "Registers a VDPAU VdpOutputSurface object."), [cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuVDPAUGetDevice](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1g0cce87525545da2cf1e84e007d5fe230> "Gets the CUDA device associated with a VDPAU device."), [cudaGraphicsVDPAURegisterVideoSurface](<../cuda-runtime-api/group__CUDART__VDPAU.html#group__CUDART__VDPAU_1g0cd4adc9fe3f324927c1719b29ec12fb>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuVDPAUCtxCreate ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, VdpDeviceÂ vdpDevice, VdpGetProcAddress*Â vdpGetProcAddress )


Create a CUDA context for interoperability with VDPAU.

######  Parameters

`pCtx`
    \- Returned CUDA context
`flags`
    \- Options for CUDA context creation
`device`
    \- Device on which to create the context
`vdpDevice`
    \- The VdpDevice to interop with
`vdpGetProcAddress`
    \- VDPAU's VdpGetProcAddress function pointer

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates a new CUDA context, initializes VDPAU interoperability, and associates the CUDA context with the calling thread. It must be called before performing any other VDPAU interoperability operations. It may fail if the needed VDPAU driver facilities are not available. For usage of the `flags` parameter, see [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuGraphicsVDPAURegisterVideoSurface](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1ga5e00ff2d3ff2f8b680a69f3bc5cd891> "Registers a VDPAU VdpVideoSurface object."), [cuGraphicsVDPAURegisterOutputSurface](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1g54874c7f771e51f27292a562c92cee28> "Registers a VDPAU VdpOutputSurface object."), [cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuVDPAUGetDevice](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1g0cce87525545da2cf1e84e007d5fe230> "Gets the CUDA device associated with a VDPAU device.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuVDPAUGetDevice ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pDevice, VdpDeviceÂ vdpDevice, VdpGetProcAddress*Â vdpGetProcAddress )


Gets the CUDA device associated with a VDPAU device.

######  Parameters

`pDevice`
    \- Device associated with vdpDevice
`vdpDevice`
    \- A VdpDevice handle
`vdpGetProcAddress`
    \- VDPAU's VdpGetProcAddress function pointer

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*pDevice` the CUDA device associated with a `vdpDevice`, if applicable.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuVDPAUCtxCreate](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1gbfca4396947b5f194901c279cc6973d4> "Create a CUDA context for interoperability with VDPAU."), [cuGraphicsVDPAURegisterVideoSurface](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1ga5e00ff2d3ff2f8b680a69f3bc5cd891> "Registers a VDPAU VdpVideoSurface object."), [cuGraphicsVDPAURegisterOutputSurface](<group__CUDA__VDPAU.html#group__CUDA__VDPAU_1g54874c7f771e51f27292a562c92cee28> "Registers a VDPAU VdpOutputSurface object."), [cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cudaVDPAUGetDevice](<../cuda-runtime-api/group__CUDART__VDPAU.html#group__CUDART__VDPAU_1g242a0ba3eef80229ac3702e05f9eb1d9>)

* * *
