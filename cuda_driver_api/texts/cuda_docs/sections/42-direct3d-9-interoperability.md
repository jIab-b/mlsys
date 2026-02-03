# Direct3D 9 Interoperability

## 6.41.Â Direct3D 9 Interoperability

This section describes the Direct3D 9 interoperability functions of the low-level CUDA driver application programming interface. Note that mapping of Direct3D 9 resources is performed with the graphics API agnostic, resource mapping interface described in [Graphics Interoperability](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS>).

### Modules

Â

[Direct3D 9 Interoperability [DEPRECATED]](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED>)

     [](<group__CUDA__D3D9__DEPRECATED.html#group__CUDA__D3D9__DEPRECATED>)

### Enumerations

enumÂ [CUd3d9DeviceList](<#group__CUDA__D3D9_1g2cf4b668539659fec0dba41bcc999f6d>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9CtxCreate](<#group__CUDA__D3D9_1gab201a2284d11b00cdb4d6bba492e520>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, unsigned int Â Flags, IDirect3DDevice9*Â pD3DDevice )
     Create a CUDA context for interoperability with Direct3D 9.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9CtxCreateOnDevice](<#group__CUDA__D3D9_1gcaca5329caf0c0253a5a944ecc958742>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, IDirect3DDevice9*Â pD3DDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â cudaDevice )
     Create a CUDA context for interoperability with Direct3D 9.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9GetDevice](<#group__CUDA__D3D9_1ge293c667e76dafaaf47ce64d0bd91c4d>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, const char*Â pszAdapterName )
     Gets the CUDA device corresponding to a display adapter.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9GetDevices](<#group__CUDA__D3D9_1g2c53ac0b20c57738fa497d1f8992f7ad>) ( unsigned int*Â pCudaDeviceCount, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevices, unsigned int Â cudaDeviceCount, IDirect3DDevice9*Â pD3D9Device, [CUd3d9DeviceList](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g2cf4b668539659fec0dba41bcc999f6d>)Â deviceList )
     Gets the CUDA devices corresponding to a Direct3D 9 device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D9GetDirect3DDevice](<#group__CUDA__D3D9_1g439e074e2b46156f859c40ddaaf3d3fb>) ( IDirect3DDevice9**Â ppD3DDevice )
     Get the Direct3D 9 device against which the current CUDA context was created.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsD3D9RegisterResource](<#group__CUDA__D3D9_1g391835e0d3c5a34bdba99840157194bf>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, IDirect3DResource9*Â pD3DResource, unsigned int Â Flags )
     Register a Direct3D 9 resource for access by CUDA.

### Enumerations

enum CUd3d9DeviceList


CUDA devices corresponding to a D3D9 device

######  Values

CU_D3D9_DEVICE_LIST_ALL = 0x01
    The CUDA devices for all GPUs used by a D3D9 device
CU_D3D9_DEVICE_LIST_CURRENT_FRAME = 0x02
    The CUDA devices for the GPUs used by a D3D9 device in its currently rendering frame
CU_D3D9_DEVICE_LIST_NEXT_FRAME = 0x03
    The CUDA devices for the GPUs to be used by a D3D9 device in the next frame

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9CtxCreate ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, unsigned int Â Flags, IDirect3DDevice9*Â pD3DDevice )


Create a CUDA context for interoperability with Direct3D 9.

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

###### Description

Creates a new CUDA context, enables interoperability for that context with the Direct3D device `pD3DDevice`, and associates the created CUDA context with the calling thread. The created [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) will be returned in `*pCtx`. Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context. If `pCudaDevice` is non-NULL then the [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) on which this CUDA context was created will be returned in `*pCudaDevice`.

On success, this call will increase the internal reference count on `pD3DDevice`. This reference count will be decremented upon destruction of this context through [cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."). This context will cease to function if `pD3DDevice` is destroyed or encounters an error.

Note that this function is never required for correct functionality. Use of this function will result in accelerated interoperability only when the operating system is Windows Vista or Windows 7, and the device `pD3DDdevice` is not an IDirect3DDevice9Ex. In all other circumstances, this function is not necessary.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D9GetDevice](<group__CUDA__D3D9.html#group__CUDA__D3D9_1ge293c667e76dafaaf47ce64d0bd91c4d> "Gets the CUDA device corresponding to a display adapter."), [cuGraphicsD3D9RegisterResource](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g391835e0d3c5a34bdba99840157194bf> "Register a Direct3D 9 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9CtxCreateOnDevice ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, IDirect3DDevice9*Â pD3DDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â cudaDevice )


Create a CUDA context for interoperability with Direct3D 9.

######  Parameters

`pCtx`
    \- Returned newly created CUDA context
`flags`
    \- Context creation flags (see [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") for details)
`pD3DDevice`
    \- Direct3D device to create interoperability context with
`cudaDevice`
    \- The CUDA device on which to create the context. This device must be among the devices returned when querying CU_D3D9_DEVICES_ALL from [cuD3D9GetDevices](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g2c53ac0b20c57738fa497d1f8992f7ad> "Gets the CUDA devices corresponding to a Direct3D 9 device.").

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Creates a new CUDA context, enables interoperability for that context with the Direct3D device `pD3DDevice`, and associates the created CUDA context with the calling thread. The created [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) will be returned in `*pCtx`. Direct3D resources from this device may be registered and mapped through the lifetime of this CUDA context.

On success, this call will increase the internal reference count on `pD3DDevice`. This reference count will be decremented upon destruction of this context through [cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."). This context will cease to function if `pD3DDevice` is destroyed or encounters an error.

Note that this function is never required for correct functionality. Use of this function will result in accelerated interoperability only when the operating system is Windows Vista or Windows 7, and the device `pD3DDdevice` is not an IDirect3DDevice9Ex. In all other circumstances, this function is not necessary.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D9GetDevices](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g2c53ac0b20c57738fa497d1f8992f7ad> "Gets the CUDA devices corresponding to a Direct3D 9 device."), [cuGraphicsD3D9RegisterResource](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g391835e0d3c5a34bdba99840157194bf> "Register a Direct3D 9 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9GetDevice ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, const char*Â pszAdapterName )


Gets the CUDA device corresponding to a display adapter.

######  Parameters

`pCudaDevice`
    \- Returned CUDA device corresponding to pszAdapterName
`pszAdapterName`
    \- Adapter name to query for device

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*pCudaDevice` the CUDA-compatible device corresponding to the adapter name `pszAdapterName` obtained from EnumDisplayDevices() or IDirect3D9::GetAdapterIdentifier().

If no device on the adapter with name `pszAdapterName` is CUDA-compatible, then the call will fail.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D9CtxCreate](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gab201a2284d11b00cdb4d6bba492e520> "Create a CUDA context for interoperability with Direct3D 9."), [cudaD3D9GetDevice](<../cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9_1gcd070306b3ce6540a3bc309d415f19b2>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9GetDevices ( unsigned int*Â pCudaDeviceCount, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevices, unsigned int Â cudaDeviceCount, IDirect3DDevice9*Â pD3D9Device, [CUd3d9DeviceList](<group__CUDA__D3D9.html#group__CUDA__D3D9_1g2cf4b668539659fec0dba41bcc999f6d>)Â deviceList )


Gets the CUDA devices corresponding to a Direct3D 9 device.

######  Parameters

`pCudaDeviceCount`
    \- Returned number of CUDA devices corresponding to `pD3D9Device`
`pCudaDevices`
    \- Returned CUDA devices corresponding to `pD3D9Device`
`cudaDeviceCount`
    \- The size of the output device array `pCudaDevices`
`pD3D9Device`
    \- Direct3D 9 device to query for CUDA devices
`deviceList`
    \- The set of devices to return. This set may be [CU_D3D9_DEVICE_LIST_ALL](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gg2cf4b668539659fec0dba41bcc999f6d7d126d9cf1514d7c41b38cf84113a4dd>) for all devices, [CU_D3D9_DEVICE_LIST_CURRENT_FRAME](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gg2cf4b668539659fec0dba41bcc999f6df8bd68d9d5a86fbf3c9e60ab2e607e41>) for the devices used to render the current frame (in SLI), or [CU_D3D9_DEVICE_LIST_NEXT_FRAME](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gg2cf4b668539659fec0dba41bcc999f6d111afda57f25cc5c51a02078ca0530f2>) for the devices used to render the next frame (in SLI).

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NO_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9761bea84083d384d4fb88d51d972aada>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*pCudaDeviceCount` the number of CUDA-compatible device corresponding to the Direct3D 9 device `pD3D9Device`. Also returns in `*pCudaDevices` at most `cudaDeviceCount` of the CUDA-compatible devices corresponding to the Direct3D 9 device `pD3D9Device`.

If any of the GPUs being used to render `pDevice` are not CUDA capable then the call will return [CUDA_ERROR_NO_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9761bea84083d384d4fb88d51d972aada>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D9CtxCreate](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gab201a2284d11b00cdb4d6bba492e520> "Create a CUDA context for interoperability with Direct3D 9."), [cudaD3D9GetDevices](<../cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9_1g113d44c4c588818e27de685a58412736>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D9GetDirect3DDevice ( IDirect3DDevice9**Â ppD3DDevice )


Get the Direct3D 9 device against which the current CUDA context was created.

######  Parameters

`ppD3DDevice`
    \- Returned Direct3D device corresponding to CUDA context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)[CUDA_ERROR_INVALID_GRAPHICS_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98667df54ca7b35fa911703978a679839>)

###### Description

Returns in `*ppD3DDevice` the Direct3D device against which this CUDA context was created in [cuD3D9CtxCreate()](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gab201a2284d11b00cdb4d6bba492e520> "Create a CUDA context for interoperability with Direct3D 9.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D9GetDevice](<group__CUDA__D3D9.html#group__CUDA__D3D9_1ge293c667e76dafaaf47ce64d0bd91c4d> "Gets the CUDA device corresponding to a display adapter."), [cudaD3D9GetDirect3DDevice](<../cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9_1g911fe6061c4e0015abf8124ac8e07582>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsD3D9RegisterResource ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, IDirect3DResource9*Â pD3DResource, unsigned int Â Flags )


Register a Direct3D 9 resource for access by CUDA.

######  Parameters

`pCudaResource`
    \- Returned graphics resource handle
`pD3DResource`
    \- Direct3D resource to register
`Flags`
    \- Parameters for resource registration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Registers the Direct3D 9 resource `pD3DResource` for access by CUDA and returns a CUDA handle to `pD3Dresource` in `pCudaResource`. The handle returned in `pCudaResource` may be used to map and unmap this resource until it is unregistered. On success this call will increase the internal reference count on `pD3DResource`. This reference count will be decremented when this resource is unregistered through [cuGraphicsUnregisterResource()](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA.").

This call is potentially high-overhead and should not be called every frame in interactive applications.

The type of `pD3DResource` must be one of the following.

  * IDirect3DVertexBuffer9: may be accessed through a device pointer

  * IDirect3DIndexBuffer9: may be accessed through a device pointer

  * IDirect3DSurface9: may be accessed through an array. Only stand-alone objects of type IDirect3DSurface9 may be explicitly shared. In particular, individual mipmap levels and faces of cube maps may not be registered directly. To access individual surfaces associated with a texture, one must register the base texture object.

  * IDirect3DBaseTexture9: individual surfaces on this texture may be accessed through an array.


The `Flags` argument may be used to specify additional parameters at register time. The valid values for this parameter are

  * CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be used.

  * CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will bind this resource to a surface reference.

  * CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform texture gather operations on this resource.


Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some limitations.

  * The primary rendertarget may not be registered with CUDA.

  * Resources allocated as shared may not be registered with CUDA.

  * Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data cannot be shared.

  * Surfaces of depth or stencil formats cannot be shared.


A complete list of supported formats is as follows:

  * D3DFMT_L8

  * D3DFMT_L16

  * D3DFMT_A8R8G8B8

  * D3DFMT_X8R8G8B8

  * D3DFMT_G16R16

  * D3DFMT_A8B8G8R8

  * D3DFMT_A8

  * D3DFMT_A8L8

  * D3DFMT_Q8W8V8U8

  * D3DFMT_V16U16

  * D3DFMT_A16B16G16R16F

  * D3DFMT_A16B16G16R16

  * D3DFMT_R32F

  * D3DFMT_G16R16F

  * D3DFMT_A32B32G32R32F

  * D3DFMT_G32R32F

  * D3DFMT_R16F


If Direct3D interoperability is not initialized for this context using [cuD3D9CtxCreate](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gab201a2284d11b00cdb4d6bba492e520> "Create a CUDA context for interoperability with Direct3D 9.") then [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned. If `pD3DResource` is of incorrect type or is already registered then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pD3DResource` cannot be registered then [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>) is returned. If `Flags` is not one of the above specified value then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D9CtxCreate](<group__CUDA__D3D9.html#group__CUDA__D3D9_1gab201a2284d11b00cdb4d6bba492e520> "Create a CUDA context for interoperability with Direct3D 9."), [cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cudaGraphicsD3D9RegisterResource](<../cuda-runtime-api/group__CUDART__D3D9.html#group__CUDART__D3D9_1gab5efa8a8882a6e0ee99717a434730b0>)

### Direct3D 9 Interoperability [DEPRECATED]

* * *
