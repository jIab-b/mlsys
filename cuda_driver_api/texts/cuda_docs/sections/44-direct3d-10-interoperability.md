# Direct3D 10 Interoperability

## 6.42.Â Direct3D 10 Interoperability

This section describes the Direct3D 10 interoperability functions of the low-level CUDA driver application programming interface. Note that mapping of Direct3D 10 resources is performed with the graphics API agnostic, resource mapping interface described in [Graphics Interoperability](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS>).

### Modules

Â

[Direct3D 10 Interoperability [DEPRECATED]](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED>)

     [](<group__CUDA__D3D10__DEPRECATED.html#group__CUDA__D3D10__DEPRECATED>)

### Enumerations

enumÂ [CUd3d10DeviceList](<#group__CUDA__D3D10_1g6c6dadb6de5493a111271be299f62b99>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10GetDevice](<#group__CUDA__D3D10_1g98e0c9dcac9771d45112053045e0c34f>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, IDXGIAdapter*Â pAdapter )
     Gets the CUDA device corresponding to a display adapter.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D10GetDevices](<#group__CUDA__D3D10_1gdcc33dea972d5b834f45a0acefe5fe77>) ( unsigned int*Â pCudaDeviceCount, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevices, unsigned int Â cudaDeviceCount, ID3D10Device*Â pD3D10Device, [CUd3d10DeviceList](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g6c6dadb6de5493a111271be299f62b99>)Â deviceList )
     Gets the CUDA devices corresponding to a Direct3D 10 device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsD3D10RegisterResource](<#group__CUDA__D3D10_1g87fb2a189c27c4b63538d23f53b2c8e6>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, ID3D10Resource*Â pD3DResource, unsigned int Â Flags )
     Register a Direct3D 10 resource for access by CUDA.

### Enumerations

enum CUd3d10DeviceList


CUDA devices corresponding to a D3D10 device

######  Values

CU_D3D10_DEVICE_LIST_ALL = 0x01
    The CUDA devices for all GPUs used by a D3D10 device
CU_D3D10_DEVICE_LIST_CURRENT_FRAME = 0x02
    The CUDA devices for the GPUs used by a D3D10 device in its currently rendering frame
CU_D3D10_DEVICE_LIST_NEXT_FRAME = 0x03
    The CUDA devices for the GPUs to be used by a D3D10 device in the next frame

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10GetDevice ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, IDXGIAdapter*Â pAdapter )


Gets the CUDA device corresponding to a display adapter.

######  Parameters

`pCudaDevice`
    \- Returned CUDA device corresponding to `pAdapter`
`pAdapter`
    \- Adapter to query for CUDA device

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*pCudaDevice` the CUDA-compatible device corresponding to the adapter `pAdapter` obtained from IDXGIFactory::EnumAdapters.

If no device on `pAdapter` is CUDA-compatible then the call will fail.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D10GetDevices](<group__CUDA__D3D10.html#group__CUDA__D3D10_1gdcc33dea972d5b834f45a0acefe5fe77> "Gets the CUDA devices corresponding to a Direct3D 10 device."), [cudaD3D10GetDevice](<../cuda-runtime-api/group__CUDART__D3D10.html#group__CUDART__D3D10_1g9c053ca39e9c4a3dfcc65326db155fa6>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D10GetDevices ( unsigned int*Â pCudaDeviceCount, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevices, unsigned int Â cudaDeviceCount, ID3D10Device*Â pD3D10Device, [CUd3d10DeviceList](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g6c6dadb6de5493a111271be299f62b99>)Â deviceList )


Gets the CUDA devices corresponding to a Direct3D 10 device.

######  Parameters

`pCudaDeviceCount`
    \- Returned number of CUDA devices corresponding to `pD3D10Device`
`pCudaDevices`
    \- Returned CUDA devices corresponding to `pD3D10Device`
`cudaDeviceCount`
    \- The size of the output device array `pCudaDevices`
`pD3D10Device`
    \- Direct3D 10 device to query for CUDA devices
`deviceList`
    \- The set of devices to return. This set may be [CU_D3D10_DEVICE_LIST_ALL](<group__CUDA__D3D10.html#group__CUDA__D3D10_1gg6c6dadb6de5493a111271be299f62b99245d88858362d72f1e235d99895cb2f4>) for all devices, [CU_D3D10_DEVICE_LIST_CURRENT_FRAME](<group__CUDA__D3D10.html#group__CUDA__D3D10_1gg6c6dadb6de5493a111271be299f62b9955e91512320c16f4334e3607a10ae61d>) for the devices used to render the current frame (in SLI), or [CU_D3D10_DEVICE_LIST_NEXT_FRAME](<group__CUDA__D3D10.html#group__CUDA__D3D10_1gg6c6dadb6de5493a111271be299f62b99049f702f1ca930b270efbb55b4b0093c>) for the devices used to render the next frame (in SLI).

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NO_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9761bea84083d384d4fb88d51d972aada>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*pCudaDeviceCount` the number of CUDA-compatible device corresponding to the Direct3D 10 device `pD3D10Device`. Also returns in `*pCudaDevices` at most `cudaDeviceCount` of the CUDA-compatible devices corresponding to the Direct3D 10 device `pD3D10Device`.

If any of the GPUs being used to render `pDevice` are not CUDA capable then the call will return [CUDA_ERROR_NO_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9761bea84083d384d4fb88d51d972aada>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D10GetDevice](<group__CUDA__D3D10.html#group__CUDA__D3D10_1g98e0c9dcac9771d45112053045e0c34f> "Gets the CUDA device corresponding to a display adapter."), [cudaD3D10GetDevices](<../cuda-runtime-api/group__CUDART__D3D10.html#group__CUDART__D3D10_1g70ccbdf2ed995cb7182eb97c6780996d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsD3D10RegisterResource ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, ID3D10Resource*Â pD3DResource, unsigned int Â Flags )


Register a Direct3D 10 resource for access by CUDA.

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

Registers the Direct3D 10 resource `pD3DResource` for access by CUDA and returns a CUDA handle to `pD3Dresource` in `pCudaResource`. The handle returned in `pCudaResource` may be used to map and unmap this resource until it is unregistered. On success this call will increase the internal reference count on `pD3DResource`. This reference count will be decremented when this resource is unregistered through [cuGraphicsUnregisterResource()](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA.").

This call is potentially high-overhead and should not be called every frame in interactive applications.

The type of `pD3DResource` must be one of the following.

  * ID3D10Buffer: may be accessed through a device pointer.

  * ID3D10Texture1D: individual subresources of the texture may be accessed via arrays

  * ID3D10Texture2D: individual subresources of the texture may be accessed via arrays

  * ID3D10Texture3D: individual subresources of the texture may be accessed via arrays


The `Flags` argument may be used to specify additional parameters at register time. The valid values for this parameter are

  * CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be used.

  * CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will bind this resource to a surface reference.

  * CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform texture gather operations on this resource.


Not all Direct3D resources of the above types may be used for interoperability with CUDA. The following are some limitations.

  * The primary rendertarget may not be registered with CUDA.

  * Textures which are not of a format which is 1, 2, or 4 channels of 8, 16, or 32-bit integer or floating-point data cannot be shared.

  * Surfaces of depth or stencil formats cannot be shared.


A complete list of supported DXGI formats is as follows. For compactness the notation A_{B,C,D} represents A_B, A_C, and A_D.

  * DXGI_FORMAT_A8_UNORM

  * DXGI_FORMAT_B8G8R8A8_UNORM

  * DXGI_FORMAT_B8G8R8X8_UNORM

  * DXGI_FORMAT_R16_FLOAT

  * DXGI_FORMAT_R16G16B16A16_{FLOAT,SINT,SNORM,UINT,UNORM}

  * DXGI_FORMAT_R16G16_{FLOAT,SINT,SNORM,UINT,UNORM}

  * DXGI_FORMAT_R16_{SINT,SNORM,UINT,UNORM}

  * DXGI_FORMAT_R32_FLOAT

  * DXGI_FORMAT_R32G32B32A32_{FLOAT,SINT,UINT}

  * DXGI_FORMAT_R32G32_{FLOAT,SINT,UINT}

  * DXGI_FORMAT_R32_{SINT,UINT}

  * DXGI_FORMAT_R8G8B8A8_{SINT,SNORM,UINT,UNORM,UNORM_SRGB}

  * DXGI_FORMAT_R8G8_{SINT,SNORM,UINT,UNORM}

  * DXGI_FORMAT_R8_{SINT,SNORM,UINT,UNORM}


If `pD3DResource` is of incorrect type or is already registered then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `pD3DResource` cannot be registered then [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>) is returned. If `Flags` is not one of the above specified value then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cudaGraphicsD3D10RegisterResource](<../cuda-runtime-api/group__CUDART__D3D10.html#group__CUDART__D3D10_1g438731f7af2b799fb757910be6cff62b>)

### Direct3D 10 Interoperability [DEPRECATED]

* * *
