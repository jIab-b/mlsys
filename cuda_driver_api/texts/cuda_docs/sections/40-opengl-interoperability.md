# OpenGL Interoperability

## 6.40.Â OpenGL Interoperability

This section describes the OpenGL interoperability functions of the low-level CUDA driver application programming interface. Note that mapping of OpenGL resources is performed with the graphics API agnostic, resource mapping interface described in [Graphics Interoperability](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS>).

### Modules

Â

[OpenGL Interoperability [DEPRECATED]](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED>)

     [](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED>)

### Enumerations

enumÂ [CUGLDeviceList](<#group__CUDA__GL_1g7676f0c02ef846176f6ef26accbb9e3b>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLGetDevices](<#group__CUDA__GL_1g98bb15525b04d2f6a817c21e07d8b7cd>) ( unsigned int*Â pCudaDeviceCount, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevices, unsigned int Â cudaDeviceCount, [CUGLDeviceList](<group__CUDA__GL.html#group__CUDA__GL_1g7676f0c02ef846176f6ef26accbb9e3b>)Â deviceList )
     Gets the CUDA devices associated with the current OpenGL context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsGLRegisterBuffer](<#group__CUDA__GL_1gd530f66cc9ab43a31a98527e75f343a0>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, GLuintÂ buffer, unsigned int Â Flags )
     Registers an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsGLRegisterImage](<#group__CUDA__GL_1g52c3a36c4c92611b6fcf0662b2f74e40>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, GLuintÂ image, GLenumÂ target, unsigned int Â Flags )
     Register an OpenGL texture or renderbuffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuWGLGetDevice](<#group__CUDA__GL_1g21ff8296192dc38dff42ba3346078282>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pDevice, HGPUNVÂ hGpu )
     Gets the CUDA device associated with hGpu.

### Enumerations

enum CUGLDeviceList


CUDA devices corresponding to an OpenGL device

######  Values

CU_GL_DEVICE_LIST_ALL = 0x01
    The CUDA devices for all GPUs used by the current OpenGL context
CU_GL_DEVICE_LIST_CURRENT_FRAME = 0x02
    The CUDA devices for the GPUs used by the current OpenGL context in its currently rendering frame
CU_GL_DEVICE_LIST_NEXT_FRAME = 0x03
    The CUDA devices for the GPUs to be used by the current OpenGL context in the next frame

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLGetDevices ( unsigned int*Â pCudaDeviceCount, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevices, unsigned int Â cudaDeviceCount, [CUGLDeviceList](<group__CUDA__GL.html#group__CUDA__GL_1g7676f0c02ef846176f6ef26accbb9e3b>)Â deviceList )


Gets the CUDA devices associated with the current OpenGL context.

######  Parameters

`pCudaDeviceCount`
    \- Returned number of CUDA devices.
`pCudaDevices`
    \- Returned CUDA devices.
`cudaDeviceCount`
    \- The size of the output device array pCudaDevices.
`deviceList`
    \- The set of devices to return.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NO_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9761bea84083d384d4fb88d51d972aada>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_GRAPHICS_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98667df54ca7b35fa911703978a679839>), [CUDA_ERROR_OPERATING_SYSTEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c849a151611f6e2ed1b3ae923f79ef3c>)

###### Description

Returns in `*pCudaDeviceCount` the number of CUDA-compatible devices corresponding to the current OpenGL context. Also returns in `*pCudaDevices` at most cudaDeviceCount of the CUDA-compatible devices corresponding to the current OpenGL context. If any of the GPUs being used by the current OpenGL context are not CUDA capable then the call will return CUDA_ERROR_NO_DEVICE.

The `deviceList` argument may be any of the following:

  * [CU_GL_DEVICE_LIST_ALL](<group__CUDA__GL.html#group__CUDA__GL_1gg7676f0c02ef846176f6ef26accbb9e3b43d2b311a54ae5c914c264e8395c3164>): Query all devices used by the current OpenGL context.

  * [CU_GL_DEVICE_LIST_CURRENT_FRAME](<group__CUDA__GL.html#group__CUDA__GL_1gg7676f0c02ef846176f6ef26accbb9e3bdc48f515019afdc0a78cd366871b0fe1>): Query the devices used by the current OpenGL context to render the current frame (in SLI).

  * [CU_GL_DEVICE_LIST_NEXT_FRAME](<group__CUDA__GL.html#group__CUDA__GL_1gg7676f0c02ef846176f6ef26accbb9e3b9d6d52b3167bd1d144eb08db8b60325a>): Query the devices used by the current OpenGL context to render the next frame (in SLI). Note that this is a prediction, it can't be guaranteed that this is correct in all cases.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuWGLGetDevice](<group__CUDA__GL.html#group__CUDA__GL_1g21ff8296192dc38dff42ba3346078282> "Gets the CUDA device associated with hGpu."), [cudaGLGetDevices](<../cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g3471ecaa5b827c94f2c55ab51fde1751>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsGLRegisterBuffer ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, GLuintÂ buffer, unsigned int Â Flags )


Registers an OpenGL buffer object.

######  Parameters

`pCudaResource`
    \- Pointer to the returned object handle
`buffer`
    \- name of buffer object to be registered
`Flags`
    \- Register flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_OPERATING_SYSTEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c849a151611f6e2ed1b3ae923f79ef3c>)

###### Description

Registers the buffer object specified by `buffer` for access by CUDA. A handle to the registered object is returned as `pCudaResource`. The register flags `Flags` specify the intended usage, as follows:

  * CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA. This is the default value.

  * CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that CUDA will not write to this resource.

  * CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cudaGraphicsGLRegisterBuffer](<../cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g0fd33bea77ca7b1e69d1619caf44214b>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsGLRegisterImage ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, GLuintÂ image, GLenumÂ target, unsigned int Â Flags )


Register an OpenGL texture or renderbuffer object.

######  Parameters

`pCudaResource`
    \- Pointer to the returned object handle
`image`
    \- name of texture or renderbuffer object to be registered
`target`
    \- Identifies the type of object specified by `image`
`Flags`
    \- Register flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_OPERATING_SYSTEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c849a151611f6e2ed1b3ae923f79ef3c>)

###### Description

Registers the texture or renderbuffer object specified by `image` for access by CUDA. A handle to the registered object is returned as `pCudaResource`.

`target` must match the type of the object, and must be one of GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE, GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.

The register flags `Flags` specify the intended usage, as follows:

  * CU_GRAPHICS_REGISTER_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA. This is the default value.

  * CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY: Specifies that CUDA will not write to this resource.

  * CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.

  * CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST: Specifies that CUDA will bind this resource to a surface reference.

  * CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER: Specifies that CUDA will perform texture gather operations on this resource.


The following image formats are supported. For brevity's sake, the list is abbreviated. For ex., {GL_R, GL_RG} X {8, 16} would expand to the following 4 formats {GL_R8, GL_R16, GL_RG8, GL_RG16} :

  * GL_RED, GL_RG, GL_RGBA, GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY

  * {GL_R, GL_RG, GL_RGBA} X {8, 16, 16F, 32F, 8UI, 16UI, 32UI, 8I, 16I, 32I}

  * {GL_LUMINANCE, GL_ALPHA, GL_LUMINANCE_ALPHA, GL_INTENSITY} X {8, 16, 16F_ARB, 32F_ARB, 8UI_EXT, 16UI_EXT, 32UI_EXT, 8I_EXT, 16I_EXT, 32I_EXT}


The following image classes are currently disallowed:

  * Textures with borders

  * Multisampled renderbuffers


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cudaGraphicsGLRegisterImage](<../cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g80d12187ae7590807c7676697d9fe03d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuWGLGetDevice ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pDevice, HGPUNVÂ hGpu )


Gets the CUDA device associated with hGpu.

######  Parameters

`pDevice`
    \- Device associated with hGpu
`hGpu`
    \- Handle to a GPU, as queried via WGL_NV_gpu_affinity()

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*pDevice` the CUDA device associated with a `hGpu`, if applicable.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGLMapBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g68f705554e3630cabea768a7621689ee> "Maps an OpenGL buffer object."), [cuGLRegisterBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gaeba53543521eec9ad519bf3fa5574c0> "Registers an OpenGL buffer object."), [cuGLUnmapBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g83c72b84acd61fbdab204000b6daea0d> "Unmaps an OpenGL buffer object."), [cuGLUnregisterBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g5564309322313e2e5df222647227f3a6> "Unregister an OpenGL buffer object."), [cuGLUnmapBufferObjectAsync](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g901b90f8543042cf59f51b99a0c96f3b> "Unmaps an OpenGL buffer object."), [cuGLSetBufferObjectMapFlags](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gb04334f5028a5ad24640949164bd5cc9> "Set the map flags for an OpenGL buffer object."), [cudaWGLGetDevice](<../cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1gcbad4f3a7ed30ee479322f9923a05a2c>)

### OpenGL Interoperability [DEPRECATED]

* * *
