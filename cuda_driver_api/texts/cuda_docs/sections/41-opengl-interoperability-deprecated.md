# OpenGL Interoperability (Deprecated)

## 6.40.1.Â OpenGL Interoperability [DEPRECATED]

## [[OpenGL Interoperability](<group__CUDA__GL.html#group__CUDA__GL>)]

This section describes deprecated OpenGL interoperability functionality.

### Enumerations

enumÂ [CUGLmap_flags](<#group__CUDA__GL__DEPRECATED_1gd0d7e6d08165785f7f801feef8a71c9a>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLCtxCreate](<#group__CUDA__GL__DEPRECATED_1g931f6d260d7db412b37497cb4b2fdf5d>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â Flags, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )
     Create a CUDA context for interoperability with OpenGL.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLInit](<#group__CUDA__GL__DEPRECATED_1g393d6b6cc9bc93185c45bb6c3ec87fe9>) ( void )
     Initializes OpenGL interoperability.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLMapBufferObject](<#group__CUDA__GL__DEPRECATED_1g68f705554e3630cabea768a7621689ee>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â size, GLuintÂ buffer )
     Maps an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLMapBufferObjectAsync](<#group__CUDA__GL__DEPRECATED_1gc309e0c027b9fba9a65ba533ec5f834e>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â size, GLuintÂ buffer, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Maps an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLRegisterBufferObject](<#group__CUDA__GL__DEPRECATED_1gaeba53543521eec9ad519bf3fa5574c0>) ( GLuintÂ buffer )
     Registers an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLSetBufferObjectMapFlags](<#group__CUDA__GL__DEPRECATED_1gb04334f5028a5ad24640949164bd5cc9>) ( GLuintÂ buffer, unsigned int Â Flags )
     Set the map flags for an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLUnmapBufferObject](<#group__CUDA__GL__DEPRECATED_1g83c72b84acd61fbdab204000b6daea0d>) ( GLuintÂ buffer )
     Unmaps an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLUnmapBufferObjectAsync](<#group__CUDA__GL__DEPRECATED_1g901b90f8543042cf59f51b99a0c96f3b>) ( GLuintÂ buffer, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Unmaps an OpenGL buffer object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGLUnregisterBufferObject](<#group__CUDA__GL__DEPRECATED_1g5564309322313e2e5df222647227f3a6>) ( GLuintÂ buffer )
     Unregister an OpenGL buffer object.

### Enumerations

enum CUGLmap_flags


Flags to map or unmap a resource

######  Values

CU_GL_MAP_RESOURCE_FLAGS_NONE = 0x00

CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01

CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLCtxCreate ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â Flags, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )


Create a CUDA context for interoperability with OpenGL.

######  Parameters

`pCtx`
    \- Returned CUDA context
`Flags`
    \- Options for CUDA context creation
`device`
    \- Device on which to create the context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000071>)

This function is deprecated as of Cuda 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with an OpenGL context in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuGLInit](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g393d6b6cc9bc93185c45bb6c3ec87fe9> "Initializes OpenGL interoperability."), [cuGLMapBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g68f705554e3630cabea768a7621689ee> "Maps an OpenGL buffer object."), [cuGLRegisterBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gaeba53543521eec9ad519bf3fa5574c0> "Registers an OpenGL buffer object."), [cuGLUnmapBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g83c72b84acd61fbdab204000b6daea0d> "Unmaps an OpenGL buffer object."), [cuGLUnregisterBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g5564309322313e2e5df222647227f3a6> "Unregister an OpenGL buffer object."), [cuGLMapBufferObjectAsync](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gc309e0c027b9fba9a65ba533ec5f834e> "Maps an OpenGL buffer object."), [cuGLUnmapBufferObjectAsync](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g901b90f8543042cf59f51b99a0c96f3b> "Unmaps an OpenGL buffer object."), [cuGLSetBufferObjectMapFlags](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gb04334f5028a5ad24640949164bd5cc9> "Set the map flags for an OpenGL buffer object."), [cuWGLGetDevice](<group__CUDA__GL.html#group__CUDA__GL_1g21ff8296192dc38dff42ba3346078282> "Gets the CUDA device associated with hGpu.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLInit ( void )


Initializes OpenGL interoperability.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000072>)

This function is deprecated as of Cuda 3.0.

###### Description

Initializes OpenGL interoperability. This function is deprecated and calling it is no longer required. It may fail if the needed OpenGL driver facilities are not available.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGLMapBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g68f705554e3630cabea768a7621689ee> "Maps an OpenGL buffer object."), [cuGLRegisterBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gaeba53543521eec9ad519bf3fa5574c0> "Registers an OpenGL buffer object."), [cuGLUnmapBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g83c72b84acd61fbdab204000b6daea0d> "Unmaps an OpenGL buffer object."), [cuGLUnregisterBufferObject](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g5564309322313e2e5df222647227f3a6> "Unregister an OpenGL buffer object."), [cuGLMapBufferObjectAsync](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gc309e0c027b9fba9a65ba533ec5f834e> "Maps an OpenGL buffer object."), [cuGLUnmapBufferObjectAsync](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1g901b90f8543042cf59f51b99a0c96f3b> "Unmaps an OpenGL buffer object."), [cuGLSetBufferObjectMapFlags](<group__CUDA__GL__DEPRECATED.html#group__CUDA__GL__DEPRECATED_1gb04334f5028a5ad24640949164bd5cc9> "Set the map flags for an OpenGL buffer object."), [cuWGLGetDevice](<group__CUDA__GL.html#group__CUDA__GL_1g21ff8296192dc38dff42ba3346078282> "Gets the CUDA device associated with hGpu.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLMapBufferObject ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â size, GLuintÂ buffer )


Maps an OpenGL buffer object.

######  Parameters

`dptr`
    \- Returned mapped base pointer
`size`
    \- Returned size of mapping
`buffer`
    \- The name of the buffer object to map

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000074>)

This function is deprecated as of Cuda 3.0.

###### Description

Maps the buffer object specified by `buffer` into the address space of the current CUDA context and returns in `*dptr` and `*size` the base pointer and size of the resulting mapping.

There must be a valid OpenGL context bound to the current thread when this function is called. This must be the same context, or a member of the same shareGroup, as the context that was bound when the buffer was registered.

All streams in the current CUDA context are synchronized with the current GL context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLMapBufferObjectAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â size, GLuintÂ buffer, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Maps an OpenGL buffer object.

######  Parameters

`dptr`
    \- Returned mapped base pointer
`size`
    \- Returned size of mapping
`buffer`
    \- The name of the buffer object to map
`hStream`
    \- Stream to synchronize

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000078>)

This function is deprecated as of Cuda 3.0.

###### Description

Maps the buffer object specified by `buffer` into the address space of the current CUDA context and returns in `*dptr` and `*size` the base pointer and size of the resulting mapping.

There must be a valid OpenGL context bound to the current thread when this function is called. This must be the same context, or a member of the same shareGroup, as the context that was bound when the buffer was registered.

Stream `hStream` in the current CUDA context is synchronized with the current GL context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLRegisterBufferObject ( GLuintÂ buffer )


Registers an OpenGL buffer object.

######  Parameters

`buffer`
    \- The name of the buffer object to register.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000073>)

This function is deprecated as of Cuda 3.0.

###### Description

Registers the buffer object specified by `buffer` for access by CUDA. This function must be called before CUDA can map the buffer object. There must be a valid OpenGL context bound to the current thread when this function is called, and the buffer name is resolved by that context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsGLRegisterBuffer](<group__CUDA__GL.html#group__CUDA__GL_1gd530f66cc9ab43a31a98527e75f343a0> "Registers an OpenGL buffer object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLSetBufferObjectMapFlags ( GLuintÂ buffer, unsigned int Â Flags )


Set the map flags for an OpenGL buffer object.

######  Parameters

`buffer`
    \- Buffer object to unmap
`Flags`
    \- Map flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000077>)

This function is deprecated as of Cuda 3.0.

###### Description

Sets the map flags for the buffer object specified by `buffer`.

Changes to `Flags` will take effect the next time `buffer` is mapped. The `Flags` argument may be any of the following:

  * CU_GL_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA kernels. This is the default value.

  * CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA kernels which access this resource will not write to this resource.

  * CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA kernels which access this resource will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


If `buffer` has not been registered for use with CUDA, then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If `buffer` is presently mapped for access by CUDA, then [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>) is returned.

There must be a valid OpenGL context bound to the current thread when this function is called. This must be the same context, or a member of the same shareGroup, as the context that was bound when the buffer was registered.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLUnmapBufferObject ( GLuintÂ buffer )


Unmaps an OpenGL buffer object.

######  Parameters

`buffer`
    \- Buffer object to unmap

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000075>)

This function is deprecated as of Cuda 3.0.

###### Description

Unmaps the buffer object specified by `buffer` for access by CUDA.

There must be a valid OpenGL context bound to the current thread when this function is called. This must be the same context, or a member of the same shareGroup, as the context that was bound when the buffer was registered.

All streams in the current CUDA context are synchronized with the current GL context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLUnmapBufferObjectAsync ( GLuintÂ buffer, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Unmaps an OpenGL buffer object.

######  Parameters

`buffer`
    \- Name of the buffer object to unmap
`hStream`
    \- Stream to synchronize

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000079>)

This function is deprecated as of Cuda 3.0.

###### Description

Unmaps the buffer object specified by `buffer` for access by CUDA.

There must be a valid OpenGL context bound to the current thread when this function is called. This must be the same context, or a member of the same shareGroup, as the context that was bound when the buffer was registered.

Stream `hStream` in the current CUDA context is synchronized with the current GL context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGLUnregisterBufferObject ( GLuintÂ buffer )


Unregister an OpenGL buffer object.

######  Parameters

`buffer`
    \- Name of the buffer object to unregister

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000076>)

This function is deprecated as of Cuda 3.0.

###### Description

Unregisters the buffer object specified by `buffer`. This releases any resources associated with the registered buffer. After this call, the buffer may no longer be mapped for access by CUDA.

There must be a valid OpenGL context bound to the current thread when this function is called. This must be the same context, or a member of the same shareGroup, as the context that was bound when the buffer was registered.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA.")

* * *
