# External Resource Interoperability

## 6.20.Â External Resource Interoperability

This section describes the external resource interoperability functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDestroyExternalMemory](<#group__CUDA__EXTRES__INTEROP_1g1b586dda86565617e7e0883b956c7052>) ( [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)Â extMem )
     Destroys an external memory object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDestroyExternalSemaphore](<#group__CUDA__EXTRES__INTEROP_1g7f13444973542fa50b7e75bcfb2f923d>) ( [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)Â extSem )
     Destroys an external semaphore.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuExternalMemoryGetMappedBuffer](<#group__CUDA__EXTRES__INTEROP_1gb9fec33920400c70961b4e33d838da91>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â devPtr, [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)Â extMem, const [CUDA_EXTERNAL_MEMORY_BUFFER_DESC](<structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1>)*Â bufferDesc )
     Maps a buffer onto an imported memory object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuExternalMemoryGetMappedMipmappedArray](<#group__CUDA__EXTRES__INTEROP_1g02debbfa1b997e4f0e05300a312c17cc>) ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â mipmap, [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)Â extMem, const [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1>)*Â mipmapDesc )
     Maps a CUDA mipmapped array onto an external memory object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuImportExternalMemory](<#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735>) ( [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)*Â extMem_out, const [CUDA_EXTERNAL_MEMORY_HANDLE_DESC](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1>)*Â memHandleDesc )
     Imports an external memory object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuImportExternalSemaphore](<#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326>) ( [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)*Â extSem_out, const [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1>)*Â semHandleDesc )
     Imports an external semaphore.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuSignalExternalSemaphoresAsync](<#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3>) ( const [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)*Â extSemArray, const [CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS](<structCUDA__EXTERNAL__SEMAPHORE__SIGNAL__PARAMS__v1.html#structCUDA__EXTERNAL__SEMAPHORE__SIGNAL__PARAMS__v1>)*Â paramsArray, unsigned int Â numExtSems, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â stream )
     Signals a set of external semaphore objects.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuWaitExternalSemaphoresAsync](<#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0>) ( const [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)*Â extSemArray, const [CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS](<structCUDA__EXTERNAL__SEMAPHORE__WAIT__PARAMS__v1.html#structCUDA__EXTERNAL__SEMAPHORE__WAIT__PARAMS__v1>)*Â paramsArray, unsigned int Â numExtSems, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â stream )
     Waits on a set of external semaphore objects.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDestroyExternalMemory ( [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)Â extMem )


Destroys an external memory object.

######  Parameters

`extMem`
    \- External memory object to be destroyed

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Destroys the specified external memory object. Any existing buffers and CUDA mipmapped arrays mapped onto this object must no longer be used and must be explicitly freed using [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory.") and [cuMipmappedArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1ge0d7c768b6a6963c4d4bde5bbc74f0ad> "Destroys a CUDA mipmapped array.") respectively.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuImportExternalMemory](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735> "Imports an external memory object."), [cuExternalMemoryGetMappedBuffer](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1gb9fec33920400c70961b4e33d838da91> "Maps a buffer onto an imported memory object."), [cuExternalMemoryGetMappedMipmappedArray](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g02debbfa1b997e4f0e05300a312c17cc> "Maps a CUDA mipmapped array onto an external memory object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDestroyExternalSemaphore ( [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)Â extSem )


Destroys an external semaphore.

######  Parameters

`extSem`
    \- External semaphore to be destroyed

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Destroys an external semaphore object and releases any references to the underlying resource. Any outstanding signals or waits must have completed before the semaphore is destroyed.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuExternalMemoryGetMappedBuffer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â devPtr, [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)Â extMem, const [CUDA_EXTERNAL_MEMORY_BUFFER_DESC](<structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1>)*Â bufferDesc )


Maps a buffer onto an imported memory object.

######  Parameters

`devPtr`
    \- Returned device pointer to buffer
`extMem`
    \- Handle to external memory object
`bufferDesc`
    \- Buffer descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Maps a buffer onto an imported memory object and returns a device pointer in `devPtr`.

The properties of the buffer being mapped must be described in `bufferDesc`. The CUDA_EXTERNAL_MEMORY_BUFFER_DESC structure is defined as follows:


    â        typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
                      unsigned long long offset;
                      unsigned long long size;
                      unsigned int flags;
                  } [CUDA_EXTERNAL_MEMORY_BUFFER_DESC](<structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1>);

where [CUDA_EXTERNAL_MEMORY_BUFFER_DESC::offset](<structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1_1734130661fb389658d29d10ed6cf41cd>) is the offset in the memory object where the buffer's base address is. [CUDA_EXTERNAL_MEMORY_BUFFER_DESC::size](<structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1_1b736c75bec5e1461f565e95500c8227f>) is the size of the buffer. [CUDA_EXTERNAL_MEMORY_BUFFER_DESC::flags](<structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__BUFFER__DESC__v1_186aad4fc1ddac67d953daddec0a5c94a>) must be zero.

The offset and size have to be suitably aligned to match the requirements of the external API. Mapping two buffers whose ranges overlap may or may not result in the same virtual address being returned for the overlapped portion. In such cases, the application must ensure that all accesses to that region from the GPU are volatile. Otherwise writes made via one address are not guaranteed to be visible via the other address, even if they're issued by the same thread. It is recommended that applications map the combined range instead of mapping separate buffers and then apply the appropriate offsets to the returned pointer to derive the individual buffers.

The returned pointer `devPtr` must be freed using [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuImportExternalMemory](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735> "Imports an external memory object."), [cuDestroyExternalMemory](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g1b586dda86565617e7e0883b956c7052> "Destroys an external memory object."), [cuExternalMemoryGetMappedMipmappedArray](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g02debbfa1b997e4f0e05300a312c17cc> "Maps a CUDA mipmapped array onto an external memory object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuExternalMemoryGetMappedMipmappedArray ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â mipmap, [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)Â extMem, const [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1>)*Â mipmapDesc )


Maps a CUDA mipmapped array onto an external memory object.

######  Parameters

`mipmap`
    \- Returned CUDA mipmapped array
`extMem`
    \- Handle to external memory object
`mipmapDesc`
    \- CUDA array descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Maps a CUDA mipmapped array onto an external object and returns a handle to it in `mipmap`.

The properties of the CUDA mipmapped array being mapped must be described in `mipmapDesc`. The structure CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC is defined as follows:


    â        typedef struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
                      unsigned long long offset;
                      [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>) arrayDesc;
                      unsigned int numLevels;
                  } [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1>);

where [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::offset](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1_1c4a2c246eab269279434d8153d6e15aa>) is the offset in the memory object where the base level of the mipmap chain is. [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::arrayDesc](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1_17c3e0998380d9922a9fcddadeecd186d>) describes the format, dimensions and type of the base level of the mipmap chain. For further details on these parameters, please refer to the documentation for [cuMipmappedArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1ga5d2e311c7f9b0bc6d130af824a40bd3> "Creates a CUDA mipmapped array."). Note that if the mipmapped array is bound as a color target in the graphics API, then the flag [CUDA_ARRAY3D_COLOR_ATTACHMENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g26e6ae0e2d1dcef8205a840ebc193022>) must be specified in CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::arrayDesc::Flags. [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::numLevels](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1_1938185e3178ee0fb5920ac104035b321>) specifies the total number of levels in the mipmap chain.

If `extMem` was imported from a handle of type [CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8adf93206205d155aed8228f4a118d6ee>), then [CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC::numLevels](<structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__MIPMAPPED__ARRAY__DESC__v1_1938185e3178ee0fb5920ac104035b321>) must be equal to 1.

Mapping `extMem` imported from a handle of type [CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8eaccb8a4e89ddf36b2e432bca0e53791>), is not supported.

The returned CUDA mipmapped array must be freed using [cuMipmappedArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1ge0d7c768b6a6963c4d4bde5bbc74f0ad> "Destroys a CUDA mipmapped array.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuImportExternalMemory](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g52aba3a7f780157d8ba12972b2481735> "Imports an external memory object."), [cuDestroyExternalMemory](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g1b586dda86565617e7e0883b956c7052> "Destroys an external memory object."), [cuExternalMemoryGetMappedBuffer](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1gb9fec33920400c70961b4e33d838da91> "Maps a buffer onto an imported memory object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuImportExternalMemory ( [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>)*Â extMem_out, const [CUDA_EXTERNAL_MEMORY_HANDLE_DESC](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1>)*Â memHandleDesc )


Imports an external memory object.

######  Parameters

`extMem_out`
    \- Returned handle to an external memory object
`memHandleDesc`
    \- Memory import handle descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OPERATING_SYSTEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c849a151611f6e2ed1b3ae923f79ef3c>)

###### Description

Imports an externally allocated memory object and returns a handle to that in `extMem_out`.

The properties of the handle being imported must be described in `memHandleDesc`. The CUDA_EXTERNAL_MEMORY_HANDLE_DESC structure is defined as follows:


    â        typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
                      [CUexternalMemoryHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gbca0bdde9a33c47058b5c97f21e2edd8>) type;
                      union {
                          int fd;
                          struct {
                              void *handle;
                              const void *name;
                          } win32;
                          const void *nvSciBufObject;
                      } handle;
                      unsigned long long size;
                      unsigned int flags;
                  } [CUDA_EXTERNAL_MEMORY_HANDLE_DESC](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1>);

where [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) specifies the type of handle being imported. [CUexternalMemoryHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gbca0bdde9a33c47058b5c97f21e2edd8>) is defined as:


    â        typedef enum CUexternalMemoryHandleType_enum {
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd864efb8402f2268e489336396b2048c07>)          = 1,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8d76fbf1ffdee6cee58aa8a4a41f6b5fd>)       = 2,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8fde7150bec78dae8ee4cd6e88a7f4064>)   = 3,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd81a96f6957350ec0508b6ac4ec17f465f>)         = 4,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8532abdef8908d5d35a773e491ea68f5b>)     = 5,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd89c10fcedb0f4e95a6cbf600f95be2369>)     = 6,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8159b39907f15e7077609b488333cd390>) = 7,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8adf93206205d155aed8228f4a118d6ee>)           = 8,
                      [CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8eaccb8a4e89ddf36b2e432bca0e53791>)          = 9
                  } [CUexternalMemoryHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gbca0bdde9a33c47058b5c97f21e2edd8>);

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd864efb8402f2268e489336396b2048c07>), then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::fd must be a valid file descriptor referencing a memory object. Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported successfully. Performing any operations on the file descriptor after it is imported results in undefined behavior.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8d76fbf1ffdee6cee58aa8a4a41f6b5fd>), then exactly one of CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be NULL. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle is not NULL, then it must represent a valid shared NT handle that references a memory object. Ownership of this handle is not transferred to CUDA after the import operation, so the application must release the handle using the appropriate system call. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name is not NULL, then it must point to a NULL-terminated array of UTF-16 characters that refers to a memory object.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8fde7150bec78dae8ee4cd6e88a7f4064>), then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must be non-NULL and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must be NULL. The handle specified must be a globally shared KMT handle. This handle does not hold a reference to the underlying object, and thus will be invalid when all references to the memory object are destroyed.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd81a96f6957350ec0508b6ac4ec17f465f>), then exactly one of CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be NULL. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle is not NULL, then it must represent a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle when referring to a ID3D12Heap object. This handle holds a reference to the underlying object. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name is not NULL, then it must point to a NULL-terminated array of UTF-16 characters that refers to a ID3D12Heap object.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8532abdef8908d5d35a773e491ea68f5b>), then exactly one of CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be NULL. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle is not NULL, then it must represent a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle when referring to a ID3D12Resource object. This handle holds a reference to the underlying object. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name is not NULL, then it must point to a NULL-terminated array of UTF-16 characters that refers to a ID3D12Resource object.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd89c10fcedb0f4e95a6cbf600f95be2369>), then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must represent a valid shared NT handle that is returned by IDXGIResource1::CreateSharedHandle when referring to a ID3D11Resource object. If CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name is not NULL, then it must point to a NULL-terminated array of UTF-16 characters that refers to a ID3D11Resource object.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8159b39907f15e7077609b488333cd390>), then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must represent a valid shared KMT handle that is returned by IDXGIResource::GetSharedHandle when referring to a ID3D11Resource object and CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must be NULL.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8adf93206205d155aed8228f4a118d6ee>), then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::nvSciBufObject must be non-NULL and reference a valid NvSciBuf object. If the NvSciBuf object imported into CUDA is also mapped by other drivers, then the application must use [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.") or [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects.") as appropriate barriers to maintain coherence between CUDA and the other drivers. See [CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1a6161a80f60177235f479cd74de7e04>) and [CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf07a1d15f2696b915c068c892e6f1a35>) for memory synchronization.

If [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is [CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8eaccb8a4e89ddf36b2e432bca0e53791>), then CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::fd must be a valid file descriptor referencing a dma_buf object and [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::flags](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_144d7c253f6b6f34e46b8726496e425c9>) must be zero. Importing a dma_buf object is supported only on Tegra Jetson platform starting with Thor series. Mapping an imported dma_buf object as CUDA mipmapped array using [cuExternalMemoryGetMappedMipmappedArray](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g02debbfa1b997e4f0e05300a312c17cc> "Maps a CUDA mipmapped array onto an external memory object.") is not supported.

The size of the memory object must be specified in [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::size](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_136570cee9bbfa12b3f34d0d1d98029ce>).

Specifying the flag [CUDA_EXTERNAL_MEMORY_DEDICATED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7a3c833643e392f32a52c131aa87ccac>) in [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::flags](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_144d7c253f6b6f34e46b8726496e425c9>) indicates that the resource is a dedicated resource. The definition of what a dedicated resource is outside the scope of this extension. This flag must be set if [CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type](<structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__MEMORY__HANDLE__DESC__v1_1d4e3663348d28278d066980b422ab70e>) is one of the following: [CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8532abdef8908d5d35a773e491ea68f5b>)[CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd89c10fcedb0f4e95a6cbf600f95be2369>)[CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8159b39907f15e7077609b488333cd390>)

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * If the Vulkan memory imported into CUDA is mapped on the CPU then the application must use vkInvalidateMappedMemoryRanges/vkFlushMappedMemoryRanges as well as appropriate Vulkan pipeline barriers to maintain coherence between CPU and GPU. For more information on these APIs, please refer to "Synchronization and Cache Control" chapter from Vulkan specification.


**See also:**

[cuDestroyExternalMemory](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g1b586dda86565617e7e0883b956c7052> "Destroys an external memory object."), [cuExternalMemoryGetMappedBuffer](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1gb9fec33920400c70961b4e33d838da91> "Maps a buffer onto an imported memory object."), [cuExternalMemoryGetMappedMipmappedArray](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g02debbfa1b997e4f0e05300a312c17cc> "Maps a CUDA mipmapped array onto an external memory object.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuImportExternalSemaphore ( [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)*Â extSem_out, const [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1>)*Â semHandleDesc )


Imports an external semaphore.

######  Parameters

`extSem_out`
    \- Returned handle to an external semaphore
`semHandleDesc`
    \- Semaphore import handle descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OPERATING_SYSTEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c849a151611f6e2ed1b3ae923f79ef3c>)

###### Description

Imports an externally allocated synchronization object and returns a handle to that in `extSem_out`.

The properties of the handle being imported must be described in `semHandleDesc`. The CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC is defined as follows:


    â        typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
                      [CUexternalSemaphoreHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfecb27c44dbd154273d24d35896a2920>) type;
                      union {
                          int fd;
                          struct {
                              void *handle;
                              const void *name;
                          } win32;
                          const void* NvSciSyncObj;
                      } handle;
                      unsigned int flags;
                  } [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1>);

where [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) specifies the type of handle being imported. [CUexternalSemaphoreHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfecb27c44dbd154273d24d35896a2920>) is defined as:


    â        typedef enum CUexternalSemaphoreHandleType_enum {
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920771961c5b13e123cbc9b44ef0886067d>)                = 1,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920e56d4d1a5f92a866d6d8da5249a7f068>)             = 2,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292014138c9067a288abae77ee3b7efacf4b>)         = 3,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a29200f65d633a7b72b5cd5d2bb741c0edfd7>)              = 4,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292068cb46b3e49fa0dc11c89376e93a833b>)              = 5,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>)                = 6,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920a93c602f3eb2b39ab95b0dfeb72f8eba>)        = 7,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920c97816c259dde37432b63177377473da>)    = 8,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d133ab71b234267c59da61978951e020>)    = 9,
                      [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d3625359c063deedf0cef25b16fbba4b>) = 10
                  } [CUexternalSemaphoreHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfecb27c44dbd154273d24d35896a2920>);

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920771961c5b13e123cbc9b44ef0886067d>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid file descriptor referencing a synchronization object. Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported successfully. Performing any operations on the file descriptor after it is imported results in undefined behavior.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920e56d4d1a5f92a866d6d8da5249a7f068>), then exactly one of CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be NULL. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle is not NULL, then it must represent a valid shared NT handle that references a synchronization object. Ownership of this handle is not transferred to CUDA after the import operation, so the application must release the handle using the appropriate system call. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it must name a valid synchronization object.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292014138c9067a288abae77ee3b7efacf4b>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle must be non-NULL and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must be NULL. The handle specified must be a globally shared KMT handle. This handle does not hold a reference to the underlying object, and thus will be invalid when all references to the synchronization object are destroyed.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a29200f65d633a7b72b5cd5d2bb741c0edfd7>), then exactly one of CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be NULL. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle is not NULL, then it must represent a valid shared NT handle that is returned by ID3D12Device::CreateSharedHandle when referring to a ID3D12Fence object. This handle holds a reference to the underlying object. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it must name a valid synchronization object that refers to a valid ID3D12Fence object.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292068cb46b3e49fa0dc11c89376e93a833b>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle represents a valid shared NT handle that is returned by ID3D11Fence::CreateSharedHandle. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it must name a valid synchronization object that refers to a valid ID3D11Fence object.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::nvSciSyncObj represents a valid NvSciSyncObj.

[CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920a93c602f3eb2b39ab95b0dfeb72f8eba>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle represents a valid shared NT handle that is returned by IDXGIResource1::CreateSharedHandle when referring to a IDXGIKeyedMutex object. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it must name a valid synchronization object that refers to a valid IDXGIKeyedMutex object.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920c97816c259dde37432b63177377473da>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle represents a valid shared KMT handle that is returned by IDXGIResource::GetSharedHandle when referring to a IDXGIKeyedMutex object and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must be NULL.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d133ab71b234267c59da61978951e020>), then CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid file descriptor referencing a synchronization object. Ownership of the file descriptor is transferred to the CUDA driver when the handle is imported successfully. Performing any operations on the file descriptor after it is imported results in undefined behavior.

If [CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type](<structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1.html#structCUDA__EXTERNAL__SEMAPHORE__HANDLE__DESC__v1_1cf32b96f55beec6f904b2455effed87d>) is [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d3625359c063deedf0cef25b16fbba4b>), then exactly one of CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be NULL. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle is not NULL, then it must represent a valid shared NT handle that references a synchronization object. Ownership of this handle is not transferred to CUDA after the import operation, so the application must release the handle using the appropriate system call. If CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name is not NULL, then it must name a valid synchronization object.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDestroyExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g7f13444973542fa50b7e75bcfb2f923d> "Destroys an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuSignalExternalSemaphoresAsync ( const [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)*Â extSemArray, const [CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS](<structCUDA__EXTERNAL__SEMAPHORE__SIGNAL__PARAMS__v1.html#structCUDA__EXTERNAL__SEMAPHORE__SIGNAL__PARAMS__v1>)*Â paramsArray, unsigned int Â numExtSems, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â stream )


Signals a set of external semaphore objects.

######  Parameters

`extSemArray`
    \- Set of external semaphores to be signaled
`paramsArray`
    \- Array of semaphore parameters
`numExtSems`
    \- Number of semaphores to signal
`stream`
    \- Stream to enqueue the signal operations in

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Enqueues a signal operation on a set of externally allocated semaphore object in the specified stream. The operations will be executed when all prior operations in the stream complete.

The exact semantics of signaling a semaphore depends on the type of the object.

If the semaphore object is any one of the following types: [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920771961c5b13e123cbc9b44ef0886067d>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920e56d4d1a5f92a866d6d8da5249a7f068>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292014138c9067a288abae77ee3b7efacf4b>) then signaling the semaphore will set it to the signaled state.

If the semaphore object is any one of the following types: [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a29200f65d633a7b72b5cd5d2bb741c0edfd7>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292068cb46b3e49fa0dc11c89376e93a833b>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d133ab71b234267c59da61978951e020>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d3625359c063deedf0cef25b16fbba4b>) then the semaphore will be set to the value specified in CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::fence::value.

If the semaphore object is of the type [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>) this API sets CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence to a value that can be used by subsequent waiters of the same NvSciSync object to order operations with those currently submitted in `stream`. Such an update will overwrite previous contents of CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence. By default, signaling such an external semaphore object causes appropriate memory synchronization operations to be performed over all external memory objects that are imported as [CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8adf93206205d155aed8228f4a118d6ee>). This ensures that any subsequent accesses made by other importers of the same set of NvSciBuf memory object(s) are coherent. These operations can be skipped by specifying the flag [CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1a6161a80f60177235f479cd74de7e04>), which can be used as a performance optimization when data coherency is not required. But specifying this flag in scenarios where data coherency is required results in undefined behavior. Also, for semaphore object of the type [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>), if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in [cuDeviceGetNvSciSyncAttributes](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g0991e2b2b3cedee1ca77d6376e581335> "Return NvSciSync attributes that this device can support.") to CUDA_NVSCISYNC_ATTR_SIGNAL, this API will return CUDA_ERROR_NOT_SUPPORTED. NvSciSyncFence associated with semaphore object of the type [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>) can be deterministic. For this the NvSciSyncAttrList used to create the semaphore object must have value of NvSciSyncAttrKey_RequireDeterministicFences key set to true. Deterministic fences allow users to enqueue a wait over the semaphore object even before corresponding signal is enqueued. For such a semaphore object, CUDA guarantees that each signal operation will increment the fence value by '1'. Users are expected to track count of signals enqueued on the semaphore object and insert waits accordingly. When such a semaphore object is signaled from multiple streams, due to concurrent stream execution, it is possible that the order in which the semaphore gets signaled is indeterministic. This could lead to waiters of the semaphore getting unblocked incorrectly. Users are expected to handle such situations, either by not using the same semaphore object with deterministic fence support enabled in different streams or by adding explicit dependency amongst such streams so that the semaphore is signaled in order. NvSciSyncFence associated with semaphore object of the type [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>) can be timestamp enabled. For this the NvSciSyncAttrList used to create the object must have the value of NvSciSyncAttrKey_WaiterRequireTimestamps key set to true. Timestamps are emitted asynchronously by the GPU and CUDA saves the GPU timestamp in the corresponding NvSciSyncFence at the time of signal on GPU. Users are expected to convert GPU clocks to CPU clocks using appropriate scaling functions. Users are expected to wait for the completion of the fence before extracting timestamp using appropriate NvSciSync APIs. Users are expected to ensure that there is only one outstanding timestamp enabled fence per Cuda-NvSciSync object at any point of time, failing which leads to undefined behavior. Extracting the timestamp before the corresponding fence is signalled could lead to undefined behaviour. Timestamp extracted via appropriate NvSciSync API would be in microseconds.

If the semaphore object is any one of the following types: [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920a93c602f3eb2b39ab95b0dfeb72f8eba>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920c97816c259dde37432b63177377473da>) then the keyed mutex will be released with the key specified in CUDA_EXTERNAL_SEMAPHORE_PARAMS::params::keyedmutex::key.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuDestroyExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g7f13444973542fa50b7e75bcfb2f923d> "Destroys an external semaphore."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuWaitExternalSemaphoresAsync ( const [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>)*Â extSemArray, const [CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS](<structCUDA__EXTERNAL__SEMAPHORE__WAIT__PARAMS__v1.html#structCUDA__EXTERNAL__SEMAPHORE__WAIT__PARAMS__v1>)*Â paramsArray, unsigned int Â numExtSems, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â stream )


Waits on a set of external semaphore objects.

######  Parameters

`extSemArray`
    \- External semaphores to be waited on
`paramsArray`
    \- Array of semaphore parameters
`numExtSems`
    \- Number of semaphores to wait on
`stream`
    \- Stream to enqueue the wait operations in

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e92cd1f40fc3a327d2bc6e9ff650d1af11>)

###### Description

Enqueues a wait operation on a set of externally allocated semaphore object in the specified stream. The operations will be executed when all prior operations in the stream complete.

The exact semantics of waiting on a semaphore depends on the type of the object.

If the semaphore object is any one of the following types: [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920771961c5b13e123cbc9b44ef0886067d>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920e56d4d1a5f92a866d6d8da5249a7f068>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292014138c9067a288abae77ee3b7efacf4b>) then waiting on the semaphore will wait until the semaphore reaches the signaled state. The semaphore will then be reset to the unsignaled state. Therefore for every signal operation, there can only be one wait operation.

If the semaphore object is any one of the following types: [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a29200f65d633a7b72b5cd5d2bb741c0edfd7>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a292068cb46b3e49fa0dc11c89376e93a833b>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d133ab71b234267c59da61978951e020>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920d3625359c063deedf0cef25b16fbba4b>) then waiting on the semaphore will wait until the value of the semaphore is greater than or equal to CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::fence::value.

If the semaphore object is of the type [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>) then, waiting on the semaphore will wait until the CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::nvSciSync::fence is signaled by the signaler of the NvSciSyncObj that was associated with this semaphore object. By default, waiting on such an external semaphore object causes appropriate memory synchronization operations to be performed over all external memory objects that are imported as [CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggbca0bdde9a33c47058b5c97f21e2edd8adf93206205d155aed8228f4a118d6ee>). This ensures that any subsequent accesses made by other importers of the same set of NvSciBuf memory object(s) are coherent. These operations can be skipped by specifying the flag [CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf07a1d15f2696b915c068c892e6f1a35>), which can be used as a performance optimization when data coherency is not required. But specifying this flag in scenarios where data coherency is required results in undefined behavior. Also, for semaphore object of the type [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920908455daa7bbeb91a83930f977c0f1c1>), if the NvSciSyncAttrList used to create the NvSciSyncObj had not set the flags in [cuDeviceGetNvSciSyncAttributes](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g0991e2b2b3cedee1ca77d6376e581335> "Return NvSciSync attributes that this device can support.") to CUDA_NVSCISYNC_ATTR_WAIT, this API will return CUDA_ERROR_NOT_SUPPORTED.

If the semaphore object is any one of the following types: [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920a93c602f3eb2b39ab95b0dfeb72f8eba>), [CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggfecb27c44dbd154273d24d35896a2920c97816c259dde37432b63177377473da>) then the keyed mutex will be acquired when it is released with the key specified in CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::keyedmutex::key or until the timeout specified by CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::keyedmutex::timeoutMs has lapsed. The timeout interval can either be a finite value specified in milliseconds or an infinite value. In case an infinite value is specified the timeout never elapses. The windows INFINITE macro must be used to specify infinite timeout.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuDestroyExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g7f13444973542fa50b7e75bcfb2f923d> "Destroys an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects.")

* * *
