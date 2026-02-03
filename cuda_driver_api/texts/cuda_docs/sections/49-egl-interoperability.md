# EGL Interoperability

## 6.45.Â EGL Interoperability

This section describes the EGL interoperability functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamConsumerAcquireFrame](<#group__CUDA__EGL_1g10507a0acb74a90136caacb363a3c6a7>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream, unsigned int Â timeout )
     Acquire an image frame from the EGLStream with CUDA as a consumer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamConsumerConnect](<#group__CUDA__EGL_1g3f59b85a292d59c19c8b64b8ade8a658>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, EGLStreamKHRÂ stream )
     Connect CUDA to EGLStream as a consumer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamConsumerConnectWithFlags](<#group__CUDA__EGL_1g7be3b064ea600a7bac4906e5d61ba4b7>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, EGLStreamKHRÂ stream, unsigned int Â flags )
     Connect CUDA to EGLStream as a consumer with given flags.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamConsumerDisconnect](<#group__CUDA__EGL_1g3ab15cff9be3b25447714101ecda6a61>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn )
     Disconnect CUDA as a consumer to EGLStream .
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamConsumerReleaseFrame](<#group__CUDA__EGL_1g4dadfefc718210e91c8f44f6a8e4b233>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â pCudaResource, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream )
     Releases the last frame acquired from the EGLStream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamProducerConnect](<#group__CUDA__EGL_1g5d181803d994a06f1bf9b05f52757bef>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, EGLStreamKHRÂ stream, EGLintÂ width, EGLintÂ height )
     Connect CUDA to EGLStream as a producer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamProducerDisconnect](<#group__CUDA__EGL_1gbdc9664bfb17dd3fa1e0a3ca68a8cafd>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn )
     Disconnect CUDA as a producer to EGLStream .
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamProducerPresentFrame](<#group__CUDA__EGL_1g60dcaadeabcbaedb4a271d529306687b>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>)Â eglframe, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream )
     Present a CUDA eglFrame to the EGLStream with CUDA as a producer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEGLStreamProducerReturnFrame](<#group__CUDA__EGL_1g70c84d9d01f343fc07cd632f9cfc3a06>) ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>)*Â eglframe, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream )
     Return the CUDA eglFrame to the EGLStream released by the consumer.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventCreateFromEGLSync](<#group__CUDA__EGL_1gc1f625de07ffc410973fcc9709e36342>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â phEvent, EGLSyncKHRÂ eglSync, unsigned int Â flags )
     Creates an event from EGLSync object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsEGLRegisterImage](<#group__CUDA__EGL_1g9f9b026d175238be6f6e79048d6879c5>) ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, EGLImageKHRÂ image, unsigned int Â flags )
     Registers an EGL image.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphicsResourceGetMappedEglFrame](<#group__CUDA__EGL_1ge1e57193ad1dbf554af60d5b2d096ede>) ( [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>)*Â eglFrame, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource, unsigned int Â index, unsigned int Â mipLevel )
     Get an eglFrame through which to access a registered EGL graphics resource.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamConsumerAcquireFrame ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream, unsigned int Â timeout )


Acquire an image frame from the EGLStream with CUDA as a consumer.

######  Parameters

`conn`
    \- Connection on which to acquire
`pCudaResource`
    \- CUDA resource on which the stream frame will be mapped for use.
`pStream`
    \- CUDA stream for synchronization and any data migrations implied by [CUeglResourceLocationFlags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf2804cd7cda3b8716c31ba620f644cd3>).
`timeout`
    \- Desired timeout in usec for a new frame to be acquired. If set as [CUDA_EGL_INFINITE_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g71b9a58998751468a873848efd699af3>), acquire waits infinitely. After timeout occurs CUDA consumer tries to acquire an old frame if available and EGL_SUPPORT_REUSE_NV flag is set.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>),

###### Description

Acquire an image frame from EGLStreamKHR. This API can also acquire an old frame presented by the producer unless explicitly disabled by setting EGL_SUPPORT_REUSE_NV flag to EGL_FALSE during stream initialization. By default, EGLStream is created with this flag set to EGL_TRUE. [cuGraphicsResourceGetMappedEglFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1ge1e57193ad1dbf554af60d5b2d096ede> "Get an eglFrame through which to access a registered EGL graphics resource.") can be called on `pCudaResource` to get CUeglFrame.

**See also:**

[cuEGLStreamConsumerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3f59b85a292d59c19c8b64b8ade8a658> "Connect CUDA to EGLStream as a consumer."), [cuEGLStreamConsumerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3ab15cff9be3b25447714101ecda6a61> "Disconnect CUDA as a consumer to EGLStream ."), [cuEGLStreamConsumerAcquireFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g10507a0acb74a90136caacb363a3c6a7> "Acquire an image frame from the EGLStream with CUDA as a consumer."), [cuEGLStreamConsumerReleaseFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g4dadfefc718210e91c8f44f6a8e4b233> "Releases the last frame acquired from the EGLStream."), [cudaEGLStreamConsumerAcquireFrame](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g83dd1bfea48c093d3f0b247754970f58>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamConsumerConnect ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, EGLStreamKHRÂ stream )


Connect CUDA to EGLStream as a consumer.

######  Parameters

`conn`
    \- Pointer to the returned connection handle
`stream`
    \- EGLStreamKHR handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Connect CUDA as a consumer to EGLStreamKHR specified by `stream`.

The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one API to another.

**See also:**

[cuEGLStreamConsumerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3f59b85a292d59c19c8b64b8ade8a658> "Connect CUDA to EGLStream as a consumer."), [cuEGLStreamConsumerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3ab15cff9be3b25447714101ecda6a61> "Disconnect CUDA as a consumer to EGLStream ."), [cuEGLStreamConsumerAcquireFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g10507a0acb74a90136caacb363a3c6a7> "Acquire an image frame from the EGLStream with CUDA as a consumer."), [cuEGLStreamConsumerReleaseFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g4dadfefc718210e91c8f44f6a8e4b233> "Releases the last frame acquired from the EGLStream."), [cudaEGLStreamConsumerConnect](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g7993b0e3802420547e3f403549be65a1>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamConsumerConnectWithFlags ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, EGLStreamKHRÂ stream, unsigned int Â flags )


Connect CUDA to EGLStream as a consumer with given flags.

######  Parameters

`conn`
    \- Pointer to the returned connection handle
`stream`
    \- EGLStreamKHR handle
`flags`
    \- Flags denote intended location - system or video.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Connect CUDA as a consumer to EGLStreamKHR specified by `stream` with specified `flags` defined by CUeglResourceLocationFlags.

The flags specify whether the consumer wants to access frames from system memory or video memory. Default is [CU_EGL_RESOURCE_LOCATION_VIDMEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggf2804cd7cda3b8716c31ba620f644cd3115dedc3b0a393b00d2c38d996daeedc>).

**See also:**

[cuEGLStreamConsumerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3f59b85a292d59c19c8b64b8ade8a658> "Connect CUDA to EGLStream as a consumer."), [cuEGLStreamConsumerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3ab15cff9be3b25447714101ecda6a61> "Disconnect CUDA as a consumer to EGLStream ."), [cuEGLStreamConsumerAcquireFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g10507a0acb74a90136caacb363a3c6a7> "Acquire an image frame from the EGLStream with CUDA as a consumer."), [cuEGLStreamConsumerReleaseFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g4dadfefc718210e91c8f44f6a8e4b233> "Releases the last frame acquired from the EGLStream."), [cudaEGLStreamConsumerConnectWithFlags](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g4e2d79eb6bcb9eca4f6e3f13eb3f7fc3>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamConsumerDisconnect ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn )


Disconnect CUDA as a consumer to EGLStream .

######  Parameters

`conn`
    \- Conection to disconnect.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Disconnect CUDA as a consumer to EGLStreamKHR.

**See also:**

[cuEGLStreamConsumerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3f59b85a292d59c19c8b64b8ade8a658> "Connect CUDA to EGLStream as a consumer."), [cuEGLStreamConsumerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3ab15cff9be3b25447714101ecda6a61> "Disconnect CUDA as a consumer to EGLStream ."), [cuEGLStreamConsumerAcquireFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g10507a0acb74a90136caacb363a3c6a7> "Acquire an image frame from the EGLStream with CUDA as a consumer."), [cuEGLStreamConsumerReleaseFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g4dadfefc718210e91c8f44f6a8e4b233> "Releases the last frame acquired from the EGLStream."), [cudaEGLStreamConsumerDisconnect](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1gb2ef252e72ad2419506f3cf305753c6a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamConsumerReleaseFrame ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â pCudaResource, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream )


Releases the last frame acquired from the EGLStream.

######  Parameters

`conn`
    \- Connection on which to release
`pCudaResource`
    \- CUDA resource whose corresponding frame is to be released
`pStream`
    \- CUDA stream on which release will be done.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>),

###### Description

Release the acquired image frame specified by `pCudaResource` to EGLStreamKHR. If EGL_SUPPORT_REUSE_NV flag is set to EGL_TRUE, at the time of EGL creation this API doesn't release the last frame acquired on the EGLStream. By default, EGLStream is created with this flag set to EGL_TRUE.

**See also:**

[cuEGLStreamConsumerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3f59b85a292d59c19c8b64b8ade8a658> "Connect CUDA to EGLStream as a consumer."), [cuEGLStreamConsumerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g3ab15cff9be3b25447714101ecda6a61> "Disconnect CUDA as a consumer to EGLStream ."), [cuEGLStreamConsumerAcquireFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g10507a0acb74a90136caacb363a3c6a7> "Acquire an image frame from the EGLStream with CUDA as a consumer."), [cuEGLStreamConsumerReleaseFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g4dadfefc718210e91c8f44f6a8e4b233> "Releases the last frame acquired from the EGLStream."), [cudaEGLStreamConsumerReleaseFrame](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g51b3df89a3e0eb8baad7449674797467>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamProducerConnect ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, EGLStreamKHRÂ stream, EGLintÂ width, EGLintÂ height )


Connect CUDA to EGLStream as a producer.

######  Parameters

`conn`
    \- Pointer to the returned connection handle
`stream`
    \- EGLStreamKHR handle
`width`
    \- width of the image to be submitted to the stream
`height`
    \- height of the image to be submitted to the stream

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Connect CUDA as a producer to EGLStreamKHR specified by `stream`.

The EGLStreamKHR is an EGL object that transfers a sequence of image frames from one API to another.

**See also:**

[cuEGLStreamProducerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g5d181803d994a06f1bf9b05f52757bef> "Connect CUDA to EGLStream as a producer."), [cuEGLStreamProducerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1gbdc9664bfb17dd3fa1e0a3ca68a8cafd> "Disconnect CUDA as a producer to EGLStream ."), [cuEGLStreamProducerPresentFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g60dcaadeabcbaedb4a271d529306687b> "Present a CUDA eglFrame to the EGLStream with CUDA as a producer."), [cudaEGLStreamProducerConnect](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1gf35966d50689874614985f688a888c03>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamProducerDisconnect ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn )


Disconnect CUDA as a producer to EGLStream .

######  Parameters

`conn`
    \- Conection to disconnect.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Disconnect CUDA as a producer to EGLStreamKHR.

**See also:**

[cuEGLStreamProducerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g5d181803d994a06f1bf9b05f52757bef> "Connect CUDA to EGLStream as a producer."), [cuEGLStreamProducerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1gbdc9664bfb17dd3fa1e0a3ca68a8cafd> "Disconnect CUDA as a producer to EGLStream ."), [cuEGLStreamProducerPresentFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g60dcaadeabcbaedb4a271d529306687b> "Present a CUDA eglFrame to the EGLStream with CUDA as a producer."), [cudaEGLStreamProducerDisconnect](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g381335525d81342c29c0b62cc4f64dc9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamProducerPresentFrame ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>)Â eglframe, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream )


Present a CUDA eglFrame to the EGLStream with CUDA as a producer.

######  Parameters

`conn`
    \- Connection on which to present the CUDA array
`eglframe`
    \- CUDA Eglstream Proucer Frame handle to be sent to the consumer over EglStream.
`pStream`
    \- CUDA stream on which to present the frame.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>),

###### Description

When a frame is presented by the producer, it gets associated with the EGLStream and thus it is illegal to free the frame before the producer is disconnected. If a frame is freed and reused it may lead to undefined behavior.

If producer and consumer are on different GPUs (iGPU and dGPU) then frametype [CU_EGL_FRAME_TYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggec3f4a4e1a5785b1aa0fcc209cd47c38c02019e2bb4a56d31db30925d567d101>) is not supported. [CU_EGL_FRAME_TYPE_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggec3f4a4e1a5785b1aa0fcc209cd47c38fc6cb007c686d8cad86705005c55bf33>) can be used for such cross-device applications.

The CUeglFrame is defined as:


    â typedef struct CUeglFrame_st {
               union {
                   [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) pArray[[MAX_PLANES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4269718bae6e29c6059d666ec76df24b>)];
                   void*   pPitch[[MAX_PLANES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4269718bae6e29c6059d666ec76df24b>)];
               } frame;
               unsigned int width;
               unsigned int height;
               unsigned int depth;
               unsigned int pitch;
               unsigned int planeCount;
               unsigned int numChannels;
               [CUeglFrameType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec3f4a4e1a5785b1aa0fcc209cd47c38>) frameType;
               [CUeglColorFormat](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g38337000e43e400e77ad36c7e197a9f2>) eglColorFormat;
               [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) cuFormat;
           } [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>);

For CUeglFrame of type [CU_EGL_FRAME_TYPE_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggec3f4a4e1a5785b1aa0fcc209cd47c38fc6cb007c686d8cad86705005c55bf33>), the application may present sub-region of a memory allocation. In that case, the pitched pointer will specify the start address of the sub-region in the allocation and corresponding CUeglFrame fields will specify the dimensions of the sub-region.

**See also:**

[cuEGLStreamProducerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g5d181803d994a06f1bf9b05f52757bef> "Connect CUDA to EGLStream as a producer."), [cuEGLStreamProducerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1gbdc9664bfb17dd3fa1e0a3ca68a8cafd> "Disconnect CUDA as a producer to EGLStream ."), [cuEGLStreamProducerReturnFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g70c84d9d01f343fc07cd632f9cfc3a06> "Return the CUDA eglFrame to the EGLStream released by the consumer."), [cudaEGLStreamProducerPresentFrame](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g5c84a3778586dda401df00052ae5753b>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEGLStreamProducerReturnFrame ( [CUeglStreamConnection](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g60c595264697050bc2bb8e00cf5f86e7>)*Â conn, [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>)*Â eglframe, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â pStream )


Return the CUDA eglFrame to the EGLStream released by the consumer.

######  Parameters

`conn`
    \- Connection on which to return
`eglframe`
    \- CUDA Eglstream Proucer Frame handle returned from the consumer over EglStream.
`pStream`
    \- CUDA stream on which to return the frame.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_LAUNCH_TIMEOUT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e965460d83f63575af9805ca59f8f19d74>)

###### Description

This API can potentially return CUDA_ERROR_LAUNCH_TIMEOUT if the consumer has not returned a frame to EGL stream. If timeout is returned the application can retry.

**See also:**

[cuEGLStreamProducerConnect](<group__CUDA__EGL.html#group__CUDA__EGL_1g5d181803d994a06f1bf9b05f52757bef> "Connect CUDA to EGLStream as a producer."), [cuEGLStreamProducerDisconnect](<group__CUDA__EGL.html#group__CUDA__EGL_1gbdc9664bfb17dd3fa1e0a3ca68a8cafd> "Disconnect CUDA as a producer to EGLStream ."), [cuEGLStreamProducerPresentFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1g60dcaadeabcbaedb4a271d529306687b> "Present a CUDA eglFrame to the EGLStream with CUDA as a producer."), [cudaEGLStreamProducerReturnFrame](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g631d1080365d32a35a19b87584725748>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventCreateFromEGLSync ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â phEvent, EGLSyncKHRÂ eglSync, unsigned int Â flags )


Creates an event from EGLSync object.

######  Parameters

`phEvent`
    \- Returns newly created event
`eglSync`
    \- Opaque handle to EGLSync object
`flags`
    \- Event creation flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates an event *phEvent from an EGLSyncKHR eglSync with the flags specified via `flags`. Valid flags include:

  * [CU_EVENT_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629e22adf5df73b0d43c6374a12ebee1333>): Default event creation flag.

  * [CU_EVENT_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f6296813b3b31fdb737133124f3c35044362>): Specifies that the created event should use blocking synchronization. A CPU thread that uses [cuEventSynchronize()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete.") to wait on an event created with this flag will block until the event has actually been completed.


Once the `eglSync` gets destroyed, [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event.") is the only API that can be invoked on the event.

[cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") and TimingData are not supported for events created from EGLSync.

The EGLSyncKHR is an opaque handle to an EGL sync object. typedef void* EGLSyncKHR

**See also:**

[cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsEGLRegisterImage ( [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)*Â pCudaResource, EGLImageKHRÂ image, unsigned int Â flags )


Registers an EGL image.

######  Parameters

`pCudaResource`
    \- Pointer to the returned object handle
`image`
    \- An EGLImageKHR image which can be used to create target resource.
`flags`
    \- Map flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ALREADY_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9240bb253a699176d9f49ee2f2c91b61b>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Registers the EGLImageKHR specified by `image` for access by CUDA. A handle to the registered object is returned as `pCudaResource`. Additional Mapping/Unmapping is not required for the registered resource and [cuGraphicsResourceGetMappedEglFrame](<group__CUDA__EGL.html#group__CUDA__EGL_1ge1e57193ad1dbf554af60d5b2d096ede> "Get an eglFrame through which to access a registered EGL graphics resource.") can be directly called on the `pCudaResource`.

The application will be responsible for synchronizing access to shared objects. The application must ensure that any pending operation which access the objects have completed before passing control to CUDA. This may be accomplished by issuing and waiting for glFinish command on all GLcontexts (for OpenGL and likewise for other APIs). The application will be also responsible for ensuring that any pending operation on the registered CUDA resource has completed prior to executing subsequent commands in other APIs accesing the same memory objects. This can be accomplished by calling cuCtxSynchronize or cuEventSynchronize (preferably).

The surface's intended usage is specified using `flags`, as follows:

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource will be used. It is therefore assumed that this resource will be read from and written to by CUDA. This is the default value.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY: Specifies that CUDA will not write to this resource.

  * CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD: Specifies that CUDA will not read from this resource and will write over the entire contents of the resource, so none of the data previously stored in the resource will be preserved.


The EGLImageKHR is an object which can be used to create EGLImage target resource. It is defined as a void pointer. typedef void* EGLImageKHR

**See also:**

[cuGraphicsEGLRegisterImage](<group__CUDA__EGL.html#group__CUDA__EGL_1g9f9b026d175238be6f6e79048d6879c5> "Registers an EGL image."), [cuGraphicsUnregisterResource](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1ga7e5e97b74eaa13dfa6582e853e4c96d> "Unregisters a graphics resource for access by CUDA."), [cuGraphicsResourceSetMapFlags](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gfe96aa7747f8b11d44a6fa6a851e1b39> "Set usage flags for mapping a graphics resource."), [cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsUnmapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8e9ff25d071375a0df1cb5aee924af32> "Unmap graphics resources."), [cudaGraphicsEGLRegisterImage](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1g8813b57a44bdd30177666110530d1dcf>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphicsResourceGetMappedEglFrame ( [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>)*Â eglFrame, [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>)Â resource, unsigned int Â index, unsigned int Â mipLevel )


Get an eglFrame through which to access a registered EGL graphics resource.

######  Parameters

`eglFrame`
    \- Returned eglFrame.
`resource`
    \- Registered resource to access.
`index`
    \- Index for cubemap surfaces.
`mipLevel`
    \- Mipmap level for the subresource to access.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>)

###### Description

Returns in `*eglFrame` an eglFrame pointer through which the registered graphics resource `resource` may be accessed. This API can only be called for registered EGL graphics resources.

The CUeglFrame is defined as:


    â typedef struct CUeglFrame_st {
               union {
                   [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) pArray[[MAX_PLANES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4269718bae6e29c6059d666ec76df24b>)];
                   void*   pPitch[[MAX_PLANES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4269718bae6e29c6059d666ec76df24b>)];
               } frame;
               unsigned int width;
               unsigned int height;
               unsigned int depth;
               unsigned int pitch;
               unsigned int planeCount;
               unsigned int numChannels;
               [CUeglFrameType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec3f4a4e1a5785b1aa0fcc209cd47c38>) frameType;
               [CUeglColorFormat](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g38337000e43e400e77ad36c7e197a9f2>) eglColorFormat;
               [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) cuFormat;
           } [CUeglFrame](<structCUeglFrame__v1.html#structCUeglFrame__v1>);

If `resource` is not registered then [CUDA_ERROR_NOT_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e098433b926c9afdb6b6bdf191629447>) is returned. *

**See also:**

[cuGraphicsMapResources](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1gffcfd8e78d82cc4f6dd987e8bce4edb0> "Map graphics resources for access by CUDA."), [cuGraphicsSubResourceGetMappedArray](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g61c55e987e54558cce547240d6123078> "Get an array through which to access a subresource of a mapped graphics resource."), [cuGraphicsResourceGetMappedPointer](<group__CUDA__GRAPHICS.html#group__CUDA__GRAPHICS_1g8a634cf4150d399f0018061580592457> "Get a device pointer through which to access a mapped graphics resource."), [cudaGraphicsResourceGetMappedEglFrame](<../cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL_1gdd6215655a241c047d6d4939e242202a>)

* * *
