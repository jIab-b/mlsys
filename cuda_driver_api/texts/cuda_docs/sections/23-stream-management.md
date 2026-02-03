# Stream Management

## 6.18.Â Stream Management

This section describes the stream management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamAddCallback](<#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCallback](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge5743a8c48527f1040107a68205c5ba9>)Â callback, void*Â userData, unsigned int Â flags )
     Add a callback to a compute stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamAttachMemAsync](<#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ length, unsigned int Â flags )
     Attach memory to a stream asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamBeginCapture](<#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>)Â mode )
     Begins graph capture on a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamBeginCaptureToGraph](<#group__CUDA__STREAM_1gac495e0527d1dd6437f95ee482f61865>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â dependencyData, size_tÂ numDependencies, [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>)Â mode )
     Begins graph capture on a stream to an existing graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamCopyAttributes](<#group__CUDA__STREAM_1g680f5399f6126cc4a99bc5eee4c2eff0>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â dst, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â src )
     Copies attributes from source stream to destination stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamCreate](<#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â phStream, unsigned int Â Flags )
     Create a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamCreateWithPriority](<#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â phStream, unsigned int Â flags, int Â priority )
     Create a stream with the given priority.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamDestroy](<#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Destroys a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamEndCapture](<#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph )
     Ends capture on a stream, returning the captured graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetAttribute](<#group__CUDA__STREAM_1g0598bb5295f3a62761b93c2d63d2266c>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, [CUstreamAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value_out )
     Queries stream attribute.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetCaptureInfo](<#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCaptureStatus](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91>)*Â captureStatus_out, cuuint64_t*Â id_out, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â graph_out, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)**Â dependencies_out, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)**Â edgeData_out, size_t*Â numDependencies_out )
     Query a stream's capture state.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetCtx](<#group__CUDA__STREAM_1g1107907025eaa3387fdc590a9379a681>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx )
     Query the context associated with a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetCtx_v2](<#group__CUDA__STREAM_1gd7eab81f618ec370a92c5e7d88ea11fa>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)*Â pGreenCtx )
     Query the contexts associated with a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetDevice](<#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device )
     Returns the device handle of the stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetFlags](<#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, unsigned int*Â flags )
     Query the flags of a given stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetId](<#group__CUDA__STREAM_1g5dafd2b6f48caeb13d5110a7f21e60e3>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, unsigned long long*Â streamId )
     Returns the unique Id associated with the stream handle supplied.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetPriority](<#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, int*Â priority )
     Query the priority of a given stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamIsCapturing](<#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCaptureStatus](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91>)*Â captureStatus )
     Returns a stream's capture status.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamQuery](<#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Determine status of a compute stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamSetAttribute](<#group__CUDA__STREAM_1ga2c5fc0292861a42f264af6ca48be8c0>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, const [CUstreamAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value )
     Sets stream attribute.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamSynchronize](<#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Wait until a stream's tasks are completed.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamUpdateCaptureDependencies](<#group__CUDA__STREAM_1g0cd3210434f3e0796c492cfa0d4b4bd1>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â dependencyData, size_tÂ numDependencies, unsigned int Â flags )
     Update the set of dependencies in a capturing stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamWaitEvent](<#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent, unsigned int Â Flags )
     Make a compute stream wait on an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuThreadExchangeStreamCaptureMode](<#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d>) ( [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>)*Â mode )
     Swaps the stream capture interaction mode for a thread.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamAddCallback ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCallback](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge5743a8c48527f1040107a68205c5ba9>)Â callback, void*Â userData, unsigned int Â flags )


Add a callback to a compute stream.

######  Parameters

`hStream`
    \- Stream to add callback to
`callback`
    \- The function to call once preceding stream operations are complete
`userData`
    \- User specified data to be passed to the callback function
`flags`
    \- Reserved for future use, must be 0

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Note:

This function is slated for eventual deprecation and removal. If you do not require the callback to execute in case of a device error, consider using [cuLaunchHostFunc](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f> "Enqueues a host function call in a stream."). Additionally, this function is not supported with [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.") and [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph."), unlike [cuLaunchHostFunc](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f> "Enqueues a host function call in a stream.").

Adds a callback to be called on the host after all currently enqueued items in the stream have completed. For each cuStreamAddCallback call, the callback will be executed exactly once. The callback will block later work in the stream until it is finished.

The callback may be passed [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) or an error code. In the event of a device error, all subsequently executed callbacks will receive an appropriate [CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>).

Callbacks must not make any CUDA API calls. Attempting to use a CUDA API will result in [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>). Callbacks must not perform any synchronization that may depend on outstanding device work or other callbacks that are not mandated to run earlier. Callbacks without a mandated order (in independent streams) execute in undefined order and may be serialized.

For the purposes of Unified Memory, callback execution makes a number of guarantees:

  * The callback stream is considered idle for the duration of the callback. Thus, for example, a callback may always use memory attached to the callback stream.

  * The start of execution of a callback has the same effect as synchronizing an event recorded in the same stream immediately prior to the callback. It thus synchronizes streams which have been "joined" prior to the callback.

  * Adding device work to any stream does not have the effect of making the stream active until all preceding host functions and stream callbacks have executed. Thus, for example, a callback might use global attached memory even if work has been added to another stream, if the work has been ordered behind the callback with an event.

  * Completion of a callback does not cause a stream to become active except as described above. The callback stream will remain idle if no device work follows the callback, and will remain idle across consecutive callbacks without device work in between. Thus, for example, stream synchronization can be done by signaling from a callback at the end of the stream.


Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."), [cuStreamAttachMemAsync](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533> "Attach memory to a stream asynchronously."), [cuLaunchHostFunc](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f> "Enqueues a host function call in a stream."), [cudaStreamAddCallback](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g74aa9f4b1c2f12d994bf13876a5a2498>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamAttachMemAsync ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ length, unsigned int Â flags )


Attach memory to a stream asynchronously.

######  Parameters

`hStream`
    \- Stream in which to enqueue the attach operation
`dptr`
    \- Pointer to memory (must be a pointer to managed memory or to a valid host-accessible region of system-allocated pageable memory)
`length`
    \- Length of memory
`flags`
    \- Must be one of [CUmemAttach_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g17c5d5f9b585aa2d6f121847d1a78f4c>)

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Enqueues an operation in `hStream` to specify stream association of `length` bytes of memory starting from `dptr`. This function is a stream-ordered operation, meaning that it is dependent on, and will only take effect when, previous work in stream has completed. Any previous association is automatically replaced.

`dptr` must point to one of the following types of memories:

  * managed memory declared using the __managed__ keyword or allocated with [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.").

  * a valid host-accessible region of system-allocated pageable memory. This type of memory may only be specified if the device associated with the stream reports a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>).


For managed allocations, `length` must be either zero or the entire allocation's size. Both indicate that the entire allocation's stream association is being changed. Currently, it is not possible to change stream association for a portion of a managed allocation.

For pageable host allocations, `length` must be non-zero.

The stream association is specified using `flags` which must be one of [CUmemAttach_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g17c5d5f9b585aa2d6f121847d1a78f4c>). If the [CU_MEM_ATTACH_GLOBAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313>) flag is specified, the memory can be accessed by any stream on any device. If the [CU_MEM_ATTACH_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c8b59c62cab9c7a762338e5fae92e2e9c>) flag is specified, the program makes a guarantee that it won't access the memory on the device from any stream on a device that has a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). If the [CU_MEM_ATTACH_SINGLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c4b70b6a5e039f61eccc6b8db3dfac442>) flag is specified and `hStream` is associated with a device that has a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>), the program makes a guarantee that it will only access the memory on the device from `hStream`. It is illegal to attach singly to the NULL stream, because the NULL stream is a virtual global stream and not a specific stream. An error will be returned in this case.

When memory is associated with a single stream, the Unified Memory system will allow CPU access to this memory region so long as all operations in `hStream` have completed, regardless of whether other streams are active. In effect, this constrains exclusive ownership of the managed memory region by an active GPU to per-stream activity instead of whole-GPU activity.

Accessing memory on the device from streams that are not associated with it will produce undefined results. No error checking is performed by the Unified Memory system to ensure that kernels launched into other streams do not access this region.

It is a program's responsibility to order calls to [cuStreamAttachMemAsync](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533> "Attach memory to a stream asynchronously.") via events, synchronization or other means to ensure legal access to memory at all times. Data visibility and coherency will be changed appropriately for all kernels which follow a stream-association change.

If `hStream` is destroyed while data is associated with it, the association is removed and the association reverts to the default visibility of the allocation as specified at [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."). For __managed__ variables, the default association is always [CU_MEM_ATTACH_GLOBAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313>). Note that destroying a stream is an asynchronous operation, and as a result, the change to default association won't happen until all work in the stream has completed.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."), [cudaStreamAttachMemAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g496353d630c29c44a2e33f531a3944d1>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamBeginCapture ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>)Â mode )


Begins graph capture on a stream.

######  Parameters

`hStream`
    \- Stream in which to initiate capture
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cuThreadExchangeStreamCaptureMode](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d> "Swaps the stream capture interaction mode for a thread.").

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Begin graph capture on `hStream`. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into a graph, which will be returned via [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph."). Capture may not be initiated if `stream` is CU_STREAM_LEGACY. Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via [cuStreamIsCapturing](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca> "Returns a stream's capture status."). A unique id representing the capture sequence may be queried via [cuStreamGetCaptureInfo](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce> "Query a stream's capture state.").

If `mode` is not CU_STREAM_CAPTURE_MODE_RELAXED, [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.") must be called on this stream from the same thread.

Note:

Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamIsCapturing](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca> "Returns a stream's capture status."), [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph."), [cuThreadExchangeStreamCaptureMode](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d> "Swaps the stream capture interaction mode for a thread.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamBeginCaptureToGraph ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â dependencyData, size_tÂ numDependencies, [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>)Â mode )


Begins graph capture on a stream to an existing graph.

######  Parameters

`hStream`
    \- Stream in which to initiate capture.
`hGraph`
    \- Graph to capture into.
`dependencies`
    \- Dependencies of the first node captured in the stream. Can be NULL if numDependencies is 0.
`dependencyData`
    \- Optional array of data associated with each dependency.
`numDependencies`
    \- Number of dependencies.
`mode`
    \- Controls the interaction of this capture sequence with other API calls that are potentially unsafe. For more details see [cuThreadExchangeStreamCaptureMode](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d> "Swaps the stream capture interaction mode for a thread.").

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Begin graph capture on `hStream`, placing new nodes into an existing graph. When a stream is in capture mode, all operations pushed into the stream will not be executed, but will instead be captured into `hGraph`. The graph will not be instantiable until the user calls [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.").

Capture may not be initiated if `stream` is CU_STREAM_LEGACY. Capture must be ended on the same stream in which it was initiated, and it may only be initiated if the stream is not already in capture mode. The capture mode may be queried via [cuStreamIsCapturing](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca> "Returns a stream's capture status."). A unique id representing the capture sequence may be queried via [cuStreamGetCaptureInfo](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce> "Query a stream's capture state.").

If `mode` is not CU_STREAM_CAPTURE_MODE_RELAXED, [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.") must be called on this stream from the same thread.

Note:

Kernels captured using this API must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamIsCapturing](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca> "Returns a stream's capture status."), [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph."), [cuThreadExchangeStreamCaptureMode](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d> "Swaps the stream capture interaction mode for a thread."), [cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamCopyAttributes ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â dst, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â src )


Copies attributes from source stream to destination stream.

######  Parameters

`dst`
    Destination stream
`src`
    Source stream For list of attributes see CUstreamAttrID

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies attributes from source stream `src` to destination stream `dst`. Both streams must have the same context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamCreate ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â phStream, unsigned int Â Flags )


Create a stream.

######  Parameters

`phStream`
    \- Returned newly created stream
`Flags`
    \- Parameters for stream creation

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates a stream and returns a handle in `phStream`. The `Flags` argument determines behaviors of the stream.

Valid values for `Flags` are:

  * [CU_STREAM_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752aaa5df0ec96f491f1be1124fdf265a066>): Default stream creation flag.

  * [CU_STREAM_NON_BLOCKING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752a89727d1d315214a6301abe98b419aff6>): Specifies that work running in the created stream may run concurrently with work in stream 0 (the NULL stream), and that the created stream should perform no implicit synchronization with stream 0.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream.")[cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamCreateWithPriority ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â phStream, unsigned int Â flags, int Â priority )


Create a stream with the given priority.

######  Parameters

`phStream`
    \- Returned newly created stream
`flags`
    \- Flags for stream creation. See [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream.") for a list of valid flags
`priority`
    \- Stream priority. Lower numbers represent higher priorities. See [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities.") for more information about meaningful stream priorities that can be passed.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates a stream with the specified priority and returns a handle in `phStream`. This affects the scheduling priority of work in the stream. Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order.

`priority` follows a convention where lower numbers represent higher priorities. '0' represents default priority. The range of meaningful numerical priorities can be queried using [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."). If the specified priority is outside the numerical range returned by [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), it will automatically be clamped to the lowest or the highest number in the range.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * Stream priorities are supported only on GPUs with compute capability 3.5 or higher.

  * In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory operations.


**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamCreateWithPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamDestroy ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Destroys a stream.

######  Parameters

`hStream`
    \- Stream to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Destroys the stream specified by `hStream`.

In case the device is still doing work in the stream `hStream` when [cuStreamDestroy()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream.") is called, the function will return immediately and the resources associated with `hStream` will be released automatically once the device has completed all work in `hStream`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamDestroy](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gfda584f1788ca983cb21c5f4d2033a62>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamEndCapture ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph )


Ends capture on a stream, returning the captured graph.

######  Parameters

`hStream`
    \- Stream to query
`phGraph`
    \- The captured graph

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e979282fa9b0bd6a56167b5ddf44391440>)

###### Description

End capture on `hStream`, returning the captured graph via `phGraph`. Capture must have been initiated on `hStream` via a call to [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream."). If capture was invalidated, due to a violation of the rules of stream capture, then a NULL graph will be returned.

If the `mode` argument to [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.") was not CU_STREAM_CAPTURE_MODE_RELAXED, this call must be from the same thread as [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream."), [cuStreamIsCapturing](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca> "Returns a stream's capture status."), [cuGraphDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8> "Destroys a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetAttribute ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, [CUstreamAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value_out )


Queries stream attribute.

######  Parameters

`hStream`

`attr`

`value_out`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Queries attribute `attr` from `hStream` and stores it in corresponding member of `value_out`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetCaptureInfo ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCaptureStatus](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91>)*Â captureStatus_out, cuuint64_t*Â id_out, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â graph_out, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)**Â dependencies_out, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)**Â edgeData_out, size_t*Â numDependencies_out )


Query a stream's capture state.

######  Parameters

`hStream`
    \- The stream to query
`captureStatus_out`
    \- Location to return the capture status of the stream; required
`id_out`
    \- Optional location to return an id for the capture sequence, which is unique over the lifetime of the process
`graph_out`
    \- Optional location to return the graph being captured into. All operations other than destroy and node removal are permitted on the graph while the capture sequence is in progress. This API does not transfer ownership of the graph, which is transferred or destroyed at [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph."). Note that the graph handle may be invalidated before end of capture for certain errors. Nodes that are or become unreachable from the original stream at [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.") due to direct actions on the graph do not trigger [CUDA_ERROR_STREAM_CAPTURE_UNJOINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9426e5dd5af746f6ee25aeb0f9fd32402>).
`dependencies_out`
    \- Optional location to store a pointer to an array of nodes. The next node to be captured in the stream will depend on this set of nodes, absent operations such as event wait which modify this set. The array pointer is valid until the next API call which operates on the stream or until the capture is terminated. The node handles may be copied out and are valid until they or the graph is destroyed. The driver-owned array may also be passed directly to APIs that operate on the graph (not the stream) without copying.
`edgeData_out`
    \- Optional location to store a pointer to an array of graph edge data. This array parallels `dependencies_out`; the next node to be added has an edge to `dependencies_out`[i] with annotation `edgeData_out`[i] for each `i`. The array pointer is valid until the next API call which operates on the stream or until the capture is terminated.
`numDependencies_out`
    \- Optional location to store the size of the array returned in dependencies_out.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe>), [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>)

###### Description

Query stream state related to stream capture.

If called on [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) (the "null stream") while a stream not created with [CU_STREAM_NON_BLOCKING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752a89727d1d315214a6301abe98b419aff6>) is capturing, returns [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe>).

Valid data (other than capture status) is returned only if both of the following are true:

  * the call returns CUDA_SUCCESS

  * the returned capture status is [CU_STREAM_CAPTURE_STATUS_ACTIVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb91c799fa3d867e2b300dfc45a6e90bc15d>)


If `edgeData_out` is non-NULL then `dependencies_out` must be as well. If `dependencies_out` is non-NULL and `edgeData_out` is NULL, but there is non-zero edge data for one or more of the current stream dependencies, the call will return [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>).

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream."), [cuStreamIsCapturing](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca> "Returns a stream's capture status."), [cuStreamUpdateCaptureDependencies](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g0cd3210434f3e0796c492cfa0d4b4bd1> "Update the set of dependencies in a capturing stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetCtx ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx )


Query the context associated with a stream.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`pctx`
    \- Returned context associated with the stream

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Returns the CUDA context that the stream is associated with.

If the stream was created via the API [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), the returned context is equivalent to the one returned by [cuCtxFromGreenCtx()](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context.") on the green context associated with the stream at creation time.

The stream handle `hStream` can refer to any of the following:

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream.") and [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), or their runtime API equivalents such as [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>) and [cudaStreamCreateWithPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540>). The returned context is the context that was active in the calling thread when the stream was created. Passing an invalid handle will result in undefined behavior.

  * any of the special streams such as the NULL stream, [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) and [CU_STREAM_PER_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c>). The runtime API equivalents of these are also accepted, which are NULL, [cudaStreamLegacy](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246>) and [cudaStreamPerThread](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197>) respectively. Specifying any of the special handles will return the context current to the calling thread. If no context is current to the calling thread, [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream.")[cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetCtx_v2 ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)*Â pGreenCtx )


Query the contexts associated with a stream.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`pCtx`
    \- Returned regular context associated with the stream
`pGreenCtx`
    \- Returned green context if the stream is associated with a green context or NULL if not

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Returns the contexts that the stream is associated with.

If the stream is associated with a green context, the API returns the green context in `pGreenCtx` and the primary context of the associated device in `pCtx`.

If the stream is associated with a regular context, the API returns the regular context in `pCtx` and NULL in `pGreenCtx`.

The stream handle `hStream` can refer to any of the following:

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority.") and [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), or their runtime API equivalents such as [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>) and [cudaStreamCreateWithPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540>). Passing an invalid handle will result in undefined behavior.

  * any of the special streams such as the NULL stream, [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) and [CU_STREAM_PER_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c>). The runtime API equivalents of these are also accepted, which are NULL, [cudaStreamLegacy](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246>) and [cudaStreamPerThread](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197>) respectively. If any of the special handles are specified, the API will operate on the context current to the calling thread. If a green context (that was converted via [cuCtxFromGreenCtx()](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context.") before setting it current) is current to the calling thread, the API will return the green context in `pGreenCtx` and the primary context of the associated device in `pCtx`. If a regular context is current, the API returns the regular context in `pCtx` and NULL in `pGreenCtx`. Note that specifying [CU_STREAM_PER_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c>) or [cudaStreamPerThread](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197>) will return [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) if a green context is current to the calling thread. If no context is current to the calling thread, [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) is returned.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream.")[cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>),

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetDevice ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device )


Returns the device handle of the stream.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`device`
    \- Returns the device to which a stream belongs

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Returns in `*device` the device handle of the stream

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetFlags ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, unsigned int*Â flags )


Query the flags of a given stream.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`flags`
    \- Pointer to an unsigned integer in which the stream's flags are returned The value returned in `flags` is a logical 'OR' of all flags that were used while creating this stream. See [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream.") for the list of valid flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Query the flags of a stream created using [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority.") or [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context.") and return the flags in `flags`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cudaStreamGetFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ga7f311f88126d751b9a7d3302ad6d0f8>), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetId ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, unsigned long long*Â streamId )


Returns the unique Id associated with the stream handle supplied.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`streamId`
    \- Pointer to store the Id of the stream

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Returns in `streamId` the unique Id which is associated with the given stream handle. The Id is unique for the life of the program.

The stream handle `hStream` can refer to any of the following:

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream.") and [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), or their runtime API equivalents such as [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>) and [cudaStreamCreateWithPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540>). Passing an invalid handle will result in undefined behavior.

  * any of the special streams such as the NULL stream, [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) and [CU_STREAM_PER_THREAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g02e40b82600f62c42ed29abb150f857c>). The runtime API equivalents of these are also accepted, which are NULL, [cudaStreamLegacy](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4e62d09dde16ba457b0a97f3a5262246>) and [cudaStreamPerThread](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7b7129befd6f52708309acafd1c46197>) respectively.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cudaStreamGetId](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g5799ae8dd744e561dfdeda02c53e82df>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetPriority ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, int*Â priority )


Query the priority of a given stream.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`priority`
    \- Pointer to a signed integer in which the stream's priority is returned

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Query the priority of a stream created using [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority.") or [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context.") and return the priority in `priority`. Note that if the stream was created with a priority outside the numerical range returned by [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), this function returns the clamped priority. See [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority.") for details about priority clamping.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream."), [cudaStreamGetPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamIsCapturing ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamCaptureStatus](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7480c0f2bd19894e54fcd2c04d6efb91>)*Â captureStatus )


Returns a stream's capture status.

######  Parameters

`hStream`
    \- Stream to query
`captureStatus`
    \- Returns the stream's capture status

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe>)

###### Description

Return the capture status of `hStream` via `captureStatus`. After a successful call, `*captureStatus` will contain one of the following:

  * [CU_STREAM_CAPTURE_STATUS_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb91e4023001f651dbdd3e3f55a1afc87fb3>): The stream is not capturing.

  * [CU_STREAM_CAPTURE_STATUS_ACTIVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb91c799fa3d867e2b300dfc45a6e90bc15d>): The stream is capturing.

  * [CU_STREAM_CAPTURE_STATUS_INVALIDATED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg7480c0f2bd19894e54fcd2c04d6efb916b8a69837a782cd52243d481a2c6f51a>): The stream was capturing but an error has invalidated the capture sequence. The capture sequence must be terminated with [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.") on the stream where it was initiated in order to continue using `hStream`.


Note that, if this is called on [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>) (the "null stream") while a blocking stream in the same context is capturing, it will return [CUDA_ERROR_STREAM_CAPTURE_IMPLICIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9960a55453736ec87ca941f9bc2d80abe>) and `*captureStatus` is unspecified after the call. The blocking stream capture is not invalidated.

When a blocking stream is capturing, the legacy stream is in an unusable state until the blocking stream capture is terminated. The legacy stream is not supported for stream capture, but attempted use would have an implicit dependency on the capturing stream(s).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream."), [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamQuery ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Determine status of a compute stream.

######  Parameters

`hStream`
    \- Stream to query status of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>)

###### Description

Returns [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) if all operations in the stream specified by `hStream` have completed, or [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>) if not.

For the purposes of Unified Memory, a return value of [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) is equivalent to having called [cuStreamSynchronize()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed.").

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamQuery](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamSetAttribute ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUstreamAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, const [CUstreamAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value )


Sets stream attribute.

######  Parameters

`hStream`

`attr`

`value`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Sets attribute `attr` on `hStream` from corresponding attribute of `value`. The updated attribute will be applied to subsequent work submitted to the stream. It will not affect previously submitted work.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamSynchronize ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Wait until a stream's tasks are completed.

######  Parameters

`hStream`
    \- Stream to wait for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Waits until the device has completed all operations in the stream specified by `hStream`. If the context was created with the [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>) flag, the CPU thread will block until the stream is finished with all of its tasks.

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamSynchronize](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamUpdateCaptureDependencies ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â dependencyData, size_tÂ numDependencies, unsigned int Â flags )


Update the set of dependencies in a capturing stream.

######  Parameters

`hStream`
    \- The stream to update
`dependencies`
    \- The set of dependencies to add
`dependencyData`
    \- Optional array of data associated with each dependency.
`numDependencies`
    \- The size of the dependencies array
`flags`
    \- See above

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>)

###### Description

Modifies the dependency set of a capturing stream. The dependency set is the set of nodes that the next captured node in the stream will depend on along with the edge data for those dependencies.

Valid flags are [CU_STREAM_ADD_CAPTURE_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggef58550e3d1f6d73c7e326455e744663bab808cd5e4e683f7000cb109973604e>) and [CU_STREAM_SET_CAPTURE_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggef58550e3d1f6d73c7e326455e744663e3ada3eef9666e592a2d4c3301d08fca>). These control whether the set passed to the API is added to the existing set or replaces it. A flags value of 0 defaults to [CU_STREAM_ADD_CAPTURE_DEPENDENCIES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggef58550e3d1f6d73c7e326455e744663bab808cd5e4e683f7000cb109973604e>).

Nodes that are removed from the dependency set via this API do not result in [CUDA_ERROR_STREAM_CAPTURE_UNJOINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9426e5dd5af746f6ee25aeb0f9fd32402>) if they are unreachable from the stream at [cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.").

Returns [CUDA_ERROR_ILLEGAL_STATE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9f5fd9c15b97a258f341bab23b0b505a5>) if the stream is not capturing.

**See also:**

[cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream."), [cuStreamGetCaptureInfo](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g85f03299332d6cf37578409d0e4b47ce> "Query a stream's capture state.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamWaitEvent ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent, unsigned int Â Flags )


Make a compute stream wait on an event.

######  Parameters

`hStream`
    \- Stream to wait
`hEvent`
    \- Event to wait on (may not be NULL)
`Flags`
    \- See CUevent_capture_flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>),

###### Description

Makes all future work submitted to `hStream` wait for all work captured in `hEvent`. See [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") for details on what is captured by an event. The synchronization will be performed efficiently on the device when applicable. `hEvent` may be from a different context or device than `hStream`.

flags include:

  * [CU_EVENT_WAIT_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg0dbe4cec219cab20846e3f269a5440d4ab2546b7da3337d9dd2bdec73c032e18>): Default event creation flag.

  * [CU_EVENT_WAIT_EXTERNAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg0dbe4cec219cab20846e3f269a5440d42e696252699844df830094402b2a83d7>): Event is captured in the graph as an external event node when performing stream capture. This flag is invalid outside of stream capture.


Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cudaStreamWaitEvent](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g7840e3984799941a61839de40413d1d9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuThreadExchangeStreamCaptureMode ( [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>)*Â mode )


Swaps the stream capture interaction mode for a thread.

######  Parameters

`mode`
    \- Pointer to mode value to swap with the current mode

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the calling thread's stream capture interaction mode to the value contained in `*mode`, and overwrites `*mode` with the previous mode for the thread. To facilitate deterministic behavior across function or module boundaries, callers are encouraged to use this API in a push-pop fashion:


    â     [CUstreamCaptureMode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd984ba65031d63f1ed11ec76728c2669>) mode = desiredMode;
               [cuThreadExchangeStreamCaptureMode](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d> "Swaps the stream capture interaction mode for a thread.")(&mode);
               ...
               [cuThreadExchangeStreamCaptureMode](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g378135b262f02a43a7caeab239ae493d> "Swaps the stream capture interaction mode for a thread.")(&mode); // restore previous mode

During stream capture (see [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.")), some actions, such as a call to [cudaMalloc](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356>), may be unsafe. In the case of [cudaMalloc](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356>), the operation is not enqueued asynchronously to a stream, and is not observed by stream capture. Therefore, if the sequence of operations captured via [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.") depended on the allocation being replayed whenever the graph is launched, the captured graph would be invalid.

Therefore, stream capture places restrictions on API calls that can be made within or concurrently to a [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.")-[cuStreamEndCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c> "Ends capture on a stream, returning the captured graph.") sequence. This behavior can be controlled via this API and flags to [cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.").

A thread's mode is one of the following:

  * `CU_STREAM_CAPTURE_MODE_GLOBAL:` This is the default mode. If the local thread has an ongoing capture sequence that was not initiated with `CU_STREAM_CAPTURE_MODE_RELAXED` at `cuStreamBeginCapture`, or if any other thread has a concurrent capture sequence initiated with `CU_STREAM_CAPTURE_MODE_GLOBAL`, this thread is prohibited from potentially unsafe API calls.

  * `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL:` If the local thread has an ongoing capture sequence not initiated with `CU_STREAM_CAPTURE_MODE_RELAXED`, it is prohibited from potentially unsafe API calls. Concurrent capture sequences in other threads are ignored.

  * `CU_STREAM_CAPTURE_MODE_RELAXED:` The local thread is not prohibited from potentially unsafe API calls. Note that the thread is still prohibited from API calls which necessarily conflict with stream capture, for example, attempting [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") on an event that was last recorded inside a capture sequence.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamBeginCapture](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143> "Begins graph capture on a stream.")

* * *
