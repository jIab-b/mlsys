# Event Management

## 6.19.Â Event Management

This section describes the event management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventCreate](<#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â phEvent, unsigned int Â Flags )
     Creates an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventDestroy](<#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Destroys an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventElapsedTime](<#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97>) ( float*Â pMilliseconds, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hStart, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEnd )
     Computes the elapsed time between two events.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventQuery](<#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Queries an event's status.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventRecord](<#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Records an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventRecordWithFlags](<#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, unsigned int Â flags )
     Records an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuEventSynchronize](<#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Waits for an event to complete.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventCreate ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â phEvent, unsigned int Â Flags )


Creates an event.

######  Parameters

`phEvent`
    \- Returns newly created event
`Flags`
    \- Event creation flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates an event *phEvent for the current context with the flags specified via `Flags`. Valid flags include:

  * [CU_EVENT_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629e22adf5df73b0d43c6374a12ebee1333>): Default event creation flag.

  * [CU_EVENT_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f6296813b3b31fdb737133124f3c35044362>): Specifies that the created event should use blocking synchronization. A CPU thread that uses [cuEventSynchronize()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete.") to wait on an event created with this flag will block until the event has actually been recorded.

  * [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>): Specifies that the created event does not need to record timing data. Events created with this flag specified and the [CU_EVENT_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f6296813b3b31fdb737133124f3c35044362>) flag not specified will provide the best performance when used with [cuStreamWaitEvent()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.") and [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.").

  * [CU_EVENT_INTERPROCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629adab662356d24cf59f3d7de07c3cd52e>): Specifies that the created event may be used as an interprocess event by [cuIpcGetEventHandle()](<group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c> "Gets an interprocess handle for a previously allocated event."). [CU_EVENT_INTERPROCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629adab662356d24cf59f3d7de07c3cd52e>) must be specified along with [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>).


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventElapsedTime](<group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97> "Computes the elapsed time between two events."), [cudaEventCreate](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g4b5fdb19d7fb5f6f8862559f9279f6c3>), [cudaEventCreateWithFlags](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g7b317e07ff385d85aa656204b971a042>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventDestroy ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Destroys an event.

######  Parameters

`hEvent`
    \- Event to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Destroys the event specified by `hEvent`.

An event may be destroyed before it is complete (i.e., while [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") would return [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>)). In this case, the call does not block on completion of the event, and any associated resources will automatically be released asynchronously at completion.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventElapsedTime](<group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97> "Computes the elapsed time between two events."), [cudaEventDestroy](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2cb6baa0830a1cd0bd957bfd8705045b>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventElapsedTime ( float*Â pMilliseconds, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hStart, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEnd )


Computes the elapsed time between two events.

######  Parameters

`pMilliseconds`
    \- Time between `hStart` and `hEnd` in ms
`hStart`
    \- Starting event
`hEnd`
    \- Ending event

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds). Note this API is not guaranteed to return the latest errors for pending work. As such this API is intended to serve as an elapsed time calculation only and any polling for completion on the events to be compared should be done with [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") instead.

If either event was last recorded in a non-NULL stream, the resulting time may be greater than expected (even if both used the same stream handle). This happens because the [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") operation takes place asynchronously and there is no guarantee that the measured latency is actually just between the two events. Any number of other different stream operations could execute in between the two measured events, thus altering the timing in a significant way.

If [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") has not been called on either event then [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. If [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") has been called on both events but one or both of them has not yet been completed (that is, [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") would return [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>) on at least one of the events), [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>) is returned. If either event was created with the [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>) flag, then this function will return [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cudaEventElapsedTime](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g40159125411db92c835edb46a0989cd6>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventQuery ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Queries an event's status.

######  Parameters

`hEvent`
    \- Event to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>)

###### Description

Queries the status of all work currently captured by `hEvent`. See [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") for details on what is captured by an event.

Returns [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) if all captured work has been completed, or [CUDA_ERROR_NOT_READY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9edd9cef666ce620352e619a36b6c3f34>) if any captured work is incomplete.

For the purposes of Unified Memory, a return value of [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) is equivalent to having called [cuEventSynchronize()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventElapsedTime](<group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97> "Computes the elapsed time between two events."), [cudaEventQuery](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventRecord ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Records an event.

######  Parameters

`hEvent`
    \- Event to record
`hStream`
    \- Stream to record event for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Captures in `hEvent` the contents of `hStream` at the time of this call. `hEvent` and `hStream` must be from the same context otherwise [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. Calls such as [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") or [cuStreamWaitEvent()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.") will then examine or wait for completion of the work that was captured. Uses of `hStream` after this call do not modify `hEvent`. See note on default stream behavior for what is captured in the default case.

[cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") can be called multiple times on the same event and will overwrite the previously captured state. Other APIs such as [cuStreamWaitEvent()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.") use the most recently captured state at the time of the API call, and are not affected by later calls to [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."). Before the first call to [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), an event represents an empty set of work, so for example [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") would return [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>).

Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventElapsedTime](<group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97> "Computes the elapsed time between two events."), [cudaEventRecord](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446>), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventRecordWithFlags ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, unsigned int Â flags )


Records an event.

######  Parameters

`hEvent`
    \- Event to record
`hStream`
    \- Stream to record event for
`flags`
    \- See CUevent_capture_flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Captures in `hEvent` the contents of `hStream` at the time of this call. `hEvent` and `hStream` must be from the same context otherwise [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. Calls such as [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") or [cuStreamWaitEvent()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.") will then examine or wait for completion of the work that was captured. Uses of `hStream` after this call do not modify `hEvent`. See note on default stream behavior for what is captured in the default case.

[cuEventRecordWithFlags()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event.") can be called multiple times on the same event and will overwrite the previously captured state. Other APIs such as [cuStreamWaitEvent()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.") use the most recently captured state at the time of the API call, and are not affected by later calls to [cuEventRecordWithFlags()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."). Before the first call to [cuEventRecordWithFlags()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), an event represents an empty set of work, so for example [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") would return [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>).

flags include:

  * [CU_EVENT_RECORD_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg223a74c96434bb4e1d01c8685cbdef2257b646d34f7e5a14ef353de72b7ae091>): Default event creation flag.

  * [CU_EVENT_RECORD_EXTERNAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg223a74c96434bb4e1d01c8685cbdef22a47a6b397953152dfed8186e84b686b2>): Event is captured in the graph as an external event node when performing stream capture. This flag is invalid outside of stream capture.


Note:

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventElapsedTime](<group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97> "Computes the elapsed time between two events."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cudaEventRecord](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuEventSynchronize ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Waits for an event to complete.

######  Parameters

`hEvent`
    \- Event to wait for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Waits until the completion of all work currently captured in `hEvent`. See [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") for details on what is captured by an event.

Waiting for an event that was created with the [CU_EVENT_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f6296813b3b31fdb737133124f3c35044362>) flag will cause the calling CPU thread to block until the event has been completed by the device. If the [CU_EVENT_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f6296813b3b31fdb737133124f3c35044362>) flag has not been set, then the CPU thread will busy-wait until the event has been completed by the device.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventElapsedTime](<group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97> "Computes the elapsed time between two events."), [cudaEventSynchronize](<../cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g949aa42b30ae9e622f6ba0787129ff22>)

* * *
