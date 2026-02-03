# Context Management

## 6.8.Â Context Management

This section describes the context management functions of the low-level CUDA driver application programming interface.

Please note that some functions are described in [Primary Context Management](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX>) section.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxCreate](<#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx, [CUctxCreateParams](<structCUctxCreateParams.html#structCUctxCreateParams>)*Â ctxCreateParams, unsigned int Â flags, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Create a CUDA context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxDestroy](<#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Destroy a CUDA context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetApiVersion](<#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx, unsigned int*Â version )
     Gets the context's API version.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetCacheConfig](<#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360>) ( [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)*Â pconfig )
     Returns the preferred cache configuration for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetCurrent](<#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx )
     Returns the CUDA context bound to the calling CPU thread.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetDevice](<#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device )
     Returns the device handle for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetDevice_v2](<#group__CUDA__CTX_1gf0290a2b2de4c567f5c8c8262da58f60>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Returns the device handle for the specified context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetExecAffinity](<#group__CUDA__CTX_1g83421924a20536a4df538111cf61b405>) ( [CUexecAffinityParam](<structCUexecAffinityParam__v1.html#structCUexecAffinityParam__v1>)*Â pExecAffinity, [CUexecAffinityType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g14e6345acf2bda65be91eda77cf03f5c>)Â type )
     Returns the execution affinity setting for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetFlags](<#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d>) ( unsigned int*Â flags )
     Returns the flags for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetId](<#group__CUDA__CTX_1g32f492cd6c3f90af0d6935b294392db5>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx, unsigned long long*Â ctxId )
     Returns the unique Id associated with the context supplied.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetLimit](<#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8>) ( size_t*Â pvalue, [CUlimit](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665>)Â limit )
     Returns resource limits.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetStreamPriorityRange](<#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091>) ( int*Â leastPriority, int*Â greatestPriority )
     Returns numerical values that correspond to the least and greatest stream priorities.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxPopCurrent](<#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx )
     Pops the current CUDA context from the current CPU thread.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxPushCurrent](<#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Pushes a context on the current CPU thread.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxRecordEvent](<#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Records an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxResetPersistingL2Cache](<#group__CUDA__CTX_1gb529532b5b1aef808295a6d1d18a0823>) ( void )
     Resets all persisting lines in cache to normal status.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSetCacheConfig](<#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3>) ( [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)Â config )
     Sets the preferred cache configuration for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSetCurrent](<#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Binds the specified CUDA context to the calling CPU thread.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSetFlags](<#group__CUDA__CTX_1g66655c37602c8628eae3e40c82619f1e>) ( unsigned int Â flags )
     Sets the flags for the current context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSetLimit](<#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a>) ( [CUlimit](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665>)Â limit, size_tÂ value )
     Set resource limits.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSynchronize](<#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616>) ( void )
     Block for the current context's tasks to complete.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxSynchronize_v2](<#group__CUDA__CTX_1g7c57ec88e825af32ef8cc1754d69eca5>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Block for the specified context's tasks to complete.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxWaitEvent](<#group__CUDA__CTX_1gcf64e420275a8141b1f12bfce3f478f9>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Make a context wait on an event.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxCreate ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx, [CUctxCreateParams](<structCUctxCreateParams.html#structCUctxCreateParams>)*Â ctxCreateParams, unsigned int Â flags, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Create a CUDA context.

######  Parameters

`pctx`
    \- Returned context handle of the new context
`ctxCreateParams`
    \- Context creation parameters
`flags`
    \- Context creation flags
`dev`
    \- Device to create context on

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Creates a new CUDA context and associates it with the calling thread. The `flags` parameter is described below. The context is created with a usage count of 1 and the caller of [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") must call [cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context.") when done using the context. If a context is already current to the thread, it is supplanted by the newly created context and may be restored by a subsequent call to [cuCtxPopCurrent()](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread.").

CUDA context can be created with execution affinity. The type and the amount of execution resource the context can use is limited by `paramsArray` and `numExecAffinityParams` in `execAffinity`. The `paramsArray` is an array of `CUexecAffinityParam` and the `numExecAffinityParams` describes the size of the paramsArray. If two `CUexecAffinityParam` in the array have the same type, the latter execution affinity parameter overrides the former execution affinity parameter. The supported execution affinity types are:

  * [CU_EXEC_AFFINITY_TYPE_SM_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg14e6345acf2bda65be91eda77cf03f5cc7764c90ce81e15aba5f26a3507cd00c>) limits the portion of SMs that the context can use. The portion of SMs is specified as the number of SMs via `CUexecAffinitySmCount`. This limit will be internally rounded up to the next hardware-supported amount. Hence, it is imperative to query the actual execution affinity of the context via `cuCtxGetExecAffinity` after context creation. Currently, this attribute is only supported under Volta+ MPS.


CUDA context can be created in CIG(CUDA in Graphics) mode by setting `cigParams`. Data from graphics client is shared with CUDA via the `sharedData` in `cigParams`. Support for D3D12 graphics client can be determined using [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") with [CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a380194e4d9d3af41f1cc828eb54a99f1d>). `sharedData` is a ID3D12CommandQueue handle. Support for Vulkan graphics client can be determined using [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") with [CU_DEVICE_ATTRIBUTE_VULKAN_CIG_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3a3bed53361d118f632efe4776fc90cd3>). `sharedData` is a Nvidia specific data blob populated by calling vkGetExternalComputeQueueDataNV(). Either `execAffinityParams` or `cigParams` can be set to a non-null value. Setting both to a non-null value will result in an undefined behavior.

The three LSBs of the `flags` parameter can be used to control how the OS thread, which owns the CUDA context at the time of an API call, interacts with the OS scheduler when waiting for results from the GPU. Only one of the scheduling flags can be set when creating a context.

  * [CU_CTX_SCHED_SPIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4331d3ed1e0b55597258bd58346603afc>): Instruct CUDA to actively spin when waiting for results from the GPU. This can decrease latency when waiting for the GPU, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.


  * [CU_CTX_SCHED_YIELD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44bc43364906d8dd5a7d7c8ad46ccc548>): Instruct CUDA to yield its thread when waiting for results from the GPU. This can increase latency when waiting for the GPU, but can increase the performance of CPU threads performing work in parallel with the GPU.


  * [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>): Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the GPU to finish work.


  * [CU_CTX_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4b5bf395cc60a8cbded4c329ae9430b91>): Instruct CUDA to block the CPU thread on a synchronization primitive when waiting for the GPU to finish work.

**Deprecated:** This flag was deprecated as of CUDA 4.0 and was replaced with [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>).


  * [CU_CTX_SCHED_AUTO](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4d75f4c24f8c35ef2ee9d0793badfd88c>): The default value if the `flags` parameter is zero, uses a heuristic based on the number of active CUDA contexts in the process C and the number of logical processors in the system P. If C > P, then CUDA will yield to other OS threads when waiting for the GPU ([CU_CTX_SCHED_YIELD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44bc43364906d8dd5a7d7c8ad46ccc548>)), otherwise CUDA will not yield while waiting for results and actively spin on the processor ([CU_CTX_SCHED_SPIN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4331d3ed1e0b55597258bd58346603afc>)). Additionally, on Tegra devices, [CU_CTX_SCHED_AUTO](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4d75f4c24f8c35ef2ee9d0793badfd88c>) uses a heuristic based on the power profile of the platform and may choose [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>) for low-powered devices.


  * [CU_CTX_MAP_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd408c822db270f4322af6e6bb0a7786514>): Instruct CUDA to support mapped pinned allocations. This flag must be set in order to allocate pinned host memory that is accessible to the GPU.


  * [CU_CTX_LMEM_RESIZE_TO_MAX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4b5a83507c2a7e14d301621c40c343a81>): Instruct CUDA to not reduce local memory after resizing local memory for a kernel. This can prevent thrashing by local memory allocations when launching many kernels with high local memory usage at the cost of potentially increased memory usage.

**Deprecated:** This flag is deprecated and the behavior enabled by this flag is now the default and cannot be disabled. Instead, the per-thread stack size can be controlled with [cuCtxSetLimit()](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits.").


  * [CU_CTX_COREDUMP_ENABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44c74aa37941a780fccfbc2aa23e97809>): If GPU coredumps have not been enabled globally with [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") or environment variables, this flag can be set during context creation to instruct CUDA to create a coredump if this context raises an exception during execution. These environment variables are described in the CUDA-GDB user guide under the "GPU core dump support" section. The initial attributes will be taken from the global attributes at the time of context creation. The other attributes that control coredump output can be modified by calling [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context.") from the created context after it becomes current. This flag is not supported when CUDA context is created in CIG(CUDA in Graphics) mode.


  * [CU_CTX_USER_COREDUMP_ENABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd40703ba07a50ffbd294cb1122e08370d5>): If user-triggered GPU coredumps have not been enabled globally with [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") or environment variables, this flag can be set during context creation to instruct CUDA to create a coredump if data is written to a certain pipe that is present in the OS space. These environment variables are described in the CUDA-GDB user guide under the "GPU core dump support" section. It is important to note that the pipe name *must* be set with [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally.") before creating the context if this flag is used. Setting this flag implies that [CU_CTX_COREDUMP_ENABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd44c74aa37941a780fccfbc2aa23e97809>) is set. The initial attributes will be taken from the global attributes at the time of context creation. The other attributes that control coredump output can be modified by calling [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context.") from the created context after it becomes current. Setting this flag on any context creation is equivalent to setting the CU_COREDUMP_ENABLE_USER_TRIGGER attribute to `true` globally. This flag is not supported when CUDA context is created in CIG(CUDA in Graphics) mode.


  * [CU_CTX_SYNC_MEMOPS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd4e93e3bda3a8e71fe8f4d0de36aa881f0>): Ensures that synchronous memory operations initiated on this context will always synchronize. See further documentation in the section titled "API Synchronization behavior" to learn more about cases when synchronous memory operations can exhibit asynchronous behavior.


Context creation will fail with [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>) if the compute mode of the device is [CU_COMPUTEMODE_PROHIBITED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg637aab2eadb52e1c1c048b8bad9592d1db8a226241187db3b1f41999bb70eb47>). The function [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") can be used with [CU_DEVICE_ATTRIBUTE_COMPUTE_MODE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3f6669a29a6d42968047747cbfc501289>) to determine the compute mode of the device. The nvidia-smi tool can be used to set the compute mode for * devices. Documentation for nvidia-smi can be obtained by passing a -h option to it.

Context creation will fail with :: CUDA_ERROR_INVALID_VALUE if invalid parameter was passed by client to create the CUDA context.

Context creation in CIG mode will fail with [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>) if CIG is not supported by the device or the driver.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCoredumpSetAttributeGlobal](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1ga7645a8f68dd5379a03852b462727990> "Allows caller to set a coredump attribute value globally."), [cuCoredumpSetAttribute](<group__CUDA__COREDUMP.html#group__CUDA__COREDUMP_1g45b806050f3211e840eb3c8d91e93fcb> "Allows caller to set a coredump attribute value for the current context."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxDestroy ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Destroy a CUDA context.

######  Parameters

`ctx`
    \- Context to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Destroys the CUDA context specified by `ctx`. The context `ctx` will be destroyed regardless of how many threads it is current to. It is the responsibility of the calling function to ensure that no API call issues using `ctx` while [cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context.") is executing.

Destroys and cleans up all resources associated with the context. It is the caller's responsibility to ensure that the context or its resources are not accessed or passed in subsequent API calls and doing so will result in undefined behavior. These resources include CUDA types [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>), [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>), [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>), [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>), [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>), [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>), [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>), [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>), [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>), [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>), [CUgraphicsResource](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc0c4e1704647178d9c5ba3be46517dcd>), CUlinkState, [CUexternalMemory](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc9500ef066876b1186f8a54afff900ba>) and [CUexternalSemaphore](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0bc360152166a414e50a5167250552b8>). These resources also include memory allocations by [cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocManaged()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") and [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

If `ctx` is current to the calling thread then `ctx` will also be popped from the current thread's context stack (as though [cuCtxPopCurrent()](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread.") were called). If `ctx` is current to other threads, then `ctx` will remain current to those threads, and attempting to access `ctx` from those threads will result in the error [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>).

Note:

[cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context.") will not destroy memory allocations by [cuMemCreate()](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties."), [cuMemAllocAsync()](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") and [cuMemAllocFromPoolAsync()](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics."). These memory allocations are not associated with any CUDA context and need to be destroyed explicitly.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetApiVersion ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx, unsigned int*Â version )


Gets the context's API version.

######  Parameters

`ctx`
    \- Context to check
`version`
    \- Pointer to version

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns a version number in `version` corresponding to the capabilities of the context (e.g. 3010 or 3020), which library developers can use to direct callers to a specific API version. If `ctx` is NULL, returns the API version used to create the currently bound context.

Note that new API versions are only introduced when context capabilities are changed that break binary compatibility, so the API version and driver version may be different. For example, it is valid for the API version to be 3020 while the driver version is 4020.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetCacheConfig ( [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)*Â pconfig )


Returns the preferred cache configuration for the current context.

######  Parameters

`pconfig`
    \- Returned cache configuration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this function returns through `pconfig` the preferred cache configuration for the current context. This is only a preference. The driver will use the requested configuration if possible, but it is free to choose a different configuration if required to execute functions.

This will return a `pconfig` of [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>) on devices where the size of the L1 cache and shared memory are fixed.

The supported cache configurations are:

  * [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>): no preference for shared memory or L1 (default)

  * [CU_FUNC_CACHE_PREFER_SHARED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec712f43defb051d7985317bce426cccc8>): prefer larger shared memory and smaller L1 cache

  * [CU_FUNC_CACHE_PREFER_L1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ecb1e6c4e889e1a70ed5283172be08f6a5>): prefer larger L1 cache and smaller shared memory

  * [CU_FUNC_CACHE_PREFER_EQUAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec4434321280821d844a15b02e4d6c80a9>): prefer equal sized L1 cache and shared memory


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cudaDeviceGetCacheConfig](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gd9bf5eae6d464de05aa3840df9f5deeb>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetCurrent ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx )


Returns the CUDA context bound to the calling CPU thread.

######  Parameters

`pctx`
    \- Returned context handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>),

###### Description

Returns in `*pctx` the CUDA context bound to the calling CPU thread. If no context is bound to the calling CPU thread then `*pctx` is set to NULL and [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxSetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7> "Binds the specified CUDA context to the calling CPU thread."), [cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cudaGetDevice](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetDevice ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device )


Returns the device handle for the current context.

######  Parameters

`device`
    \- Returned device handle for the current context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns in `*device` the handle of the current context's device.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cudaGetDevice](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g80861db2ce7c29b6e8055af8ae01bc78>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetDevice_v2 ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â device, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Returns the device handle for the specified context.

######  Parameters

`device`
    \- Returned device handle for the specified context
`ctx`
    \- Context for which to obtain the device

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*device` the handle of the specified context's device. If the specified context is NULL, the API will return the current context's device.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0> "Returns the CUDA context bound to the calling CPU thread."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetExecAffinity ( [CUexecAffinityParam](<structCUexecAffinityParam__v1.html#structCUexecAffinityParam__v1>)*Â pExecAffinity, [CUexecAffinityType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g14e6345acf2bda65be91eda77cf03f5c>)Â type )


Returns the execution affinity setting for the current context.

######  Parameters

`pExecAffinity`
    \- Returned execution affinity
`type`
    \- Execution affinity type to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e909b6f0a40c9887de1bcc3ca48a75f1ad>)

###### Description

Returns in `*pExecAffinity` the current value of `type`. The supported [CUexecAffinityType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g14e6345acf2bda65be91eda77cf03f5c>) values are:

  * [CU_EXEC_AFFINITY_TYPE_SM_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg14e6345acf2bda65be91eda77cf03f5cc7764c90ce81e15aba5f26a3507cd00c>): number of SMs the context is limited to use.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUexecAffinityParam](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4e143c37c68ad44ff2b22922f5cd8341>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetFlags ( unsigned int*Â flags )


Returns the flags for the current context.

######  Parameters

`flags`
    \- Pointer to store flags of current context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns in `*flags` the flags of the current context. See [cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") for flag values.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0> "Returns the CUDA context bound to the calling CPU thread."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxGetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194> "Returns the current shared memory configuration for the current context."), [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), [cuCtxSetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1g66655c37602c8628eae3e40c82619f1e> "Sets the flags for the current context."), [cudaGetDeviceFlags](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf830794caf068b71638c6182bba8f77a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetId ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx, unsigned long long*Â ctxId )


Returns the unique Id associated with the context supplied.

######  Parameters

`ctx`
    \- Context for which to obtain the Id
`ctxId`
    \- Pointer to store the Id of the context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `ctxId` the unique Id which is associated with a given context. The Id is unique for the life of the program for this instance of CUDA. If context is supplied as NULL and there is one current, the Id of the current context is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetLimit ( size_t*Â pvalue, [CUlimit](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665>)Â limit )


Returns resource limits.

######  Parameters

`pvalue`
    \- Returned size of limit
`limit`
    \- Limit to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNSUPPORTED_LIMIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d10e6e6ef4b01290d2202d43c3ca6821>)

###### Description

Returns in `*pvalue` the current size of `limit`. The supported [CUlimit](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665>) values are:

  * [CU_LIMIT_STACK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665ebe51e384a8b4b79459915bb1c31bc39>): stack size in bytes of each GPU thread.

  * [CU_LIMIT_PRINTF_FIFO_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266516f25aa2c37a06580ab533d8ae7db948>): size in bytes of the FIFO used by the printf() device system call.

  * [CU_LIMIT_MALLOC_HEAP_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266586d01dbc431b04edd5d618257aaa246b>): size in bytes of the heap used by the malloc() and free() device system calls.

  * [CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665592fb752cc173ad7a2a4026a40e38079>): maximum grid depth at which a thread can issue the device runtime call [cudaDeviceSynchronize()](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d>) to wait on child grid launches to complete.

  * [CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665f79d7134ee03d52c0d8b1aecda1ae446>): maximum number of outstanding device runtime launches that can be made from this context.

  * [CU_LIMIT_MAX_L2_FETCH_GRANULARITY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665e75d95ea7dac6821de11d122d77f390b>): L2 cache fetch granularity.

  * [CU_LIMIT_PERSISTING_L2_CACHE_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266519ef5d58846147f46cdb4a2a886f3682>): Persisting L2 cache size in bytes


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cudaDeviceGetLimit](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetStreamPriorityRange ( int*Â leastPriority, int*Â greatestPriority )


Returns numerical values that correspond to the least and greatest stream priorities.

######  Parameters

`leastPriority`
    \- Pointer to an int in which the numerical value for least stream priority is returned
`greatestPriority`
    \- Pointer to an int in which the numerical value for greatest stream priority is returned

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns in `*leastPriority` and `*greatestPriority` the numerical values that correspond to the least and greatest stream priorities respectively. Stream priorities follow a convention where lower numbers imply greater priorities. The range of meaningful stream priorities is given by [`*greatestPriority`, `*leastPriority`]. If the user attempts to create a stream with a priority value that is outside the meaningful range as specified by this API, the priority is automatically clamped down or up to either `*leastPriority` or `*greatestPriority` respectively. See [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority.") for details on creating a priority stream. A NULL may be passed in for `*leastPriority` or `*greatestPriority` if the value is not desired.

This function will return '0' in both `*leastPriority` and `*greatestPriority` if the current context's device does not support stream priorities (see [cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.")).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cudaDeviceGetStreamPriorityRange](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gfdb79818f7c0ee7bc585648c91770275>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxPopCurrent ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pctx )


Pops the current CUDA context from the current CPU thread.

######  Parameters

`pctx`
    \- Returned popped context handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Pops the current CUDA context from the CPU thread and passes back the old context handle in `*pctx`. That context may then be made current to a different CPU thread by calling [cuCtxPushCurrent()](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread.").

If a context was current to the CPU thread before [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") or [cuCtxPushCurrent()](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread.") was called, this function makes that context current to the CPU thread again.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxPushCurrent ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Pushes a context on the current CPU thread.

######  Parameters

`ctx`
    \- Context to push

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Pushes the given context `ctx` onto the CPU thread's stack of current contexts. The specified context becomes the CPU thread's current context, so all CUDA functions that operate on the current context are affected.

The previous current context may be made current again by calling [cuCtxDestroy()](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context.") or [cuCtxPopCurrent()](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxRecordEvent ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Records an event.

######  Parameters

`hCtx`
    \- Context to record event for
`hEvent`
    \- Event to record

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>)

###### Description

Captures in `hEvent` all the activities of the context `hCtx` at the time of this call. `hEvent` and `hCtx` must be from the same CUDA context, otherwise [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) will be returned. Calls such as [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") or [cuCtxWaitEvent()](<group__CUDA__CTX.html#group__CUDA__CTX_1gcf64e420275a8141b1f12bfce3f478f9> "Make a context wait on an event.") will then examine or wait for completion of the work that was captured. Uses of `hCtx` after this call do not modify `hEvent`. If the context passed to `hCtx` is the primary context, `hEvent` will capture all the activities of the primary context and its green contexts. If the context passed to `hCtx` is a context converted from green context via [cuCtxFromGreenCtx()](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context."), `hEvent` will capture only the activities of the green context.

Note:

The API will return [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>) if the specified context `hCtx` has a stream in the capture mode. In such a case, the call will invalidate all the conflicting captures.

**See also:**

[cuCtxWaitEvent](<group__CUDA__CTX.html#group__CUDA__CTX_1gcf64e420275a8141b1f12bfce3f478f9> "Make a context wait on an event."), [cuGreenCtxRecordEvent](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g9dd087071cc217ad7ebda6df96d2ee40> "Records an event."), [cuGreenCtxWaitEvent](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6b26172117084fd024f1396fb66a8ffd> "Make a green context wait on an event."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxResetPersistingL2Cache ( void )


Resets all persisting lines in cache to normal status.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

[cuCtxResetPersistingL2Cache](<group__CUDA__CTX.html#group__CUDA__CTX_1gb529532b5b1aef808295a6d1d18a0823> "Resets all persisting lines in cache to normal status.") Resets all persisting lines in cache to normal status. Takes effect on function return.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSetCacheConfig ( [CUfunc_cache](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g3b9bbcf42528b889e9dbe9cfa2aea3ec>)Â config )


Sets the preferred cache configuration for the current context.

######  Parameters

`config`
    \- Requested cache configuration

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

On devices where the L1 cache and shared memory use the same hardware resources, this sets through `config` the preferred cache configuration for the current context. This is only a preference. The driver will use the requested configuration if possible, but it is free to choose a different configuration if required to execute the function. Any function preference set via [cuFuncSetCacheConfig()](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function.") or [cuKernelSetCacheConfig()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51> "Sets the preferred cache configuration for a device kernel.") will be preferred over this context-wide setting. Setting the context-wide cache configuration to [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>) will cause subsequent kernel launches to prefer to not change the cache configuration unless required to launch the kernel.

This setting does nothing on devices where the size of the L1 cache and shared memory are fixed.

Launching a kernel with a different preference than the most recent preference setting may insert a device-side synchronization point.

The supported cache configurations are:

  * [CU_FUNC_CACHE_PREFER_NONE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec47d2f367dc3965c27ff748688229dc22>): no preference for shared memory or L1 (default)

  * [CU_FUNC_CACHE_PREFER_SHARED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec712f43defb051d7985317bce426cccc8>): prefer larger shared memory and smaller L1 cache

  * [CU_FUNC_CACHE_PREFER_L1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ecb1e6c4e889e1a70ed5283172be08f6a5>): prefer larger L1 cache and smaller shared memory

  * [CU_FUNC_CACHE_PREFER_EQUAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg3b9bbcf42528b889e9dbe9cfa2aea3ec4434321280821d844a15b02e4d6c80a9>): prefer equal sized L1 cache and shared memory


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cuFuncSetCacheConfig](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g40f8c11e81def95dc0072a375f965681> "Sets the preferred cache configuration for a device function."), [cudaDeviceSetCacheConfig](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g6c9cc78ca80490386cf593b4baa35a15>), [cuKernelSetCacheConfig](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g8490476e5d3573c7ede78f29bd8cde51> "Sets the preferred cache configuration for a device kernel.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSetCurrent ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Binds the specified CUDA context to the calling CPU thread.

######  Parameters

`ctx`
    \- Context to bind to the calling CPU thread

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Binds the specified CUDA context to the calling CPU thread. If `ctx` is NULL then the CUDA context previously bound to the calling CPU thread is unbound and [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>) is returned.

If there exists a CUDA context stack on the calling CPU thread, this will replace the top of that stack with `ctx`. If `ctx` is NULL then this will be equivalent to popping the top of the calling CPU thread's CUDA context stack (or a no-op if the calling CPU thread's CUDA context stack is empty).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0> "Returns the CUDA context bound to the calling CPU thread."), [cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cudaSetDevice](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSetFlags ( unsigned int Â flags )


Sets the flags for the current context.

######  Parameters

`flags`
    \- Flags to set on the current context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the flags for the current context overwriting previously set ones. See [cuDevicePrimaryCtxSetFlags](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf> "Set flags for the primary context.") for flag values.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0> "Returns the CUDA context bound to the calling CPU thread."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxGetSharedMemConfig](<group__CUDA__CTX__DEPRECATED.html#group__CUDA__CTX__DEPRECATED_1gfac1414497a1a2a40bba474c6b5bf194> "Returns the current shared memory configuration for the current context."), [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cudaGetDeviceFlags](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gf830794caf068b71638c6182bba8f77a>), [cuDevicePrimaryCtxSetFlags](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1gd779a84f17acdad0d9143d9fe719cfdf> "Set flags for the primary context."),

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSetLimit ( [CUlimit](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665>)Â limit, size_tÂ value )


Set resource limits.

######  Parameters

`limit`
    \- Limit to set
`value`
    \- Size of limit

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNSUPPORTED_LIMIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d10e6e6ef4b01290d2202d43c3ca6821>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Setting `limit` to `value` is a request by the application to update the current limit maintained by the context. The driver is free to modify the requested value to meet h/w requirements (this could be clamping to minimum or maximum values, rounding up to nearest element size, etc). The application can use [cuCtxGetLimit()](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits.") to find out exactly what the limit has been set to.

Setting each [CUlimit](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge24c2d4214af24139020f1aecaf32665>) has its own specific restrictions, so each is discussed here.

  * [CU_LIMIT_STACK_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665ebe51e384a8b4b79459915bb1c31bc39>) controls the stack size in bytes of each GPU thread. The driver automatically increases the per-thread stack size for each kernel launch as needed. This size isn't reset back to the original value after each launch. Setting this value will take effect immediately, and if necessary, the device will block until all preceding requested tasks are complete.


  * [CU_LIMIT_PRINTF_FIFO_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266516f25aa2c37a06580ab533d8ae7db948>) controls the size in bytes of the FIFO used by the printf() device system call. Setting [CU_LIMIT_PRINTF_FIFO_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266516f25aa2c37a06580ab533d8ae7db948>) must be performed before launching any kernel that uses the printf() device system call, otherwise [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned.


  * [CU_LIMIT_MALLOC_HEAP_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266586d01dbc431b04edd5d618257aaa246b>) controls the size in bytes of the heap used by the malloc() and free() device system calls. Setting [CU_LIMIT_MALLOC_HEAP_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266586d01dbc431b04edd5d618257aaa246b>) must be performed before launching any kernel that uses the malloc() or free() device system calls, otherwise [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned.


  * [CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665592fb752cc173ad7a2a4026a40e38079>) controls the maximum nesting depth of a grid at which a thread can safely call [cudaDeviceSynchronize()](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d>). Setting this limit must be performed before any launch of a kernel that uses the device runtime and calls [cudaDeviceSynchronize()](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d>) above the default sync depth, two levels of grids. Calls to [cudaDeviceSynchronize()](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d>) will fail with error code [cudaErrorSyncDepthExceeded](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e0038265dbf94c45903cd582cfc40f93a176a>) if the limitation is violated. This limit can be set smaller than the default or up the maximum launch depth of 24. When setting this limit, keep in mind that additional levels of sync depth require the driver to reserve large amounts of device memory which can no longer be used for user allocations. If these reservations of device memory fail, [cuCtxSetLimit()](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits.") will return [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), and the limit can be reset to a lower value. This limit is only applicable to devices of compute capability < 9.0. Attempting to set this limit on devices of other compute capability versions will result in the error [CUDA_ERROR_UNSUPPORTED_LIMIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d10e6e6ef4b01290d2202d43c3ca6821>) being returned.


  * [CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665f79d7134ee03d52c0d8b1aecda1ae446>) controls the maximum number of outstanding device runtime launches that can be made from the current context. A grid is outstanding from the point of launch up until the grid is known to have been completed. Device runtime launches which violate this limitation fail and return [cudaErrorLaunchPendingCountExceeded](<../cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg3f51e3575c2178246db0a94a430e00382372902b9ffd65825d138e16125b1376>) when [cudaGetLastError()](<../cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g3529f94cb530a83a76613616782bd233>) is called after launch. If more pending launches than the default (2048 launches) are needed for a module using the device runtime, this limit can be increased. Keep in mind that being able to sustain additional pending launches will require the driver to reserve larger amounts of device memory upfront which can no longer be used for allocations. If these reservations fail, [cuCtxSetLimit()](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits.") will return [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), and the limit can be reset to a lower value. This limit is only applicable to devices of compute capability 3.5 and higher. Attempting to set this limit on devices of compute capability less than 3.5 will result in the error [CUDA_ERROR_UNSUPPORTED_LIMIT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d10e6e6ef4b01290d2202d43c3ca6821>) being returned.


  * [CU_LIMIT_MAX_L2_FETCH_GRANULARITY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf32665e75d95ea7dac6821de11d122d77f390b>) controls the L2 cache fetch granularity. Values can range from 0B to 128B. This is purely a performance hint and it can be ignored or clamped depending on the platform.


  * [CU_LIMIT_PERSISTING_L2_CACHE_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge24c2d4214af24139020f1aecaf3266519ef5d58846147f46cdb4a2a886f3682>) controls size in bytes available for persisting L2 cache. This is purely a performance hint and it can be ignored or clamped depending on the platform.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSynchronize](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete."), [cudaDeviceSetLimit](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSynchronize ( void )


Block for the current context's tasks to complete.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>)

###### Description

Blocks until the current context has completed all preceding requested tasks. If the current context is the primary context, green contexts that have been created will also be synchronized. [cuCtxSynchronize()](<group__CUDA__CTX.html#group__CUDA__CTX_1g7a54725f28d34b8c6299f0c6ca579616> "Block for the current context's tasks to complete.") returns an error if one of the preceding tasks failed. If the context was created with the [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>) flag, the CPU thread will block until the GPU context has finished its work.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context."), [cuCtxGetApiVersion](<group__CUDA__CTX.html#group__CUDA__CTX_1g088a90490dafca5893ef6fbebc8de8fb> "Gets the context's API version."), [cuCtxGetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g40b6b141698f76744dea6e39b9a25360> "Returns the preferred cache configuration for the current context."), [cuCtxGetDevice](<group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e> "Returns the device handle for the current context."), [cuCtxGetFlags](<group__CUDA__CTX.html#group__CUDA__CTX_1gf81eef983c1e3b2ef4f166d7a930c86d> "Returns the flags for the current context."), [cuCtxGetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8> "Returns resource limits."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuCtxSetCacheConfig](<group__CUDA__CTX.html#group__CUDA__CTX_1g54699acf7e2ef27279d013ca2095f4a3> "Sets the preferred cache configuration for the current context."), [cuCtxSetLimit](<group__CUDA__CTX.html#group__CUDA__CTX_1g0651954dfb9788173e60a9af7201e65a> "Set resource limits."), [cudaDeviceSynchronize](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxSynchronize_v2 ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Block for the specified context's tasks to complete.

######  Parameters

`ctx`
    \- Context to synchronize

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>)

###### Description

Blocks until the specified context has completed all preceding requested tasks. If the specified context is the primary context, green contexts that have been created will also be synchronized. The API returns an error if one of the preceding tasks failed.

If the context was created with the [CU_CTX_SCHED_BLOCKING_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f889e28a45a295b5c8ce13aa05f6cd462aebfe6432ade3feb32f1a409027852>) flag, the CPU thread will block until the GPU context has finished its work.

If the specified context is NULL, the API will operate on the current context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxGetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g8f13165846b73750693640fb3e8380d0> "Returns the CUDA context bound to the calling CPU thread."), [cuCtxPopCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1g2fac188026a062d92e91a8687d0a7902> "Pops the current CUDA context from the current CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources."), [cuCtxFromGreenCtx](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context."), [cudaDeviceSynchronize](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxWaitEvent ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Make a context wait on an event.

######  Parameters

`hCtx`
    \- Context to wait
`hEvent`
    \- Event to wait on

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>)

###### Description

Makes all future work submitted to context `hCtx` wait for all work captured in `hEvent`. The synchronization will be performed on the device and will not block the calling CPU thread. See [cuCtxRecordEvent()](<group__CUDA__CTX.html#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd> "Records an event.") for details on what is captured by an event. If the context passed to `hCtx` is the primary context, the primary context and its green contexts will wait for `hEvent`. If the context passed to `hCtx` is a context converted from green context via [cuCtxFromGreenCtx()](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context."), the green context will wait for `hEvent`.

Note:

  * `hEvent` may be from a different context or device than `hCtx`.

  * The API will return [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>) and invalidate the capture if the specified event `hEvent` is part of an ongoing capture sequence or if the specified context `hCtx` has a stream in the capture mode.


**See also:**

[cuCtxRecordEvent](<group__CUDA__CTX.html#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd> "Records an event."), [cuGreenCtxRecordEvent](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g9dd087071cc217ad7ebda6df96d2ee40> "Records an event."), [cuGreenCtxWaitEvent](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6b26172117084fd024f1396fb66a8ffd> "Make a green context wait on an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.")

* * *
