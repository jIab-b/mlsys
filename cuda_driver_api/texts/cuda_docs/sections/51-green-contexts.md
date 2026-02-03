# Green Contexts

## 6.35.Â Green Contexts

This section describes the APIs for creation and manipulation of green contexts in the CUDA driver. Green contexts are a lightweight alternative to traditional contexts, that can be used to select a subset of device resources. This allows the developer to, for example, select SMs from distinct spatial partitions of the GPU and target them via CUDA stream operations, kernel launches, etc.

Here are the broad initial steps to follow to get started:

  * (1) Start with an initial set of resources. For SM resources, they can be fetched via [cuDeviceGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6> "Get device resources."). In case of workqueues, a new configuration can be used or an existing one queried via the [cuDeviceGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6> "Get device resources.") API.

  * (2) Modify these resources by either partitioning them (in case of SMs) or changing the configuration (in case of workqueues). To partition SMs, we recommend [cuDevSmResourceSplit](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gc739e0a0b57766ee10437c89909865f3> "Splits a CU_DEV_RESOURCE_TYPE_SM resource into structured groups."). Changing the workqueue configuration can be done directly in place.

  * (3) Finalize the specification of resources by creating a descriptor via [cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor.").

  * (4) Create a green context via [cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources."). This provisions the resource, such as workqueues (until this step it was only a configuration specification).

  * (5) Create a stream via [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), and use it throughout your application.


**SMs**

There are two possible partition operations - with cuDevSmResourceSplitByCount the partitions created have to follow default SM count granularity requirements, so it will often be rounded up and aligned to a default value. On the other hand, [cuDevSmResourceSplit](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gc739e0a0b57766ee10437c89909865f3> "Splits a CU_DEV_RESOURCE_TYPE_SM resource into structured groups.") is explicit and allows for creation of non-equal groups. It will not round up automatically - instead it is the developerâs responsibility to query and set the correct values. These requirements can be queried with cuDeviceGetDevResource to determine the alignment granularity (sm.smCoscheduledAlignment). A general guideline on the default values for each compute architecture:

  * On Compute Architecture 7.X, 8.X, and all Tegra SoC:
    * The smCount must be a multiple of 2.

    * The alignment (and default value of coscheduledSmCount) is 2.

  * On Compute Architecture 9.0+:
    * The smCount must be a multiple of 8, or coscheduledSmCount if provided.

    * The alignment (and default value of coscheduledSmCount) is 8. While the maximum value for coscheduled SM count is 32 on all Compute Architecture 9.0+, it's recommended to follow cluster size requirements. The portable cluster size and the max cluster size should be used in order to benefit from this co-scheduling.


**Workqueues**

For `CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG`, the resource specifies the expected maximum number of concurrent stream-ordered workloads via the `wqConcurrencyLimit` field. The `sharingScope` field determines how workqueue resources are shared:

  * `CU_WORKQUEUE_SCOPE_DEVICE_CTX:` Use all shared workqueue resources across all contexts (default driver behavior).

  * `CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED:` When possible, use non-overlapping workqueue resources with other balanced green contexts.


The maximum concurrency limit depends on CUDA_DEVICE_MAX_CONNECTIONS and can be queried from the primary context via [cuCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g00d5b66f04f0591f1c752adb1a924635> "Get context resources."). Configurations may exceed this concurrency limit, but the driver will not guarantee that work submission remains non-overlapping.

For `CU_DEV_RESOURCE_TYPE_WORKQUEUE`, the resource represents a pre-existing workqueue that can be retrieved from existing contexts or green contexts. This allows reusing workqueue resources across different green contexts.

**On Concurrency**

Even if the green contexts have disjoint SM partitions, it is not guaranteed that the kernels launched in them will run concurrently or have forward progress guarantees. This is due to other resources that could cause a dependency. Using a combination of disjoint SMs and CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED workqueue configurations can provide the best chance of avoiding interference. More resources will be added in the future to provide stronger guarantees.

Additionally, there are two known scenarios, where its possible for the workload to run on more SMs than was provisioned (but never less).

  * On Volta+ MPS: When `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` is used, the set of SMs that are used for running kernels can be scaled up to the value of SMs used for the MPS client.

  * On Compute Architecture 9.x: When a module with dynamic parallelism (CDP) is loaded, all future kernels running under green contexts may use and share an additional set of 2 SMs.


### Classes

structÂ

[CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>)

     [](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>)
structÂ

[CUdevResource](<structCUdevResource.html#structCUdevResource>)

     [](<structCUdevResource.html#structCUdevResource>)
structÂ

[CUdevSmResource](<structCUdevSmResource.html#structCUdevSmResource>)

     [](<structCUdevSmResource.html#structCUdevSmResource>)
structÂ

[CUdevWorkqueueConfigResource](<structCUdevWorkqueueConfigResource.html#structCUdevWorkqueueConfigResource>)

     [](<structCUdevWorkqueueConfigResource.html#structCUdevWorkqueueConfigResource>)
structÂ

[CUdevWorkqueueResource](<structCUdevWorkqueueResource.html#structCUdevWorkqueueResource>)

     [](<structCUdevWorkqueueResource.html#structCUdevWorkqueueResource>)

### Typedefs

typedef CUdevResourceDesc_st * Â [CUdevResourceDesc](<#group__CUDA__GREEN__CONTEXTS_1g2384245122e1ee00c24a867404b55c17>)


### Enumerations

enumÂ [CUdevResourceType](<#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)

enumÂ [CUdevWorkqueueConfigScope](<#group__CUDA__GREEN__CONTEXTS_1g69e1b41d5b36d10d679b4c3b55dfb7a0>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxFromGreenCtx](<#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pContext, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx )
     Converts a green context into the primary context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxGetDevResource](<#group__CUDA__GREEN__CONTEXTS_1g00d5b66f04f0591f1c752adb1a924635>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â hCtx, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )
     Get context resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevResourceGenerateDesc](<#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5>) ( [CUdevResourceDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g2384245122e1ee00c24a867404b55c17>)*Â phDesc, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resources, unsigned int Â nbResources )
     Generate a resource descriptor.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevSmResourceSplit](<#group__CUDA__GREEN__CONTEXTS_1gc739e0a0b57766ee10437c89909865f3>) ( [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â result, unsigned int Â nbGroups, const [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â input, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â remainder, unsigned int Â flags, [CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>)*Â groupParams )
     Splits a `CU_DEV_RESOURCE_TYPE_SM` resource into structured groups.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDevSmResourceSplitByCount](<#group__CUDA__GREEN__CONTEXTS_1gf8359c74d7286ac32e5db253240d9a6c>) ( [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â result, unsigned int*Â nbGroups, const [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â input, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â remainder, unsigned int Â flags, unsigned int Â minCount )
     Splits `CU_DEV_RESOURCE_TYPE_SM` resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetDevResource](<#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )
     Get device resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxCreate](<#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65>) ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)*Â phCtx, [CUdevResourceDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g2384245122e1ee00c24a867404b55c17>)Â desc, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, unsigned int Â flags )
     Creates a green context with a specified set of resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxDestroy](<#group__CUDA__GREEN__CONTEXTS_1g7c37d959c2c030c13135366533eff57d>) ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx )
     Destroys a green context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxGetDevResource](<#group__CUDA__GREEN__CONTEXTS_1g301178535c06137ea8cc9becdbfb90b8>) ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )
     Get green context resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxGetId](<#group__CUDA__GREEN__CONTEXTS_1g2e073a7cde154f4fee132f8efd879b9c>) ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â greenCtx, unsigned long long*Â greenCtxId )
     Returns the unique Id associated with the green context supplied.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxRecordEvent](<#group__CUDA__GREEN__CONTEXTS_1g9dd087071cc217ad7ebda6df96d2ee40>) ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Records an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxStreamCreate](<#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â phStream, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â greenCtx, unsigned int Â flags, int Â priority )
     Create a stream for use in the green context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGreenCtxWaitEvent](<#group__CUDA__GREEN__CONTEXTS_1g6b26172117084fd024f1396fb66a8ffd>) ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )
     Make a green context wait on an event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetDevResource](<#group__CUDA__GREEN__CONTEXTS_1g25acc306a4e2ba88d1eb8b7bd2b2b578>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )
     Get stream resources.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuStreamGetGreenCtx](<#group__CUDA__GREEN__CONTEXTS_1gee3222277e5a433a2b279500bf11b9fe>) ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)*Â phCtx )
     Query the green context associated with a stream.

### Typedefs

typedef CUdevResourceDesc_st * CUdevResourceDesc


An opaque descriptor handle. The descriptor encapsulates multiple created and configured resources. Created via [cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor.")

### Enumerations

enum CUdevResourceType


Type of resource

######  Values

CU_DEV_RESOURCE_TYPE_INVALID = 0

CU_DEV_RESOURCE_TYPE_SM = 1
    Streaming multiprocessors related information
CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG = 1000
    Workqueue configuration related information
CU_DEV_RESOURCE_TYPE_WORKQUEUE = 10000
    Pre-existing workqueue related information

enum CUdevWorkqueueConfigScope


Sharing scope for workqueues

######  Values

CU_WORKQUEUE_SCOPE_DEVICE_CTX = 0
    Use all shared workqueue resources across all contexts. Default driver behaviour.
CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED = 1
    When possible, use non-overlapping workqueue resources with other balanced green contexts.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxFromGreenCtx ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pContext, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx )


Converts a green context into the primary context.

######  Parameters

`pContext`
    Returned primary context with green context resources
`hCtx`
    Green context to convert

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

The API converts a green context into the primary context returned in `pContext`. It is important to note that the converted context `pContext` is a normal primary context but with the resources of the specified green context `hCtx`. Once converted, it can then be used to set the context current with [cuCtxSetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7> "Binds the specified CUDA context to the calling CPU thread.") or with any of the CUDA APIs that accept a CUcontext parameter.

Users are expected to call this API before calling any CUDA APIs that accept a CUcontext. Failing to do so will result in the APIs returning [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>).

**See also:**

[cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxGetDevResource ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â hCtx, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )


Get context resources.

######  Parameters

`hCtx`
    \- Context to get resource for
`resource`
    \- Output pointer to a [CUdevResource](<structCUdevResource.html#structCUdevResource>) structure
`type`
    \- Type of resource to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Get the `type` resources available to the context represented by `hCtx` Note: The API is not supported on 32-bit platforms.

**See also:**

[cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevResourceGenerateDesc ( [CUdevResourceDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g2384245122e1ee00c24a867404b55c17>)*Â phDesc, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resources, unsigned int Â nbResources )


Generate a resource descriptor.

######  Parameters

`phDesc`
    \- Output descriptor
`resources`
    \- Array of resources to be included in the descriptor
`nbResources`
    \- Number of resources passed in `resources`

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e958d93275e1fc7c5879c1789fd9bc74a5>)

###### Description

Generates a single resource descriptor with the set of resources specified in `resources`. The generated resource descriptor is necessary for the creation of green contexts via the [cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources.") API. Resources of the same type can be passed in, provided they meet the requirements as noted below.

A successful API call must have:

  * A valid output pointer for the `phDesc` descriptor as well as a valid array of `resources` pointers, with the array size passed in `nbResources`. If multiple resources are provided in `resources`, the device they came from must be the same, otherwise CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION is returned. If multiple resources are provided in `resources` and they are of type [CU_DEV_RESOURCE_TYPE_SM](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gg28def480400d3254367d891d58f1375b3ba3ec7d6c8814ce80b9a633ca2ff332>), they must be outputs (whether `result` or `remaining`) from the same split API instance and have the same smCoscheduledAlignment values, otherwise CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION is returned.


Note: The API is not supported on 32-bit platforms.

**See also:**

[cuDevSmResourceSplitByCount](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf8359c74d7286ac32e5db253240d9a6c> "Splits CU_DEV_RESOURCE_TYPE_SM resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevSmResourceSplit ( [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â result, unsigned int Â nbGroups, const [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â input, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â remainder, unsigned int Â flags, [CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>)*Â groupParams )


Splits a `CU_DEV_RESOURCE_TYPE_SM` resource into structured groups.

######  Parameters

`result`
    \- Output array of `[CUdevResource](<structCUdevResource.html#structCUdevResource>)` resources. Can be NULL, alongside an smCount of 0, for discovery purpose.
`nbGroups`
    \- Specifies the number of groups in `result` and `groupParams`
`input`
    \- Input SM resource to be split. Must be a valid `CU_DEV_RESOURCE_TYPE_SM` resource.
`remainder`
    \- If splitting the input resource leaves any SMs, the remainder is placed in here.
`flags`
    \- Flags specifying how the API should behave. The value should be 0 for now.
`groupParams`
    \- Description of how the SMs should be split and assigned to the corresponding result entry.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e958d93275e1fc7c5879c1789fd9bc74a5>)

###### Description

This API will split a resource of [CU_DEV_RESOURCE_TYPE_SM](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gg28def480400d3254367d891d58f1375b3ba3ec7d6c8814ce80b9a633ca2ff332>) into `nbGroups` structured device resource groups (the `result` array), as well as an optional `remainder`, according to a set of requirements specified in the `groupParams` array. The term âstructuredâ is a trait that specifies the `result` has SMs that are co-scheduled together. This co-scheduling can be specified via the `coscheduledSmCount` field of the `groupParams` structure, while the `smCount` will specify how many SMs are required in total for that result. The remainder is always âunstructuredâ, it does not have any set guarantees with respect to co-scheduling and those properties will need to either be queried via the occupancy set of APIs or further split into structured groups by this API.

The API has a discovery mode for use cases where it is difficult to know ahead of time what the SM count should be. Discovery happens when the `smCount` field of a given `groupParams` array entry is set to 0 - the smCount will be filled in by the API with the derived SM count according to the provided `groupParams` fields and constraints. Discovery can be used with both a valid result array and with a NULL `result` pointer value. The latter is useful in situations where the smCount will end up being zero, which is an invalid value to create a result entry with, but allowed for discovery purposes when the `result` is NULL.

The `groupParams` array is evaluated from index 0 to `nbGroups` \- 1. For each index in the `groupParams` array, the API will evaluate which SMs may be a good fit based on constraints and assign those SMs to `result`. This evaluation order is important to consider when using discovery mode, as it helps discover the remaining SMs.

For a valid call:

  * `result` should point to a `[CUdevResource](<structCUdevResource.html#structCUdevResource>)` array of size `nbGroups`, or alternatively, may be NULL, if the developer wishes for only the groupParams entries to be updated


  * `input` should be a valid [CU_DEV_RESOURCE_TYPE_SM](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gg28def480400d3254367d891d58f1375b3ba3ec7d6c8814ce80b9a633ca2ff332>) resource that originates from querying the green context, device context, or device.


  * The `remainder` group may be NULL.


  * There are no API `flags` at this time, so the value passed in should be 0.


  * A [CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>) array of size `nbGroups`. Each entry must be zero-initialized.
    * `smCount:` must be either 0 or in the range of [2,inputSmCount] where inputSmCount is the amount of SMs the `input` resource has. `smCount` must be a multiple of 2, as well as a multiple of `coscheduledSmCount`. When assigning SMs to a group (and if results are expected by having the `result` parameter set), `smCount` cannot end up with 0 or a value less than `coscheduledSmCount` otherwise CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION will be returned.

    * `coscheduledSmCount:` allows grouping SMs together in order to be able to launch clusters on Compute Architecture 9.0+. The default value may be queried from the deviceâs [CU_DEV_RESOURCE_TYPE_SM](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gg28def480400d3254367d891d58f1375b3ba3ec7d6c8814ce80b9a633ca2ff332>) resource (8 on Compute Architecture 9.0+ and 2 otherwise). The maximum is 32 on Compute Architecture 9.0+ and 2 otherwise.

    * `preferredCoscheduledSmCount:` Attempts to merge `coscheduledSmCount` groups into larger groups, in order to make use of `preferredClusterDimensions` on Compute Architecture 10.0+. The default value is set to `coscheduledSmCount`.

    * `flags:`
      * `CU_DEV_SM_RESOURCE_SPLIT_BACKFILL:` lets `smCount` be a non-multiple of `coscheduledSmCount`, filling the difference between SM count and already assigned co-scheduled groupings with other SMs. This lets any resulting group behave similar to the `remainder` group for example.


**Example params and their effect:**

A groupParams array element is defined in the following order:


    â { .smCount, .coscheduledSmCount, .preferredCoscheduledSmCount, .flags, \/\* .reserved \*\/ }


    â// Example 1
          // Will discover how many SMs there are, that are co-scheduled in groups of smCoscheduledAlignment.
          // The rest is placed in the optional remainder.
          [CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>) params { 0, 0, 0, 0 };


    â// Example 2
          // Assuming the device has 10+ SMs, the result will have 10 SMs that are co-scheduled in groups of 2 SMs.
          // The rest is placed in the optional remainder.
          [CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>) params { 10, 2, 0, 0};
          // Setting the coscheduledSmCount to 2 guarantees that we can always have a valid result
          // as long as the SM count is less than or equal to the input resource SM count.


    â// Example 3
          // A single piece is split-off, but instead of assigning the rest to the remainder, a second group contains everything else
          // This assumes the device has 10+ SMs (8 of which are coscheduled in groups of 4),
          // otherwise the second group could end up with 0 SMs, which is not allowed.
          [CU_DEV_SM_RESOURCE_GROUP_PARAMS](<structCU__DEV__SM__RESOURCE__GROUP__PARAMS.html#structCU__DEV__SM__RESOURCE__GROUP__PARAMS>) params { {8, 4, 0, 0}, {0, 2, 0, CU_DEV_SM_RESOURCE_SPLIT_BACKFILL } }

The difference between a catch-all param group as the last entry and the remainder is in two aspects:

  * The remainder may be NULL / _TYPE_INVALID (if there are no SMs remaining), while a result group must always be valid.

  * The remainder does not have a structure, while the result group will always need to adhere to a structure of coscheduledSmCount (even if its just 2), and therefore must always have enough coscheduled SMs to cover that requirement (even with the `CU_DEV_SM_RESOURCE_SPLIT_BACKFILL` flag enabled).


Splitting an input into N groups, can be accomplished by repeatedly splitting off 1 group and re-splitting the remainder (a bisect operation). However, it's recommended to accomplish this with a single call wherever possible.

**See also:**

[cuGreenCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g301178535c06137ea8cc9becdbfb90b8> "Get green context resources."), [cuCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g00d5b66f04f0591f1c752adb1a924635> "Get context resources."), [cuDeviceGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6> "Get device resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDevSmResourceSplitByCount ( [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â result, unsigned int*Â nbGroups, const [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â input, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â remainder, unsigned int Â flags, unsigned int Â minCount )


Splits `CU_DEV_RESOURCE_TYPE_SM` resources.

######  Parameters

`result`
    \- Output array of `[CUdevResource](<structCUdevResource.html#structCUdevResource>)` resources. Can be NULL to query the number of groups.
`nbGroups`
    \- This is a pointer, specifying the number of groups that would be or should be created as described below.
`input`
    \- Input SM resource to be split. Must be a valid `CU_DEV_RESOURCE_TYPE_SM` resource.
`remainder`
    \- If the input resource cannot be cleanly split among `nbGroups`, the remainder is placed in here. Can be ommitted (NULL) if the user does not need the remaining set.
`flags`
    \- Flags specifying how these partitions are used or which constraints to abide by when splitting the input. Zero is valid for default behavior.
`minCount`
    \- Minimum number of SMs required

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e958d93275e1fc7c5879c1789fd9bc74a5>)

###### Description

Splits `CU_DEV_RESOURCE_TYPE_SM` resources into `nbGroups`, adhering to the minimum SM count specified in `minCount` and the usage flags in `flags`. If `result` is NULL, the API simulates a split and provides the amount of groups that would be created in `nbGroups`. Otherwise, `nbGroups` must point to the amount of elements in `result` and on return, the API will overwrite `nbGroups` with the amount actually created. The groups are written to the array in `result`. `nbGroups` can be less than the total amount if a smaller number of groups is needed.

This API is used to spatially partition the input resource. The input resource needs to come from one of [cuDeviceGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6> "Get device resources."), [cuCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g00d5b66f04f0591f1c752adb1a924635> "Get context resources."), or [cuGreenCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g301178535c06137ea8cc9becdbfb90b8> "Get green context resources."). A limitation of the API is that the output results cannot be split again without first creating a descriptor and a green context with that descriptor.

When creating the groups, the API will take into account the performance and functional characteristics of the input resource, and guarantee a split that will create a disjoint set of symmetrical partitions. This may lead to fewer groups created than purely dividing the total SM count by the `minCount` due to cluster requirements or alignment and granularity requirements for the minCount. These requirements can be queried with [cuDeviceGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6> "Get device resources."), [cuCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g00d5b66f04f0591f1c752adb1a924635> "Get context resources."), and [cuGreenCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g301178535c06137ea8cc9becdbfb90b8> "Get green context resources.") for [CU_DEV_RESOURCE_TYPE_SM](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gg28def480400d3254367d891d58f1375b3ba3ec7d6c8814ce80b9a633ca2ff332>), using the `minSmPartitionSize` and `smCoscheduledAlignment` fields to determine minimum partition size and alignment granularity, respectively.

The `remainder` set does not have the same functional or performance guarantees as the groups in `result`. Its use should be carefully planned and future partitions of the `remainder` set are discouraged.

The following flags are supported:

  * `CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING` : Lower the minimum SM count and alignment, and treat each SM independent of its hierarchy. This allows more fine grained partitions but at the cost of advanced features (such as large clusters on compute capability 9.0+).

  * `CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE` : Compute Capability 9.0+ only. Attempt to create groups that may allow for maximally sized thread clusters. This can be queried post green context creation using [cuOccupancyMaxPotentialClusterSize](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gd6f60814c1e3440145115ade3730365f> "Given the kernel function \(func\) and launch configuration \(config\), return the maximum cluster size in *clusterSize.").


A successful API call must either have:

  * A valid array of `result` pointers of size passed in `nbGroups`, with `input` of type `CU_DEV_RESOURCE_TYPE_SM`. Value of `minCount` must be between 0 and the SM count specified in `input`. `remainder` may be NULL.

  * NULL passed in for `result`, with a valid integer pointer in `nbGroups` and `input` of type `CU_DEV_RESOURCE_TYPE_SM`. Value of `minCount` must be between 0 and the SM count specified in `input`. `remainder` may be NULL. This queries the number of groups that would be created by the API.


Note: The API is not supported on 32-bit platforms.

**See also:**

[cuGreenCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g301178535c06137ea8cc9becdbfb90b8> "Get green context resources."), [cuCtxGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g00d5b66f04f0591f1c752adb1a924635> "Get context resources."), [cuDeviceGetDevResource](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6115d21604653f4eafb257f725538ab6> "Get device resources.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetDevResource ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )


Get device resources.

######  Parameters

`device`
    \- Device to get resource for
`resource`
    \- Output pointer to a [CUdevResource](<structCUdevResource.html#structCUdevResource>) structure
`type`
    \- Type of resource to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Get the `type` resources available to the `device`. This may often be the starting point for further partitioning or configuring of resources.

Note: The API is not supported on 32-bit platforms.

**See also:**

[cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxCreate ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)*Â phCtx, [CUdevResourceDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g2384245122e1ee00c24a867404b55c17>)Â desc, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, unsigned int Â flags )


Creates a green context with a specified set of resources.

######  Parameters

`phCtx`
    \- Pointer for the output handle to the green context
`desc`
    \- Descriptor generated via [cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor.") which contains the set of resources to be used
`dev`
    \- Device on which to create the green context.
`flags`
    \- One of the supported green context creation flags. `CU_GREEN_CTX_DEFAULT_STREAM` is required.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

This API creates a green context with the resources specified in the descriptor `desc` and returns it in the handle represented by `phCtx`. This API will retain the primary context on device `dev`, which will is released when the green context is destroyed. It is advised to have the primary context active before calling this API to avoid the heavy cost of triggering primary context initialization and deinitialization multiple times.

The API does not set the green context current. In order to set it current, you need to explicitly set it current by first converting the green context to a CUcontext using [cuCtxFromGreenCtx](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context.") and subsequently calling [cuCtxSetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7> "Binds the specified CUDA context to the calling CPU thread.") / [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."). It should be noted that a green context can be current to only one thread at a time. There is no internal synchronization to make API calls accessing the same green context from multiple threads work.

Note: The API is not supported on 32-bit platforms.

The supported flags are:

  * `CU_GREEN_CTX_DEFAULT_STREAM` : Creates a default stream to use inside the green context. Required.


**See also:**

[cuGreenCtxDestroy](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g7c37d959c2c030c13135366533eff57d> "Destroys a green context."), [cuCtxFromGreenCtx](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf0779ec72ce1d5d7eb003d7d9b25afcb> "Converts a green context into the primary context."), [cuCtxSetCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gbe562ee6258b4fcc272ca6478ca2a2f7> "Binds the specified CUDA context to the calling CPU thread."), [cuCtxPushCurrent](<group__CUDA__CTX.html#group__CUDA__CTX_1gb02d4c850eb16f861fe5a29682cc90ba> "Pushes a context on the current CPU thread."), [cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor."), [cuDevicePrimaryCtxRetain](<group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300> "Retain the primary context on the GPU."), [cuCtxCreate](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxDestroy ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx )


Destroys a green context.

######  Parameters

`hCtx`
    \- Green context to be destroyed

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Destroys the green context, releasing the primary context of the device that this green context was created for. Any resources provisioned for this green context (that were initially available via the resource descriptor) are released as well. The API does not destroy streams created via [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), or [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."). Users are expected to destroy these streams explicitly using [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream.") to avoid resource leaks. Once the green context is destroyed, any subsequent API calls involving these streams will return [CUDA_ERROR_STREAM_DETACHED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9aad337be88462a55bdcfcbff87d788c6>) with the exception of the following APIs:

  * [cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream.").


Additionally, the API will invalidate all active captures on these streams.

**See also:**

[cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources."), [cuCtxDestroy](<group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e> "Destroy a CUDA context.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxGetDevResource ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )


Get green context resources.

######  Parameters

`hCtx`
    \- Green context to get resource for
`resource`
    \- Output pointer to a [CUdevResource](<structCUdevResource.html#structCUdevResource>) structure
`type`
    \- Type of resource to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Get the `type` resources available to the green context represented by `hCtx`

**See also:**

[cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxGetId ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â greenCtx, unsigned long long*Â greenCtxId )


Returns the unique Id associated with the green context supplied.

######  Parameters

`greenCtx`
    \- Green context for which to obtain the Id
`greenCtxId`
    \- Pointer to store the Id of the green context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `greenCtxId` the unique Id which is associated with a given green context. The Id is unique for the life of the program for this instance of CUDA. If green context is supplied as NULL and the current context is set to a green context, the Id of the current green context is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources."), [cuGreenCtxDestroy](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g7c37d959c2c030c13135366533eff57d> "Destroys a green context."), [cuCtxGetId](<group__CUDA__CTX.html#group__CUDA__CTX_1g32f492cd6c3f90af0d6935b294392db5> "Returns the unique Id associated with the context supplied.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxRecordEvent ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Records an event.

######  Parameters

`hCtx`
    \- Green context to record event for
`hEvent`
    \- Event to record

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>)

###### Description

Captures in `hEvent` all the activities of the green context of `hCtx` at the time of this call. `hEvent` and `hCtx` must be from the same primary context otherwise [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>) is returned. Calls such as [cuEventQuery()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") or [cuGreenCtxWaitEvent()](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6b26172117084fd024f1396fb66a8ffd> "Make a green context wait on an event.") will then examine or wait for completion of the work that was captured. Uses of `hCtx` after this call do not modify `hEvent`.

Note:

The API will return [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>) if the specified green context `hCtx` has a stream in the capture mode. In such a case, the call will invalidate all the conflicting captures.

**See also:**

[cuGreenCtxWaitEvent](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g6b26172117084fd024f1396fb66a8ffd> "Make a green context wait on an event."), [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuCtxRecordEvent](<group__CUDA__CTX.html#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd> "Records an event."), [cuCtxWaitEvent](<group__CUDA__CTX.html#group__CUDA__CTX_1gcf64e420275a8141b1f12bfce3f478f9> "Make a context wait on an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxStreamCreate ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)*Â phStream, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â greenCtx, unsigned int Â flags, int Â priority )


Create a stream for use in the green context.

######  Parameters

`phStream`
    \- Returned newly created stream
`greenCtx`
    \- Green context for which to create the stream for
`flags`
    \- Flags for stream creation. `CU_STREAM_NON_BLOCKING` must be specified.
`priority`
    \- Stream priority. Lower numbers represent higher priorities. See [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities.") for more information about meaningful stream priorities that can be passed.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates a stream for use in the specified green context `greenCtx` and returns a handle in `phStream`. The stream can be destroyed by calling [cuStreamDestroy()](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."). Note that the API ignores the context that is current to the calling thread and creates a stream in the specified green context `greenCtx`.

The supported values for `flags` are:

  * [CU_STREAM_NON_BLOCKING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg775cb4ffbb7adf91e190067d9ad1752a89727d1d315214a6301abe98b419aff6>): This must be specified. It indicates that work running in the created stream may run concurrently with work in the default stream, and that the created stream should perform no implicit synchronization with the default stream.


Specifying `priority` affects the scheduling priority of work in the stream. Priorities provide a hint to preferentially run work with higher priority when possible, but do not preempt already-running work or provide any other functional guarantee on execution order. `priority` follows a convention where lower numbers represent higher priorities. '0' represents default priority. The range of meaningful numerical priorities can be queried using [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."). If the specified priority is outside the numerical range returned by [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), it will automatically be clamped to the lowest or the highest number in the range.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * In the current implementation, only compute kernels launched in priority streams are affected by the stream's priority. Stream priorities have no effect on host-to-device and device-to-host memory operations.


**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources.")[cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuCtxGetStreamPriorityRange](<group__CUDA__CTX.html#group__CUDA__CTX_1g137920ab61a71be6ce67605b9f294091> "Returns numerical values that correspond to the least and greatest stream priorities."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamCreateWithPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGreenCtxWaitEvent ( [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)Â hCtx, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â hEvent )


Make a green context wait on an event.

######  Parameters

`hCtx`
    \- Green context to wait
`hEvent`
    \- Event to wait on

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>)

###### Description

Makes all future work submitted to green context `hCtx` wait for all work captured in `hEvent`. The synchronization will be performed on the device and will not block the calling CPU thread. See [cuGreenCtxRecordEvent()](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g9dd087071cc217ad7ebda6df96d2ee40> "Records an event.") or [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), for details on what is captured by an event.

Note:

  * `hEvent` may be from a different context or device than `hCtx`.

  * The API will return [CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9adf26f72a5e6589c7ade9af3b1b62e3d>) and invalidate the capture if the specified event `hEvent` is part of an ongoing capture sequence or if the specified green context `hCtx` has a stream in the capture mode.


**See also:**

[cuGreenCtxRecordEvent](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g9dd087071cc217ad7ebda6df96d2ee40> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuCtxRecordEvent](<group__CUDA__CTX.html#group__CUDA__CTX_1gf3ee63561a7a371fa9d4dc0e31f94afd> "Records an event."), [cuCtxWaitEvent](<group__CUDA__CTX.html#group__CUDA__CTX_1gcf64e420275a8141b1f12bfce3f478f9> "Make a context wait on an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetDevResource ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUdevResource](<structCUdevResource.html#structCUdevResource>)*Â resource, [CUdevResourceType](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g28def480400d3254367d891d58f1375b>)Â type )


Get stream resources.

######  Parameters

`hStream`
    \- Stream to get resource for
`resource`
    \- Output pointer to a [CUdevResource](<structCUdevResource.html#structCUdevResource>) structure
`type`
    \- Type of resource to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Get the `type` resources available to the `hStream` and store them in `resource`.

Note: The API will return [CUDA_ERROR_INVALID_RESOURCE_TYPE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ec23220911f54aa1e66d8bcf86ec7ba>) is `type` is `CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG` or `CU_DEV_RESOURCE_TYPE_WORKQUEUE`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuGreenCtxCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1ga6da4f9959fd48d1f1a5cbedbec54e65> "Creates a green context with a specified set of resources."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuDevSmResourceSplitByCount](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1gf8359c74d7286ac32e5db253240d9a6c> "Splits CU_DEV_RESOURCE_TYPE_SM resources."), [cuDevResourceGenerateDesc](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g1ea7743fd633d2e2dd92eb1c84c4fbc5> "Generate a resource descriptor."), [cudaStreamGetDevResource](<../cuda-runtime-api/group__CUDART__EXECUTION__CONTEXT.html#group__CUDART__EXECUTION__CONTEXT_1g55c60bf05fec3cf837d96520c91b8396>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuStreamGetGreenCtx ( [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream, [CUgreenCtx](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g453cb79a1ceb13bec502a9c5f06a0268>)*Â phCtx )


Query the green context associated with a stream.

######  Parameters

`hStream`
    \- Handle to the stream to be queried
`phCtx`
    \- Returned green context associated with the stream

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>),

###### Description

Returns the CUDA green context that the stream is associated with, or NULL if the stream is not associated with any green context.

The stream handle `hStream` can refer to any of the following:

  * a stream created via any of the CUDA driver APIs such as [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority.") and [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), or their runtime API equivalents such as [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>) and [cudaStreamCreateWithPriority](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1ge2be9e9858849bf62ba4a8b66d1c3540>). If during stream creation the context that was active in the calling thread was obtained with cuCtxFromGreenCtx, that green context is returned in `phCtx`. Otherwise, `*phCtx` is set to NULL instead.

  * special stream such as the NULL stream or [CU_STREAM_LEGACY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ga53e8210837f039dd6434a3a4c3324aa>). In that case if context that is active in the calling thread was obtained with cuCtxFromGreenCtx, that green context is returned. Otherwise, `*phCtx` is set to NULL instead.


Passing an invalid handle will result in undefined behavior.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuStreamDestroy](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758> "Destroys a stream."), [cuStreamCreate](<group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4> "Create a stream."), [cuStreamCreateWithPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g95c1a8c7c3dacb13091692dd9c7f7471> "Create a stream with the given priority."), [cuStreamGetCtx](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1107907025eaa3387fdc590a9379a681> "Query the context associated with a stream."), [cuGreenCtxStreamCreate](<group__CUDA__GREEN__CONTEXTS.html#group__CUDA__GREEN__CONTEXTS_1g569b9e947b0f143f6ed9397a12046a8a> "Create a stream for use in the green context."), [cuStreamGetPriority](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g5bd5cb26915a2ecf1921807339488484> "Query the priority of a given stream."), [cuStreamGetFlags](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g4d39786855a6bed01215c1907fbbfbb7> "Query the flags of a given stream."), [cuStreamGetDevice](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1198995e0a122783ede50814b8c7a29b> "Returns the device handle of the stream."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuStreamQuery](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g1b0d24bbe97fa68e4bc511fb6adfeb0b> "Determine status of a compute stream."), [cuStreamSynchronize](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g15e49dd91ec15991eb7c0a741beb7dad> "Wait until a stream's tasks are completed."), [cuStreamAddCallback](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g613d97a277d7640f4cb1c03bd51c2483> "Add a callback to a compute stream."), [cudaStreamCreate](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g6a3c4b819e6a994c26d0c4824a4c80da>), [cudaStreamCreateWithFlags](<../cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1gb1e32aff9f59119e4d0a9858991c4ad3>)

* * *
