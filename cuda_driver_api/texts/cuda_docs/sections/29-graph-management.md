# Graph Management

## 6.24.Â Graph Management

This section describes the graph management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetGraphMemAttribute](<#group__CUDA__GRAPH_1g359903c2447ac22b4e1a0dce26adfef5>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUgraphMem_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5f76366f87bbdf761007768fe30a57db>)Â attr, void*Â value )
     Query asynchronous allocation attributes related to graphs.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGraphMemTrim](<#group__CUDA__GRAPH_1g57c87f4ba6af41825627cdd4e5a8c52b>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )
     Free unused memory that was cached on the specified device for use with graphs back to the OS.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceSetGraphMemAttribute](<#group__CUDA__GRAPH_1g064bd5c6a773b83d145c281ebf5dbe34>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUgraphMem_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5f76366f87bbdf761007768fe30a57db>)Â attr, void*Â value )
     Set asynchronous allocation attributes related to graphs.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddBatchMemOpNode](<#group__CUDA__GRAPH_1g5acb6914dbd18cb1ae15ea9437a73c96>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams )
     Creates a batch memory operation node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddChildGraphNode](<#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â childGraph )
     Creates a child graph node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddDependencies](<#group__CUDA__GRAPH_1g5dad91f0be4e0fde6092f15797427e2d>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â from, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â to, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_tÂ numDependencies )
     Adds dependency edges to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddEmptyNode](<#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies )
     Creates an empty node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddEventRecordNode](<#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Creates an event record node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddEventWaitNode](<#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Creates an event wait node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddExternalSemaphoresSignalNode](<#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â nodeParams )
     Creates an external semaphore signal node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddExternalSemaphoresWaitNode](<#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â nodeParams )
     Creates an external semaphore wait node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddHostNode](<#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )
     Creates a host execution node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddKernelNode](<#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )
     Creates a kernel execution node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddMemAllocNode](<#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUDA_MEM_ALLOC_NODE_PARAMS](<structCUDA__MEM__ALLOC__NODE__PARAMS__v1.html#structCUDA__MEM__ALLOC__NODE__PARAMS__v1>)*Â nodeParams )
     Creates an allocation node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddMemFreeNode](<#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )
     Creates a memory free node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddMemcpyNode](<#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â copyParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Creates a memcpy node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddMemsetNode](<#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â memsetParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Creates a memset node and adds it to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphAddNode](<#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â dependencyData, size_tÂ numDependencies, [CUgraphNodeParams](<structCUgraphNodeParams.html#structCUgraphNodeParams>)*Â nodeParams )
     Adds a node of arbitrary type to a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphBatchMemOpNodeGetParams](<#group__CUDA__GRAPH_1g1d8039468b71285c61bc03ab3c302a28>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams_out )
     Returns a batch mem op node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphBatchMemOpNodeSetParams](<#group__CUDA__GRAPH_1g625ca946b58df3d17221ff7db5cd7800>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams )
     Sets a batch mem op node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphChildGraphNodeGetGraph](<#group__CUDA__GRAPH_1gbe9fc9267316b3778ef0db507917b4fd>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph )
     Gets a handle to the embedded graph of a child graph node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphClone](<#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraphClone, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â originalGraph )
     Clones a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphConditionalHandleCreate](<#group__CUDA__GRAPH_1gece6f3b9e85d0edb8484d625fe567376>) ( [CUgraphConditionalHandle](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf5f0f00dad6aa27aff480400b77f93ee>)*Â pHandle_out, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx, unsigned int Â defaultLaunchValue, unsigned int Â flags )
     Create a conditional handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphCreate](<#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph, unsigned int Â flags )
     Creates a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphDebugDotPrint](<#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const char*Â path, unsigned int Â flags )
     Write a DOT file describing graph structure.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphDestroy](<#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph )
     Destroys a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphDestroyNode](<#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode )
     Remove a node from the graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphEventRecordNodeGetEvent](<#group__CUDA__GRAPH_1gb3608efc284aa2bbe5db61826d6e2259>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â event_out )
     Returns the event associated with an event record node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphEventRecordNodeSetEvent](<#group__CUDA__GRAPH_1g8ad8006aa7865865bf4d8c475cb21d87>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Sets an event record node's event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphEventWaitNodeGetEvent](<#group__CUDA__GRAPH_1g90b9d60f3f5f4156d1351a96ce92846e>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â event_out )
     Returns the event associated with an event wait node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphEventWaitNodeSetEvent](<#group__CUDA__GRAPH_1g2d6730d63efd399d3000952c54134930>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Sets an event wait node's event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecBatchMemOpNodeSetParams](<#group__CUDA__GRAPH_1g23f51bb4e4c029bb32fac0146e38c076>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams )
     Sets the parameters for a batch mem op node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecChildGraphNodeSetParams](<#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â childGraph )
     Updates node parameters in the child graph node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecDestroy](<#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec )
     Destroys an executable graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecEventRecordNodeSetEvent](<#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Sets the event for an event record node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecEventWaitNodeSetEvent](<#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Sets the event for an event wait node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecExternalSemaphoresSignalNodeSetParams](<#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â nodeParams )
     Sets the parameters for an external semaphore signal node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecExternalSemaphoresWaitNodeSetParams](<#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â nodeParams )
     Sets the parameters for an external semaphore wait node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecGetFlags](<#group__CUDA__GRAPH_1g5004de43ce63398a1a7d7a57edf17d9a>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, cuuint64_t*Â flags )
     Query the instantiation flags of an executable graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecGetId](<#group__CUDA__GRAPH_1g7a561a95ac508d0a99bccbf89aa01509>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, unsigned int*Â graphId )
     Returns the id of a given graph exec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecHostNodeSetParams](<#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )
     Sets the parameters for a host node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecKernelNodeSetParams](<#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )
     Sets the parameters for a kernel node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecMemcpyNodeSetParams](<#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â copyParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Sets the parameters for a memcpy node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecMemsetNodeSetParams](<#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â memsetParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )
     Sets the parameters for a memset node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecNodeSetParams](<#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNodeParams](<structCUgraphNodeParams.html#structCUgraphNodeParams>)*Â nodeParams )
     Update's a graph node's parameters in an instantiated graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExecUpdate](<#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphExecUpdateResultInfo](<structCUgraphExecUpdateResultInfo__v1.html#structCUgraphExecUpdateResultInfo__v1>)*Â resultInfo )
     Check whether an executable graph can be updated with a graph and perform the update if possible.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExternalSemaphoresSignalNodeGetParams](<#group__CUDA__GRAPH_1ga9f9b30ce6eb9f45d691190b20f34126>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â params_out )
     Returns an external semaphore signal node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExternalSemaphoresSignalNodeSetParams](<#group__CUDA__GRAPH_1g7a344ed4c6a5fcaad7bc7c53b04c6099>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â nodeParams )
     Sets an external semaphore signal node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExternalSemaphoresWaitNodeGetParams](<#group__CUDA__GRAPH_1g1430da6d26a58818a4712d135cf37a54>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â params_out )
     Returns an external semaphore wait node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphExternalSemaphoresWaitNodeSetParams](<#group__CUDA__GRAPH_1ge8b93792930a21ec352d6efd2c21c8c0>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â nodeParams )
     Sets an external semaphore wait node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphGetEdges](<#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â from, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â to, [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_t*Â numEdges )
     Returns a graph's dependency edges.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphGetId](<#group__CUDA__GRAPH_1g0f05ae29d14198ff57d722156d60aa41>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, unsigned int*Â graphId )
     Returns the id of a given graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphGetNodes](<#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â nodes, size_t*Â numNodes )
     Returns a graph's nodes.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphGetRootNodes](<#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â rootNodes, size_t*Â numRootNodes )
     Returns a graph's root nodes.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphHostNodeGetParams](<#group__CUDA__GRAPH_1g2e3ea6000089fd5523c197ab5e73d5a2>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )
     Returns a host node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphHostNodeSetParams](<#group__CUDA__GRAPH_1gae021ae8f19ee51044339db9c24dd266>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )
     Sets a host node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphInstantiate](<#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)*Â phGraphExec, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, unsigned long longÂ flags )
     Creates an executable graph from a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphInstantiateWithParams](<#group__CUDA__GRAPH_1g8d9541e4df43ee8440e794634a0d1af8>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)*Â phGraphExec, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUDA_GRAPH_INSTANTIATE_PARAMS](<structCUDA__GRAPH__INSTANTIATE__PARAMS.html#structCUDA__GRAPH__INSTANTIATE__PARAMS>)*Â instantiateParams )
     Creates an executable graph from a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphKernelNodeCopyAttributes](<#group__CUDA__GRAPH_1ga5f4e6786704bf710b61a26146c51c9e>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â dst, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â src )
     Copies attributes from source node to destination node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphKernelNodeGetAttribute](<#group__CUDA__GRAPH_1g9827e34c800e2f2cb4d9a6f4e186f796>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUkernelNodeAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, [CUkernelNodeAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value_out )
     Queries node attribute.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphKernelNodeGetParams](<#group__CUDA__GRAPH_1gb8df3f99e8dd5e4f4a5a0f19a5518252>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )
     Returns a kernel node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphKernelNodeSetAttribute](<#group__CUDA__GRAPH_1gd888774df6c1d0774bee49ec9442eefc>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUkernelNodeAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, const [CUkernelNodeAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value )
     Sets node attribute.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphKernelNodeSetParams](<#group__CUDA__GRAPH_1ga268bf2fd520f5aa3a3d700005df6703>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )
     Sets a kernel node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphLaunch](<#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Launches an executable graph in a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphMemAllocNodeGetParams](<#group__CUDA__GRAPH_1gee2c7d66d3d96b1470c1d1a769f250a2>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_MEM_ALLOC_NODE_PARAMS](<structCUDA__MEM__ALLOC__NODE__PARAMS__v1.html#structCUDA__MEM__ALLOC__NODE__PARAMS__v1>)*Â params_out )
     Returns a memory alloc node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphMemFreeNodeGetParams](<#group__CUDA__GRAPH_1gd24d9fe5769222a2367e3f571fb2f28b>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr_out )
     Returns a memory free node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphMemcpyNodeGetParams](<#group__CUDA__GRAPH_1g572889131dbc31720eff94b130f4005b>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â nodeParams )
     Returns a memcpy node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphMemcpyNodeSetParams](<#group__CUDA__GRAPH_1ga278a7ec0700c86abb0b2cfdf4d3dc1d>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â nodeParams )
     Sets a memcpy node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphMemsetNodeGetParams](<#group__CUDA__GRAPH_1g18830edcfd982f952820a0d7f91b894a>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â nodeParams )
     Returns a memset node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphMemsetNodeSetParams](<#group__CUDA__GRAPH_1gc27f3fd83a6e33c74519066fbaa0de67>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â nodeParams )
     Sets a memset node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeFindInClone](<#group__CUDA__GRAPH_1gf21f6c968e346f028737c1118bfd41c2>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phNode, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hOriginalNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hClonedGraph )
     Finds a cloned version of a node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetContainingGraph](<#group__CUDA__GRAPH_1gbbfe267adf728f1c53aa9d99ba101b92>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph )
     Returns the graph that contains a given graph node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetDependencies](<#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_t*Â numDependencies )
     Returns a node's dependencies.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetDependentNodes](<#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependentNodes, [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_t*Â numDependentNodes )
     Returns a node's dependent nodes.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetEnabled](<#group__CUDA__GRAPH_1g428f51dceec6f6211bb9c1d710925a3d>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned int*Â isEnabled )
     Query whether a node in the given graphExec is enabled.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetLocalId](<#group__CUDA__GRAPH_1g18fd5107a28aaae1e396efcb0edaa70d>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned int*Â nodeId )
     Returns the local node id of a given graph node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetToolsId](<#group__CUDA__GRAPH_1g10d4cf58921a26acce90ed1a03fcd4c1>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned long long*Â toolsNodeId )
     Returns an id used by tools to identify a given node.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeGetType](<#group__CUDA__GRAPH_1gdb1776d97aa1c9d5144774b29e4b8c3e>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNodeType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0731a28f826922120d783d8444e154dc>)*Â type )
     Returns a node's type.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeSetEnabled](<#group__CUDA__GRAPH_1g371b20eb0c0658731e38db7e68f12c78>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned int Â isEnabled )
     Enables or disables the specified node in the given graphExec.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphNodeSetParams](<#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139>) ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNodeParams](<structCUgraphNodeParams.html#structCUgraphNodeParams>)*Â nodeParams )
     Update's a graph node's parameters.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphReleaseUserObject](<#group__CUDA__GRAPH_1g232c84cc31e13e4201a421e28561eebf>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â graph, [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count )
     Release a user object reference from a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphRemoveDependencies](<#group__CUDA__GRAPH_1g25048b696f56b4d6131f068074176301>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â from, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â to, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_tÂ numDependencies )
     Removes dependency edges from a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphRetainUserObject](<#group__CUDA__GRAPH_1gaffd130c928e56740a2a5aaeb6125c8a>) ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â graph, [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count, unsigned int Â flags )
     Retain a reference to a user object from a graph.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGraphUpload](<#group__CUDA__GRAPH_1ga7eb9849e6e4604864a482b38f25be48>) ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Uploads an executable graph in a stream.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuUserObjectCreate](<#group__CUDA__GRAPH_1g58f04e0ac0ad23d2f15ea6e9f6c8a999>) ( [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)*Â object_out, void*Â ptr, [CUhostFn](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g262cd3570ff5d396db4e3dabede3c355>)Â destroy, unsigned int Â initialRefcount, unsigned int Â flags )
     Create a user object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuUserObjectRelease](<#group__CUDA__GRAPH_1ga2c16918341b8d020c9246e75658cc80>) ( [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count )
     Release a reference to a user object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuUserObjectRetain](<#group__CUDA__GRAPH_1ge022bcecdeca2d14cc8f28afc6a2eaf6>) ( [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count )
     Retain a reference to a user object.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetGraphMemAttribute ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUgraphMem_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5f76366f87bbdf761007768fe30a57db>)Â attr, void*Â value )


Query asynchronous allocation attributes related to graphs.

######  Parameters

`device`
    \- Specifies the scope of the query
`attr`
    \- attribute to get
`value`
    \- retrieved value

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Valid attributes are:

  * [CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5f76366f87bbdf761007768fe30a57dbd64476ce9c5839854a7a82cee4b882af>): Amount of memory, in bytes, currently associated with graphs

  * [CU_GRAPH_MEM_ATTR_USED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5f76366f87bbdf761007768fe30a57db21e8caa067ac3b5264197b4d445575ce>): High watermark of memory, in bytes, associated with graphs since the last time it was reset. High watermark can only be reset to zero.

  * [CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5f76366f87bbdf761007768fe30a57db365a64945f1af3af1b3e50aca699cf55>): Amount of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.

  * [CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5f76366f87bbdf761007768fe30a57dbef25946b479b24908620814513f6acd4>): High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.


**See also:**

[cuDeviceSetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g064bd5c6a773b83d145c281ebf5dbe34> "Set asynchronous allocation attributes related to graphs."), [cuGraphAddMemAllocNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9> "Creates an allocation node and adds it to a graph."), [cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGraphMemTrim ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )


Free unused memory that was cached on the specified device for use with graphs back to the OS.

######  Parameters

`device`
    \- The device for which cached memory should be freed.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Blocks which are not in use by a graph that is either currently executing or scheduled to execute are freed back to the operating system.

**See also:**

[cuGraphAddMemAllocNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9> "Creates an allocation node and adds it to a graph."), [cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph."), [cuDeviceSetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g064bd5c6a773b83d145c281ebf5dbe34> "Set asynchronous allocation attributes related to graphs."), [cuDeviceGetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g359903c2447ac22b4e1a0dce26adfef5> "Query asynchronous allocation attributes related to graphs.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceSetGraphMemAttribute ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUgraphMem_attribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5f76366f87bbdf761007768fe30a57db>)Â attr, void*Â value )


Set asynchronous allocation attributes related to graphs.

######  Parameters

`device`
    \- Specifies the scope of the query
`attr`
    \- attribute to get
`value`
    \- pointer to value to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Valid attributes are:

  * [CU_GRAPH_MEM_ATTR_USED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5f76366f87bbdf761007768fe30a57db21e8caa067ac3b5264197b4d445575ce>): High watermark of memory, in bytes, associated with graphs since the last time it was reset. High watermark can only be reset to zero.

  * [CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5f76366f87bbdf761007768fe30a57dbef25946b479b24908620814513f6acd4>): High watermark of memory, in bytes, currently allocated for use by the CUDA graphs asynchronous allocator.


**See also:**

[cuDeviceGetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g359903c2447ac22b4e1a0dce26adfef5> "Query asynchronous allocation attributes related to graphs."), [cuGraphAddMemAllocNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9> "Creates an allocation node and adds it to a graph."), [cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddBatchMemOpNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams )


Creates a batch memory operation node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Parameters for the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new batch memory operation node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

When the node is added, the paramArray inside `nodeParams` is copied and therefore it can be freed after the call returns.

Note:

Warning: Improper use of this API may deadlock the application. Synchronization ordering established through this API is not visible to CUDA. CUDA tasks that are (even indirectly) ordered by this API should also have that order expressed with CUDA-visible dependencies such as events. This ensures that the scheduler does not serialize them in an improper order.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuStreamBatchMemOp](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g764c442de9b671f9dec856e8ae531ed1> "Batch operations to synchronize the stream via memory operations."), [cuStreamWaitValue32](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g629856339de7bc6606047385addbb398> "Wait on a memory location."), [cuStreamWriteValue32](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g091455366d56dc2f1f69726aafa369b0> "Write a value to memory."), [cuStreamWaitValue64](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g6910c1258c5f15aa5d699f0fd60d6933> "Wait on a memory location."), [cuStreamWriteValue64](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1gc8af1e8b96d7561840affd5217dd6830> "Write a value to memory."), [cuGraphBatchMemOpNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g1d8039468b71285c61bc03ab3c302a28> "Returns a batch mem op node's parameters."), [cuGraphBatchMemOpNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g625ca946b58df3d17221ff7db5cd7800> "Sets a batch mem op node's parameters."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddChildGraphNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â childGraph )


Creates a child graph node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`childGraph`
    \- The graph to clone into this node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Creates a new node which executes an embedded graph, and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

If `childGraph` contains allocation nodes, free nodes, or conditional nodes, this call will return an error.

The node executes an embedded child graph. The child graph is cloned in this call.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphChildGraphNodeGetGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbe9fc9267316b3778ef0db507917b4fd> "Gets a handle to the embedded graph of a child graph node."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph."), [cuGraphClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e> "Clones a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddDependencies ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â from, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â to, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_tÂ numDependencies )


Adds dependency edges to a graph.

######  Parameters

`hGraph`
    \- Graph to which dependencies are added
`from`
    \- Array of nodes that provide the dependencies
`to`
    \- Array of dependent nodes
`edgeData`
    \- Optional array of edge data. If NULL, default (zeroed) edge data is assumed.
`numDependencies`
    \- Number of dependencies to be added

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

The number of dependencies to be added is defined by `numDependencies` Elements in `from` and `to` at corresponding indices define a dependency. Each node in `from` and `to` must belong to `hGraph`.

If `numDependencies` is 0, elements in `from` and `to` will be ignored. Specifying an existing dependency will return an error.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphRemoveDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g25048b696f56b4d6131f068074176301> "Removes dependency edges from a graph."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphNodeGetDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9> "Returns a node's dependencies."), [cuGraphNodeGetDependentNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450> "Returns a node's dependent nodes.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddEmptyNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies )


Creates an empty node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Creates a new node which performs no operation, and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

An empty node performs no operation during execution, but can be used for transitive ordering. For example, a phased execution graph with 2 groups of n nodes with a barrier between them can be represented using an empty node and 2*n dependency edges, rather than no empty node and n^2 dependency edges.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddEventRecordNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Creates an event record node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`event`
    \- Event for the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new event record node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and event specified in `event`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

Each launch of the graph will record `event` to capture execution of the node's dependencies.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddEventWaitNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Creates an event wait node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`event`
    \- Event for the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new event wait node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and event specified in `event`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

The graph node will wait for all work captured in `event`. See [cuEventRecord()](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event.") for details on what is captured by an event. `event` may be from a different context or device than the launch stream.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddExternalSemaphoresSignalNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â nodeParams )


Creates an external semaphore signal node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Parameters for the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new external semaphore signal node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

Performs a signal operation on a set of externally allocated semaphore objects when the node is launched. The operation(s) will occur after all of the node's dependencies have completed.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphExternalSemaphoresSignalNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga9f9b30ce6eb9f45d691190b20f34126> "Returns an external semaphore signal node's parameters."), [cuGraphExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a344ed4c6a5fcaad7bc7c53b04c6099> "Sets an external semaphore signal node's parameters."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddExternalSemaphoresWaitNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â nodeParams )


Creates an external semaphore wait node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Parameters for the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new external semaphore wait node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

Performs a wait operation on a set of externally allocated semaphore objects when the node is launched. The node's dependencies will not be launched until the wait operation has completed.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphExternalSemaphoresWaitNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g1430da6d26a58818a4712d135cf37a54> "Returns an external semaphore wait node's parameters."), [cuGraphExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge8b93792930a21ec352d6efd2c21c8c0> "Sets an external semaphore wait node's parameters."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphAddExternalSemaphoresSignalNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3> "Creates an external semaphore signal node and adds it to a graph."), [cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddHostNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )


Creates a host execution node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Parameters for the host node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new CPU execution node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

When the graph is launched, the node will invoke the specified CPU function. Host nodes are not supported under MPS with pre-Volta GPUs.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuLaunchHostFunc](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f> "Enqueues a host function call in a stream."), [cuGraphHostNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2e3ea6000089fd5523c197ab5e73d5a2> "Returns a host node's parameters."), [cuGraphHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gae021ae8f19ee51044339db9c24dd266> "Sets a host node's parameters."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddKernelNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )


Creates a kernel execution node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Parameters for the GPU execution node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new kernel execution node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

The CUDA_KERNEL_NODE_PARAMS structure is defined as:


    â  typedef struct CUDA_KERNEL_NODE_PARAMS_st {
                [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>) func;
                unsigned int gridDimX;
                unsigned int gridDimY;
                unsigned int gridDimZ;
                unsigned int blockDimX;
                unsigned int blockDimY;
                unsigned int blockDimZ;
                unsigned int sharedMemBytes;
                void **kernelParams;
                void **extra;
                [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) kern;
                [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) ctx;
            } [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>);

When the graph is launched, the node will invoke kernel `func` on a (`gridDimX` x `gridDimY` x `gridDimZ`) grid of blocks. Each block contains (`blockDimX` x `blockDimY` x `blockDimZ`) threads.

`sharedMemBytes` sets the amount of dynamic shared memory that will be available to each thread block.

Kernel parameters to `func` can be specified in one of two ways:

1) Kernel parameters can be specified via `kernelParams`. If the kernel has N parameters, then `kernelParams` needs to be an array of N pointers. Each pointer, from `kernelParams`[0] to `kernelParams`[N-1], points to the region of memory from which the actual parameter will be copied. The number of kernel parameters and their offsets and sizes do not need to be specified as that information is retrieved directly from the kernel's image.

2) Kernel parameters for non-cooperative kernels can also be packaged by the application into a single buffer that is passed in via `extra`. This places the burden on the application of knowing each kernel parameter's size and alignment/padding within the buffer. The `extra` parameter exists to allow this function to take additional less commonly used arguments. `extra` specifies a list of names of extra settings and their corresponding values. Each extra setting name is immediately followed by the corresponding value. The list must be terminated with either NULL or CU_LAUNCH_PARAM_END.

  * [CU_LAUNCH_PARAM_END](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd5c11cff5adfa5a69d66829399653532>), which indicates the end of the `extra` array;

  * [CU_LAUNCH_PARAM_BUFFER_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g36d10d0b40c51372877578a2cffd6acd>), which specifies that the next value in `extra` will be a pointer to a buffer containing all the kernel parameters for launching kernel `func`;

  * [CU_LAUNCH_PARAM_BUFFER_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf339c057cd94562ead93a192e11c17e9>), which specifies that the next value in `extra` will be a pointer to a size_t containing the size of the buffer specified with [CU_LAUNCH_PARAM_BUFFER_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g36d10d0b40c51372877578a2cffd6acd>);


The error [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned if kernel parameters are specified with both `kernelParams` and `extra` (i.e. both `kernelParams` and `extra` are non-NULL). [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned if `extra` is used for a cooperative kernel.

The `kernelParams` or `extra` array, as well as the argument values it points to, are copied during this call.

Note:

Kernels launched using graphs must not use texture and surface references. Reading or writing through any texture or surface reference is undefined behavior. This restriction does not apply to texture and surface objects.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuLaunchCooperativeKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g06d753134145c4584c0c62525c1894cb> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel where thread blocks can cooperate and synchronize as they execute."), [cuGraphKernelNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb8df3f99e8dd5e4f4a5a0f19a5518252> "Returns a kernel node's parameters."), [cuGraphKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga268bf2fd520f5aa3a3d700005df6703> "Sets a kernel node's parameters."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddMemAllocNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUDA_MEM_ALLOC_NODE_PARAMS](<structCUDA__MEM__ALLOC__NODE__PARAMS__v1.html#structCUDA__MEM__ALLOC__NODE__PARAMS__v1>)*Â nodeParams )


Creates an allocation node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Parameters for the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new allocation node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

When [cuGraphAddMemAllocNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9> "Creates an allocation node and adds it to a graph.") creates an allocation node, it returns the address of the allocation in `nodeParams.dptr`. The allocation's address remains fixed across instantiations and launches.

If the allocation is freed in the same graph, by creating a free node using [cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph."), the allocation can be accessed by nodes ordered after the allocation node but before the free node. These allocations cannot be freed outside the owning graph, and they can only be freed once in the owning graph.

If the allocation is not freed in the same graph, then it can be accessed not only by nodes in the graph which are ordered after the allocation node, but also by stream operations ordered after the graph's execution but before the allocation is freed.

Allocations which are not freed in the same graph can be freed by:

  * passing the allocation to [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics.") or [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory.");

  * launching a graph with a free node for that allocation; or

  * specifying [CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49561684f715bf05e39afd69aa508299a479>) during instantiation, which makes each launch behave as though it called [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics.") for every unfreed allocation.


It is not possible to free an allocation in both the owning graph and another graph. If the allocation is freed in the same graph, a free node cannot be added to another graph. If the allocation is freed in another graph, a free node can no longer be added to the owning graph.

The following restrictions apply to graphs which contain allocation and/or memory free nodes:

  * Nodes and edges of the graph cannot be deleted.

  * The graph can only be used in a child node if the ownership is moved to the parent.

  * Only one instantiation of the graph may exist at any point in time.

  * The graph cannot be cloned.


Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph."), [cuGraphMemAllocNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gee2c7d66d3d96b1470c1d1a769f250a2> "Returns a memory alloc node's parameters."), [cuDeviceGraphMemTrim](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g57c87f4ba6af41825627cdd4e5a8c52b> "Free unused memory that was cached on the specified device for use with graphs back to the OS."), [cuDeviceGetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g359903c2447ac22b4e1a0dce26adfef5> "Query asynchronous allocation attributes related to graphs."), [cuDeviceSetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g064bd5c6a773b83d145c281ebf5dbe34> "Set asynchronous allocation attributes related to graphs."), [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphAddExternalSemaphoresSignalNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3> "Creates an external semaphore signal node and adds it to a graph."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddMemFreeNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )


Creates a memory free node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`dptr`
    \- Address of memory to free

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new memory free node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies` and arguments specified in `nodeParams`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

[cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph.") will return [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if the user attempts to free:

  * an allocation twice in the same graph.

  * an address that was not returned by an allocation node.

  * an invalid address.


The following restrictions apply to graphs which contain allocation and/or memory free nodes:

  * Nodes and edges of the graph cannot be deleted.

  * The graph can only be used in a child node if the ownership is moved to the parent.

  * Only one instantiation of the graph may exist at any point in time.

  * The graph cannot be cloned.


Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphAddMemAllocNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9> "Creates an allocation node and adds it to a graph."), [cuGraphMemFreeNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd24d9fe5769222a2367e3f571fb2f28b> "Returns a memory free node's parameters."), [cuDeviceGraphMemTrim](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g57c87f4ba6af41825627cdd4e5a8c52b> "Free unused memory that was cached on the specified device for use with graphs back to the OS."), [cuDeviceGetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g359903c2447ac22b4e1a0dce26adfef5> "Query asynchronous allocation attributes related to graphs."), [cuDeviceSetGraphMemAttribute](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g064bd5c6a773b83d145c281ebf5dbe34> "Set asynchronous allocation attributes related to graphs."), [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphAddExternalSemaphoresSignalNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3> "Creates an external semaphore signal node and adds it to a graph."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddMemcpyNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â copyParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Creates a memcpy node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`copyParams`
    \- Parameters for the memory copy
`ctx`
    \- Context on which to run the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a new memcpy node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

When the graph is launched, the node will perform the memcpy described by `copyParams`. See [cuMemcpy3D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays.") for a description of the structure and its restrictions.

Memcpy nodes have some additional restrictions with regards to managed memory, if the system contains at least one device which has a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). If one or more of the operands refer to managed memory, then using the memory type [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) is disallowed for those operand(s). The managed memory will be treated as residing on either the host or the device, depending on which memory type is specified.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuGraphMemcpyNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g572889131dbc31720eff94b130f4005b> "Returns a memcpy node's parameters."), [cuGraphMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga278a7ec0700c86abb0b2cfdf4d3dc1d> "Sets a memcpy node's parameters."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddMemsetNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, size_tÂ numDependencies, const [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â memsetParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Creates a memset node and adds it to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`numDependencies`
    \- Number of dependencies
`memsetParams`
    \- Parameters for the memory set
`ctx`
    \- Context on which to run the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### Description

Creates a new memset node and adds it to `hGraph` with `numDependencies` dependencies specified via `dependencies`. It is possible for `numDependencies` to be 0, in which case the node will be placed at the root of the graph. `dependencies` may not have any duplicate entries. A handle to the new node will be returned in `phGraphNode`.

The element size must be 1, 2, or 4 bytes. When the graph is launched, the node will perform the memset described by `memsetParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuGraphMemsetNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18830edcfd982f952820a0d7f91b894a> "Returns a memset node's parameters."), [cuGraphMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gc27f3fd83a6e33c74519066fbaa0de67> "Sets a memset node's parameters."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphDestroyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9> "Remove a node from the graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphAddNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phGraphNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â dependencyData, size_tÂ numDependencies, [CUgraphNodeParams](<structCUgraphNodeParams.html#structCUgraphNodeParams>)*Â nodeParams )


Adds a node of arbitrary type to a graph.

######  Parameters

`phGraphNode`
    \- Returns newly created node
`hGraph`
    \- Graph to which to add the node
`dependencies`
    \- Dependencies of the node
`dependencyData`
    \- Optional edge data for the dependencies. If NULL, the data is assumed to be default (zeroed) for all dependencies.
`numDependencies`
    \- Number of dependencies
`nodeParams`
    \- Specification of the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Creates a new node in `hGraph` described by `nodeParams` with `numDependencies` dependencies specified via `dependencies`. `numDependencies` may be 0. `dependencies` may be null if `numDependencies` is 0. `dependencies` may not have any duplicate entries.

`nodeParams` is a tagged union. The node type should be specified in the `type` field, and type-specific parameters in the corresponding union member. All unused bytes - that is, `reserved0` and all bytes past the utilized union member - must be set to zero. It is recommended to use brace initialization or memset to ensure all bytes are initialized.

Note that for some node types, `nodeParams` may contain "out parameters" which are modified during the call, such as `nodeParams->alloc.dptr`.

A handle to the new node will be returned in `phGraphNode`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphBatchMemOpNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams_out )


Returns a batch mem op node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`nodeParams_out`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of batch mem op node `hNode` in `nodeParams_out`. The `paramArray` returned in `nodeParams_out` is owned by the node. This memory remains valid until the node is destroyed or its parameters are modified, and should not be modified directly. Use [cuGraphBatchMemOpNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g625ca946b58df3d17221ff7db5cd7800> "Sets a batch mem op node's parameters.") to update the parameters of this node.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuStreamBatchMemOp](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g764c442de9b671f9dec856e8ae531ed1> "Batch operations to synchronize the stream via memory operations."), [cuGraphAddBatchMemOpNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5acb6914dbd18cb1ae15ea9437a73c96> "Creates a batch memory operation node and adds it to a graph."), [cuGraphBatchMemOpNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g625ca946b58df3d17221ff7db5cd7800> "Sets a batch mem op node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphBatchMemOpNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams )


Sets a batch mem op node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Sets the parameters of batch mem op node `hNode` to `nodeParams`.

The paramArray inside `nodeParams` is copied and therefore it can be freed after the call returns.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuStreamBatchMemOp](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g764c442de9b671f9dec856e8ae531ed1> "Batch operations to synchronize the stream via memory operations."), [cuGraphAddBatchMemOpNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5acb6914dbd18cb1ae15ea9437a73c96> "Creates a batch memory operation node and adds it to a graph."), [cuGraphBatchMemOpNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g1d8039468b71285c61bc03ab3c302a28> "Returns a batch mem op node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphChildGraphNodeGetGraph ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph )


Gets a handle to the embedded graph of a child graph node.

######  Parameters

`hNode`
    \- Node to get the embedded graph for
`phGraph`
    \- Location to store a handle to the graph

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Gets a handle to the embedded graph in a child graph node. This call does not clone the graph. Changes to the graph will be reflected in the node, and the node retains ownership of the graph.

Allocation and free nodes cannot be added to the returned graph. Attempting to do so will return an error.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphNodeFindInClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf21f6c968e346f028737c1118bfd41c2> "Finds a cloned version of a node.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphClone ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraphClone, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â originalGraph )


Clones a graph.

######  Parameters

`phGraphClone`
    \- Returns newly created cloned graph
`originalGraph`
    \- Graph to clone

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

This function creates a copy of `originalGraph` and returns it in `phGraphClone`. All parameters are copied into the cloned graph. The original graph may be modified after this call without affecting the clone.

Child graph nodes in the original graph are recursively copied into the clone.

Note:

: Cloning is not supported for graphs which contain memory allocation nodes, memory free nodes, or conditional nodes.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphNodeFindInClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf21f6c968e346f028737c1118bfd41c2> "Finds a cloned version of a node.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphConditionalHandleCreate ( [CUgraphConditionalHandle](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf5f0f00dad6aa27aff480400b77f93ee>)*Â pHandle_out, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx, unsigned int Â defaultLaunchValue, unsigned int Â flags )


Create a conditional handle.

######  Parameters

`pHandle_out`
    \- Pointer used to return the handle to the caller.
`hGraph`
    \- Graph which will contain the conditional node using this handle.
`ctx`
    \- Context for the handle and associated conditional node.
`defaultLaunchValue`
    \- Optional initial value for the conditional variable. Applied at the beginning of each graph execution if CU_GRAPH_COND_ASSIGN_DEFAULT is set in `flags`.
`flags`
    \- Currently must be CU_GRAPH_COND_ASSIGN_DEFAULT or 0.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Creates a conditional handle associated with `hGraph`.

The conditional handle must be associated with a conditional node in this graph or one of its children.

Handles not associated with a conditional node may cause graph instantiation to fail.

Handles can only be set from the context with which they are associated.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphCreate ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph, unsigned int Â flags )


Creates a graph.

######  Parameters

`phGraph`
    \- Returns newly created graph
`flags`
    \- Graph creation flags, must be 0

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Creates an empty graph, which is returned via `phGraph`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8> "Destroys a graph."), [cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphGetRootNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd> "Returns a graph's root nodes."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e> "Clones a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphDebugDotPrint ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const char*Â path, unsigned int Â flags )


Write a DOT file describing graph structure.

######  Parameters

`hGraph`
    \- The graph to create a DOT file from
`path`
    \- The path to write the DOT file to
`flags`
    \- Flags from CUgraphDebugDot_flags for specifying which additional node information to write

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OPERATING_SYSTEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c849a151611f6e2ed1b3ae923f79ef3c>)

###### Description

Using the provided `hGraph`, write to `path` a DOT formatted description of the graph. By default this includes the graph topology, node types, node id, kernel names and memcpy direction. `flags` can be specified to write more detailed information about each node type such as parameter values, kernel attributes, node and function handles.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphDestroy ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph )


Destroys a graph.

######  Parameters

`hGraph`
    \- Graph to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Destroys the graph specified by `hGraph`, as well as all of its nodes.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphDestroyNode ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode )


Remove a node from the graph.

######  Parameters

`hNode`
    \- Node to remove

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Removes `hNode` from its graph. This operation also severs any dependencies of other nodes on `hNode` and vice versa.

Nodes which belong to a graph which contains allocation or free nodes cannot be destroyed. Any attempt to do so will return an error.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphAddEmptyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g14b625984430cb2d574c63f29c9b9223> "Creates an empty node and adds it to a graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphEventRecordNodeGetEvent ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â event_out )


Returns the event associated with an event record node.

######  Parameters

`hNode`
    \- Node to get the event for
`event_out`
    \- Pointer to return the event

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the event of event record node `hNode` in `event_out`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8ad8006aa7865865bf4d8c475cb21d87> "Sets an event record node's event."), [cuGraphEventWaitNodeGetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g90b9d60f3f5f4156d1351a96ce92846e> "Returns the event associated with an event wait node."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphEventRecordNodeSetEvent ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Sets an event record node's event.

######  Parameters

`hNode`
    \- Node to set the event for
`event`
    \- Event to use

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Sets the event of event record node `hNode` to `event`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphEventRecordNodeGetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb3608efc284aa2bbe5db61826d6e2259> "Returns the event associated with an event record node."), [cuGraphEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2d6730d63efd399d3000952c54134930> "Sets an event wait node's event."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphEventWaitNodeGetEvent ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â event_out )


Returns the event associated with an event wait node.

######  Parameters

`hNode`
    \- Node to get the event for
`event_out`
    \- Pointer to return the event

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the event of event wait node `hNode` in `event_out`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2d6730d63efd399d3000952c54134930> "Sets an event wait node's event."), [cuGraphEventRecordNodeGetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb3608efc284aa2bbe5db61826d6e2259> "Returns the event associated with an event record node."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphEventWaitNodeSetEvent ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Sets an event wait node's event.

######  Parameters

`hNode`
    \- Node to set the event for
`event`
    \- Event to use

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Sets the event of event wait node `hNode` to `event`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphEventWaitNodeGetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g90b9d60f3f5f4156d1351a96ce92846e> "Returns the event associated with an event wait node."), [cuGraphEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8ad8006aa7865865bf4d8c475cb21d87> "Sets an event record node's event."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecBatchMemOpNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_BATCH_MEM_OP_NODE_PARAMS](<structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1.html#structCUDA__BATCH__MEM__OP__NODE__PARAMS__v1>)*Â nodeParams )


Sets the parameters for a batch mem op node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Batch mem op node from the graph from which graphExec was instantiated
`nodeParams`
    \- Updated Parameters to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the parameters of a batch mem op node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

The following fields on operations may be modified on an executable graph:

op.waitValue.address op.waitValue.value[64] op.waitValue.flags bits corresponding to wait type (i.e. CU_STREAM_WAIT_VALUE_FLUSH bit cannot be modified) op.writeValue.address op.writeValue.value[64]

Other fields, such as the context, count or type of operations, and other types of operations such as membars, may not be modified.

`hNode` must not have been removed from the original graph.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

The paramArray inside `nodeParams` is copied and therefore it can be freed after the call returns.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuStreamBatchMemOp](<group__CUDA__MEMOP.html#group__CUDA__MEMOP_1g764c442de9b671f9dec856e8ae531ed1> "Batch operations to synchronize the stream via memory operations."), [cuGraphAddBatchMemOpNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5acb6914dbd18cb1ae15ea9437a73c96> "Creates a batch memory operation node and adds it to a graph."), [cuGraphBatchMemOpNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g1d8039468b71285c61bc03ab3c302a28> "Returns a batch mem op node's parameters."), [cuGraphBatchMemOpNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g625ca946b58df3d17221ff7db5cd7800> "Sets a batch mem op node's parameters."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecChildGraphNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â childGraph )


Updates node parameters in the child graph node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Host node from the graph which was used to instantiate graphExec
`childGraph`
    \- The graph supplying the updated parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Updates the work represented by `hNode` in `hGraphExec` as though the nodes contained in `hNode's` graph had the parameters contained in `childGraph's` nodes at instantiation. `hNode` must remain in the graph which was used to instantiate `hGraphExec`. Changed edges to and from `hNode` are ignored.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

The topology of `childGraph`, as well as the node insertion order, must match that of the graph contained in `hNode`. See [cuGraphExecUpdate()](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible.") for a list of restrictions on what can be updated in an instantiated graph. The update is recursive, so child graph nodes contained within the top level child graph will also be updated.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddChildGraphNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gde52afbcf91a8c79d4d7efbe0e3b6844> "Creates a child graph node and adds it to a graph."), [cuGraphChildGraphNodeGetGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbe9fc9267316b3778ef0db507917b4fd> "Gets a handle to the embedded graph of a child graph node."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecDestroy ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec )


Destroys an executable graph.

######  Parameters

`hGraphExec`
    \- Executable graph to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Destroys the executable graph specified by `hGraphExec`, as well as all of its executable nodes. If the executable graph is in-flight, it will not be terminated, but rather freed asynchronously on completion.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphUpload](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7eb9849e6e4604864a482b38f25be48> "Uploads an executable graph in a stream."), [cuGraphLaunch](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471> "Launches an executable graph in a stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecEventRecordNodeSetEvent ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Sets the event for an event record node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- event record node from the graph from which graphExec was instantiated
`event`
    \- Updated event to use

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the event of an event record node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddEventRecordNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7f6dcb109f4b7470ce6b067d39974a4> "Creates an event record node and adds it to a graph."), [cuGraphEventRecordNodeGetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb3608efc284aa2bbe5db61826d6e2259> "Returns the event associated with an event record node."), [cuGraphEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2d6730d63efd399d3000952c54134930> "Sets an event wait node's event."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecEventWaitNodeSetEvent ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Sets the event for an event wait node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- event wait node from the graph from which graphExec was instantiated
`event`
    \- Updated event to use

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the event of an event wait node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddEventWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7306f3bcbec3406d80e110cd13405c5e> "Creates an event wait node and adds it to a graph."), [cuGraphEventWaitNodeGetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g90b9d60f3f5f4156d1351a96ce92846e> "Returns the event associated with an event wait node."), [cuGraphEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8ad8006aa7865865bf4d8c475cb21d87> "Sets an event record node's event."), [cuEventRecordWithFlags](<group__CUDA__EVENT.html#group__CUDA__EVENT_1ge577e0c132d9c4961f220d79f6762c4b> "Records an event."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecExternalSemaphoresSignalNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â nodeParams )


Sets the parameters for an external semaphore signal node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- semaphore signal node from the graph from which graphExec was instantiated
`nodeParams`
    \- Updated Parameters to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the parameters of an external semaphore signal node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

`hNode` must not have been removed from the original graph.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

Changing `nodeParams->numExtSems` is not supported.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddExternalSemaphoresSignalNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3> "Creates an external semaphore signal node and adds it to a graph."), [cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecExternalSemaphoresWaitNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â nodeParams )


Sets the parameters for an external semaphore wait node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- semaphore wait node from the graph from which graphExec was instantiated
`nodeParams`
    \- Updated Parameters to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the parameters of an external semaphore wait node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

`hNode` must not have been removed from the original graph.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

Changing `nodeParams->numExtSems` is not supported.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuImportExternalSemaphore](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1ge593134f5f9650474af74db644c4a326> "Imports an external semaphore."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecGetFlags ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, cuuint64_t*Â flags )


Query the instantiation flags of an executable graph.

######  Parameters

`hGraphExec`
    \- The executable graph to query
`flags`
    \- Returns the instantiation flags

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns the flags that were passed to instantiation for the given executable graph. [CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49569557f13a16fe73b147fb4c9018e92925>) will not be returned by this API as it does not affect the resulting executable graph.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphInstantiateWithParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8d9541e4df43ee8440e794634a0d1af8> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecGetId ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, unsigned int*Â graphId )


Returns the id of a given graph exec.

######  Parameters

`hGraphExec`
    \- Graph to query
`graphId`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the id of `hGraphExec` in `*graphId`. The value in `*graphId` will match that referenced by [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.").

**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.")[cuGraphNodeGetContainingGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbbfe267adf728f1c53aa9d99ba101b92> "Returns the graph that contains a given graph node.")[cuGraphNodeGetLocalId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18fd5107a28aaae1e396efcb0edaa70d> "Returns the local node id of a given graph node.")[cuGraphNodeGetToolsId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g10d4cf58921a26acce90ed1a03fcd4c1> "Returns an id used by tools to identify a given node.")[cuGraphGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0f05ae29d14198ff57d722156d60aa41> "Returns the id of a given graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecHostNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )


Sets the parameters for a host node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Host node from the graph which was used to instantiate graphExec
`nodeParams`
    \- The updated parameters to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Updates the work represented by `hNode` in `hGraphExec` as though `hNode` had contained `nodeParams` at instantiation. hNode must remain in the graph which was used to instantiate `hGraphExec`. Changed edges to and from hNode are ignored.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. hNode is also not modified by this call.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gae021ae8f19ee51044339db9c24dd266> "Sets a host node's parameters."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecKernelNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )


Sets the parameters for a kernel node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- kernel node from the graph from which graphExec was instantiated
`nodeParams`
    \- Updated Parameters to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the parameters of a kernel node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

`hNode` must not have been removed from the original graph. All `nodeParams` fields may change, but the following restrictions apply to `func` updates:

  * The owning context of the function cannot change.

  * A node whose function originally did not use CUDA dynamic parallelism cannot be updated to a function which uses CDP

  * A node whose function originally did not make device-side update calls cannot be updated to a function which makes device-side update calls.

  * If `hGraphExec` was not instantiated for device launch, a node whose function originally did not use device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) cannot be updated to a function which uses device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) unless the node resides on the same context as nodes which contained such calls at instantiate-time. If no such calls were present at instantiation, these updates cannot be performed at all.


The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

If `hNode` is a device-updatable kernel node, the next upload/launch of `hGraphExec` will overwrite any previous device-side updates. Additionally, applying host updates to a device-updatable kernel node while it is being updated from the device will result in undefined behavior.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga268bf2fd520f5aa3a3d700005df6703> "Sets a kernel node's parameters."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecMemcpyNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â copyParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Sets the parameters for a memcpy node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Memcpy node from the graph which was used to instantiate graphExec
`copyParams`
    \- The updated parameters to set
`ctx`
    \- Context on which to run the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Updates the work represented by `hNode` in `hGraphExec` as though `hNode` had contained `copyParams` at instantiation. hNode must remain in the graph which was used to instantiate `hGraphExec`. Changed edges to and from hNode are ignored.

The source and destination memory in `copyParams` must be allocated from the same contexts as the original source and destination memory. Both the instantiation-time memory operands and the memory operands in `copyParams` must be 1-dimensional. Zero-length operations are not supported.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. hNode is also not modified by this call.

Returns CUDA_ERROR_INVALID_VALUE if the memory operands' mappings changed or either the original or new memory operands are multidimensional.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga278a7ec0700c86abb0b2cfdf4d3dc1d> "Sets a memcpy node's parameters."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecMemsetNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â memsetParams, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â ctx )


Sets the parameters for a memset node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Memset node from the graph which was used to instantiate graphExec
`memsetParams`
    \- The updated parameters to set
`ctx`
    \- Context on which to run the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Updates the work represented by `hNode` in `hGraphExec` as though `hNode` had contained `memsetParams` at instantiation. hNode must remain in the graph which was used to instantiate `hGraphExec`. Changed edges to and from hNode are ignored.

Zero sized operations are not supported.

The new destination pointer in memsetParams must be to the same kind of allocation as the original destination pointer and have the same context association and device mapping as the original destination pointer.

Both the value and pointer address may be updated. Changing other aspects of the memset (width, height, element size or pitch) may cause the update to be rejected. Specifically, for 2d memsets, all dimension changes are rejected. For 1d memsets, changes in height are explicitly rejected and other changes are opportunistically allowed if the resulting work maps onto the work resources already allocated for the node.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. hNode is also not modified by this call.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph."), [cuGraphMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gc27f3fd83a6e33c74519066fbaa0de67> "Sets a memset node's parameters."), [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec."), [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec."), [cuGraphExecHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga549b946cedb73dc2596314b2d52f8d8> "Sets the parameters for a host node in the given graphExec."), [cuGraphExecChildGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8f2d9893f6b899f992db1a2942ec03ff> "Updates node parameters in the child graph node in the given graphExec."), [cuGraphExecEventRecordNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g62fea841fdc169c3ef18e9199f28a6a7> "Sets the event for an event record node in the given graphExec."), [cuGraphExecEventWaitNodeSetEvent](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfea9619d6ff228401613febae793f996> "Sets the event for an event wait node in the given graphExec."), [cuGraphExecExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96aedf2977d0dce275fa3b3cf3700ade> "Sets the parameters for an external semaphore signal node in the given graphExec."), [cuGraphExecExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g98a93c41b057cc1b48c0498811f65ad3> "Sets the parameters for an external semaphore wait node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecNodeSetParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNodeParams](<structCUgraphNodeParams.html#structCUgraphNodeParams>)*Â nodeParams )


Update's a graph node's parameters in an instantiated graph.

######  Parameters

`hGraphExec`
    \- The executable graph in which to update the specified node
`hNode`
    \- Corresponding node from the graph from which graphExec was instantiated
`nodeParams`
    \- Updated Parameters to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Sets the parameters of a node in an executable graph `hGraphExec`. The node is identified by the corresponding node `hNode` in the non-executable graph from which the executable graph was instantiated. `hNode` must not have been removed from the original graph.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

Allowed changes to parameters on executable graphs are as follows:

Node type |  Allowed changes
---|---
kernel |  See [cuGraphExecKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c> "Sets the parameters for a kernel node in the given graphExec.")
memcpy |  Addresses for 1-dimensional copies if allocated in same context; see [cuGraphExecMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g26186d58858ab32ccc7425b53786cce5> "Sets the parameters for a memcpy node in the given graphExec.")
memset |  Addresses for 1-dimensional memsets if allocated in same context; see [cuGraphExecMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5df5be09a0b7b3513e740ebbbcd59739> "Sets the parameters for a memset node in the given graphExec.")
host |  Unrestricted
child graph |  Topology must match and restrictions apply recursively; see [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible.")
event wait |  Unrestricted
event record |  Unrestricted
external semaphore signal |  Number of semaphore operations cannot change
external semaphore wait |  Number of semaphore operations cannot change
memory allocation |  API unsupported
memory free |  API unsupported
batch memops |  Addresses, values, and operation type for wait operations; see [cuGraphExecBatchMemOpNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g23f51bb4e4c029bb32fac0146e38c076> "Sets the parameters for a batch mem op node in the given graphExec.")

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters.")[cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExecUpdate ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphExecUpdateResultInfo](<structCUgraphExecUpdateResultInfo__v1.html#structCUgraphExecUpdateResultInfo__v1>)*Â resultInfo )


Check whether an executable graph can be updated with a graph and perform the update if possible.

######  Parameters

`hGraphExec`
    The instantiated graph to be updated
`hGraph`
    The graph containing the updated parameters
`resultInfo`
    the error info structure

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9114e389888b89d68639b330f478386c6>),

###### Description

Updates the node parameters in the instantiated graph specified by `hGraphExec` with the node parameters in a topologically identical graph specified by `hGraph`.

Limitations:

  * Kernel nodes:
    * The owning context of the function cannot change.

    * A node whose function originally did not use CUDA dynamic parallelism cannot be updated to a function which uses CDP.

    * A node whose function originally did not make device-side update calls cannot be updated to a function which makes device-side update calls.

    * A cooperative node cannot be updated to a non-cooperative node, and vice-versa.

    * If the graph was instantiated with CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, the priority attribute cannot change. Equality is checked on the originally requested priority values, before they are clamped to the device's supported range.

    * If `hGraphExec` was not instantiated for device launch, a node whose function originally did not use device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) cannot be updated to a function which uses device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) unless the node resides on the same context as nodes which contained such calls at instantiate-time. If no such calls were present at instantiation, these updates cannot be performed at all.

    * Neither `hGraph` nor `hGraphExec` may contain device-updatable kernel nodes.

  * Memset and memcpy nodes:
    * The CUDA device(s) to which the operand(s) was allocated/mapped cannot change.

    * The source/destination memory must be allocated from the same contexts as the original source/destination memory.

    * For 2d memsets, only address and assigned value may be updated.

    * For 1d memsets, updating dimensions is also allowed, but may fail if the resulting operation doesn't map onto the work resources already allocated for the node.

  * Additional memcpy node restrictions:
    * Changing either the source or destination memory type(i.e. CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_ARRAY, etc.) is not supported.

  * External semaphore wait nodes and record nodes:
    * Changing the number of semaphores is not supported.

  * Conditional nodes:
    * Changing node parameters is not supported.

    * Changing parameters of nodes within the conditional body graph is subject to the rules above.

    * Conditional handle flags and default values are updated as part of the graph update.


Note: The API may add further restrictions in future releases. The return code should always be checked.

cuGraphExecUpdate sets the result member of `resultInfo` to CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED under the following conditions:

  * The count of nodes directly in `hGraphExec` and `hGraph` differ, in which case resultInfo->errorNode is set to NULL.

  * `hGraph` has more exit nodes than `hGraph`, in which case resultInfo->errorNode is set to one of the exit nodes in hGraph.

  * A node in `hGraph` has a different number of dependencies than the node from `hGraphExec` it is paired with, in which case resultInfo->errorNode is set to the node from `hGraph`.

  * A node in `hGraph` has a dependency that does not match with the corresponding dependency of the paired node from `hGraphExec`. resultInfo->errorNode will be set to the node from `hGraph`. resultInfo->errorFromNode will be set to the mismatched dependency. The dependencies are paired based on edge order and a dependency does not match when the nodes are already paired based on other edges examined in the graph.


cuGraphExecUpdate sets the result member of `resultInfo` to:

  * CU_GRAPH_EXEC_UPDATE_ERROR if passed an invalid value.

  * CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED if the graph topology changed

  * CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED if the type of a node changed, in which case `hErrorNode_out` is set to the node from `hGraph`.

  * CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE if the function changed in an unsupported way(see note above), in which case `hErrorNode_out` is set to the node from `hGraph`

  * CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED if any parameters to a node changed in a way that is not supported, in which case `hErrorNode_out` is set to the node from `hGraph`.

  * CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED if any attributes of a node changed in a way that is not supported, in which case `hErrorNode_out` is set to the node from `hGraph`.

  * CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED if something about a node is unsupported, like the node's type or configuration, in which case `hErrorNode_out` is set to the node from `hGraph`


If the update fails for a reason not listed above, the result member of `resultInfo` will be set to CU_GRAPH_EXEC_UPDATE_ERROR. If the update succeeds, the result member will be set to CU_GRAPH_EXEC_UPDATE_SUCCESS.

cuGraphExecUpdate returns CUDA_SUCCESS when the updated was performed successfully. It returns CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE if the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExternalSemaphoresSignalNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â params_out )


Returns an external semaphore signal node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`params_out`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of an external semaphore signal node `hNode` in `params_out`. The `extSemArray` and `paramsArray` returned in `params_out`, are owned by the node. This memory remains valid until the node is destroyed or its parameters are modified, and should not be modified directly. Use [cuGraphExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a344ed4c6a5fcaad7bc7c53b04c6099> "Sets an external semaphore signal node's parameters.") to update the parameters of this node.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuGraphAddExternalSemaphoresSignalNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3> "Creates an external semaphore signal node and adds it to a graph."), [cuGraphExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a344ed4c6a5fcaad7bc7c53b04c6099> "Sets an external semaphore signal node's parameters."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExternalSemaphoresSignalNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_SIGNAL_NODE_PARAMS](<structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__SIGNAL__NODE__PARAMS__v1>)*Â nodeParams )


Sets an external semaphore signal node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Sets the parameters of an external semaphore signal node `hNode` to `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuGraphAddExternalSemaphoresSignalNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6410d5401de205568457fba5e1862ad3> "Creates an external semaphore signal node and adds it to a graph."), [cuGraphExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a344ed4c6a5fcaad7bc7c53b04c6099> "Sets an external semaphore signal node's parameters."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExternalSemaphoresWaitNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â params_out )


Returns an external semaphore wait node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`params_out`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of an external semaphore wait node `hNode` in `params_out`. The `extSemArray` and `paramsArray` returned in `params_out`, are owned by the node. This memory remains valid until the node is destroyed or its parameters are modified, and should not be modified directly. Use [cuGraphExternalSemaphoresSignalNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a344ed4c6a5fcaad7bc7c53b04c6099> "Sets an external semaphore signal node's parameters.") to update the parameters of this node.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuGraphExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge8b93792930a21ec352d6efd2c21c8c0> "Sets an external semaphore wait node's parameters."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphExternalSemaphoresWaitNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_EXT_SEM_WAIT_NODE_PARAMS](<structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1.html#structCUDA__EXT__SEM__WAIT__NODE__PARAMS__v1>)*Â nodeParams )


Sets an external semaphore wait node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Sets the parameters of an external semaphore wait node `hNode` to `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuGraphExternalSemaphoresWaitNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge8b93792930a21ec352d6efd2c21c8c0> "Sets an external semaphore wait node's parameters."), [cuGraphAddExternalSemaphoresWaitNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g49131c65fcef0b60b3939e008f7b467e> "Creates an external semaphore wait node and adds it to a graph."), [cuSignalExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g86cd6c4b3f439ba786f4e65d1b8107c3> "Signals a set of external semaphore objects."), [cuWaitExternalSemaphoresAsync](<group__CUDA__EXTRES__INTEROP.html#group__CUDA__EXTRES__INTEROP_1g063f01a524818ac89bacf521c55a39f0> "Waits on a set of external semaphore objects.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphGetEdges ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â from, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â to, [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_t*Â numEdges )


Returns a graph's dependency edges.

######  Parameters

`hGraph`
    \- Graph to get the edges from
`from`
    \- Location to return edge endpoints
`to`
    \- Location to return edge endpoints
`edgeData`
    \- Optional location to return edge data
`numEdges`
    \- See description

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns a list of `hGraph's` dependency edges. Edges are returned via corresponding indices in `from`, `to` and `edgeData`; that is, the node in `to`[i] has a dependency on the node in `from`[i] with data `edgeData`[i]. `from` and `to` may both be NULL, in which case this function only returns the number of edges in `numEdges`. Otherwise, `numEdges` entries will be filled in. If `numEdges` is higher than the actual number of edges, the remaining entries in `from` and `to` will be set to NULL, and the number of edges actually returned will be written to `numEdges`. `edgeData` may alone be NULL, in which case the edges must all have default (zeroed) edge data. Attempting a lossy query via NULL `edgeData` will result in [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>). If `edgeData` is non-NULL then `from` and `to` must be as well.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphGetRootNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd> "Returns a graph's root nodes."), [cuGraphAddDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5dad91f0be4e0fde6092f15797427e2d> "Adds dependency edges to a graph."), [cuGraphRemoveDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g25048b696f56b4d6131f068074176301> "Removes dependency edges from a graph."), [cuGraphNodeGetDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9> "Returns a node's dependencies."), [cuGraphNodeGetDependentNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450> "Returns a node's dependent nodes.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphGetId ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, unsigned int*Â graphId )


Returns the id of a given graph.

######  Parameters

`hGraph`
    \- Graph to query
`graphId`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the id of `hGraph` in `*graphId`. The value in `*graphId` will match that referenced by [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.").

**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.")[cuGraphNodeGetContainingGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbbfe267adf728f1c53aa9d99ba101b92> "Returns the graph that contains a given graph node.")[cuGraphNodeGetLocalId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18fd5107a28aaae1e396efcb0edaa70d> "Returns the local node id of a given graph node.")[cuGraphNodeGetToolsId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g10d4cf58921a26acce90ed1a03fcd4c1> "Returns an id used by tools to identify a given node.")[cuGraphExecGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a561a95ac508d0a99bccbf89aa01509> "Returns the id of a given graph exec.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphGetNodes ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â nodes, size_t*Â numNodes )


Returns a graph's nodes.

######  Parameters

`hGraph`
    \- Graph to query
`nodes`
    \- Pointer to return the nodes
`numNodes`
    \- See description

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns a list of `hGraph's` nodes. `nodes` may be NULL, in which case this function will return the number of nodes in `numNodes`. Otherwise, `numNodes` entries will be filled in. If `numNodes` is higher than the actual number of nodes, the remaining entries in `nodes` will be set to NULL, and the number of nodes actually obtained will be returned in `numNodes`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphGetRootNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd> "Returns a graph's root nodes."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphNodeGetType](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb1776d97aa1c9d5144774b29e4b8c3e> "Returns a node's type."), [cuGraphNodeGetDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9> "Returns a node's dependencies."), [cuGraphNodeGetDependentNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450> "Returns a node's dependent nodes.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphGetRootNodes ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â rootNodes, size_t*Â numRootNodes )


Returns a graph's root nodes.

######  Parameters

`hGraph`
    \- Graph to query
`rootNodes`
    \- Pointer to return the root nodes
`numRootNodes`
    \- See description

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns a list of `hGraph's` root nodes. `rootNodes` may be NULL, in which case this function will return the number of root nodes in `numRootNodes`. Otherwise, `numRootNodes` entries will be filled in. If `numRootNodes` is higher than the actual number of root nodes, the remaining entries in `rootNodes` will be set to NULL, and the number of nodes actually obtained will be returned in `numRootNodes`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphNodeGetType](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb1776d97aa1c9d5144774b29e4b8c3e> "Returns a node's type."), [cuGraphNodeGetDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9> "Returns a node's dependencies."), [cuGraphNodeGetDependentNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450> "Returns a node's dependent nodes.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphHostNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )


Returns a host node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`nodeParams`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of host node `hNode` in `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuLaunchHostFunc](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f> "Enqueues a host function call in a stream."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gae021ae8f19ee51044339db9c24dd266> "Sets a host node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphHostNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_HOST_NODE_PARAMS](<structCUDA__HOST__NODE__PARAMS__v1.html#structCUDA__HOST__NODE__PARAMS__v1>)*Â nodeParams )


Sets a host node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the parameters of host node `hNode` to `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuLaunchHostFunc](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gab95a78143bae7f21eebb978f91e7f3f> "Enqueues a host function call in a stream."), [cuGraphAddHostNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0809d65e85a3c052296373954a05b1d6> "Creates a host execution node and adds it to a graph."), [cuGraphHostNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2e3ea6000089fd5523c197ab5e73d5a2> "Returns a host node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphInstantiate ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)*Â phGraphExec, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, unsigned long longÂ flags )


Creates an executable graph from a graph.

######  Parameters

`phGraphExec`
    \- Returns instantiated graph
`hGraph`
    \- Graph to instantiate
`flags`
    \- Flags to control instantiation. See [CUgraphInstantiate_flags](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g070bf5517d3a7915667c256eefce4956>).

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Instantiates `hGraph` as an executable graph. The graph is validated for any structural constraints or intra-node constraints which were not previously validated. If instantiation is successful, a handle to the instantiated graph is returned in `phGraphExec`.

The `flags` parameter controls the behavior of instantiation and subsequent graph launches. Valid flags are:

  * [CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49561684f715bf05e39afd69aa508299a479>), which configures a graph containing memory allocation nodes to automatically free any unfreed memory allocations before the graph is relaunched.


  * [CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce4956eefa4d807b378590e82a916f55f370e1>), which configures the graph for launch from the device. If this flag is passed, the executable graph handle returned can be used to launch the graph from both the host and device. This flag can only be used on platforms which support unified addressing. This flag cannot be used in conjunction with [CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49561684f715bf05e39afd69aa508299a479>).


  * [CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce4956a02ee23c391827d000f1230ad281da29>), which causes the graph to use the priorities from the per-node attributes rather than the priority of the launch stream during execution. Note that priorities are only available on kernel nodes, and are copied from stream priority during stream capture.


If `hGraph` contains any allocation or free nodes, there can be at most one executable graph in existence for that graph at a time. An attempt to instantiate a second executable graph before destroying the first with [cuGraphExecDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7> "Destroys an executable graph.") will result in an error. The same also applies if `hGraph` contains any device-updatable kernel nodes.

If `hGraph` contains kernels which call device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) from multiple contexts, this will result in an error.

Graphs instantiated for launch on the device have additional restrictions which do not apply to host graphs:

  * The graph's nodes must reside on a single context.

  * The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.

  * The graph cannot be empty and must contain at least one kernel, memcpy, or memset node. Operation-specific restrictions are outlined below.

  * Kernel nodes:
    * Use of CUDA Dynamic Parallelism is not permitted.

    * Cooperative launches are permitted as long as MPS is not in use.

  * Memcpy nodes:
    * Only copies involving device memory and/or pinned device-mapped host memory are permitted.

    * Copies involving CUDA arrays are not permitted.

    * Both operands must be accessible from the current context, and the current context must match the context of other nodes in the graph.


Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphUpload](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7eb9849e6e4604864a482b38f25be48> "Uploads an executable graph in a stream."), [cuGraphLaunch](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471> "Launches an executable graph in a stream."), [cuGraphExecDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7> "Destroys an executable graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphInstantiateWithParams ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)*Â phGraphExec, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, [CUDA_GRAPH_INSTANTIATE_PARAMS](<structCUDA__GRAPH__INSTANTIATE__PARAMS.html#structCUDA__GRAPH__INSTANTIATE__PARAMS>)*Â instantiateParams )


Creates an executable graph from a graph.

######  Parameters

`phGraphExec`
    \- Returns instantiated graph
`hGraph`
    \- Graph to instantiate
`instantiateParams`
    \- Instantiation parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Instantiates `hGraph` as an executable graph according to the `instantiateParams` structure. The graph is validated for any structural constraints or intra-node constraints which were not previously validated. If instantiation is successful, a handle to the instantiated graph is returned in `phGraphExec`.

`instantiateParams` controls the behavior of instantiation and subsequent graph launches, as well as returning more detailed information in the event of an error. [CUDA_GRAPH_INSTANTIATE_PARAMS](<structCUDA__GRAPH__INSTANTIATE__PARAMS.html#structCUDA__GRAPH__INSTANTIATE__PARAMS>) is defined as:


    â    typedef struct {
                  cuuint64_t flags;
                  [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>) hUploadStream;
                  [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>) hErrNode_out;
                  [CUgraphInstantiateResult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g863484740f7d9f82c908d228f791cc56>) result_out;
              } [CUDA_GRAPH_INSTANTIATE_PARAMS](<structCUDA__GRAPH__INSTANTIATE__PARAMS.html#structCUDA__GRAPH__INSTANTIATE__PARAMS>);

The `flags` field controls the behavior of instantiation and subsequent graph launches. Valid flags are:

  * [CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49561684f715bf05e39afd69aa508299a479>), which configures a graph containing memory allocation nodes to automatically free any unfreed memory allocations before the graph is relaunched.


  * [CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49569557f13a16fe73b147fb4c9018e92925>), which will perform an upload of the graph into `hUploadStream` once the graph has been instantiated.


  * [CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce4956eefa4d807b378590e82a916f55f370e1>), which configures the graph for launch from the device. If this flag is passed, the executable graph handle returned can be used to launch the graph from both the host and device. This flag can only be used on platforms which support unified addressing. This flag cannot be used in conjunction with [CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49561684f715bf05e39afd69aa508299a479>).


  * [CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce4956a02ee23c391827d000f1230ad281da29>), which causes the graph to use the priorities from the per-node attributes rather than the priority of the launch stream during execution. Note that priorities are only available on kernel nodes, and are copied from stream priority during stream capture.


If `hGraph` contains any allocation or free nodes, there can be at most one executable graph in existence for that graph at a time. An attempt to instantiate a second executable graph before destroying the first with [cuGraphExecDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7> "Destroys an executable graph.") will result in an error. The same also applies if `hGraph` contains any device-updatable kernel nodes.

If `hGraph` contains kernels which call device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) from multiple contexts, this will result in an error.

Graphs instantiated for launch on the device have additional restrictions which do not apply to host graphs:

  * The graph's nodes must reside on a single context.

  * The graph can only contain kernel nodes, memcpy nodes, memset nodes, and child graph nodes.

  * The graph cannot be empty and must contain at least one kernel, memcpy, or memset node. Operation-specific restrictions are outlined below.

  * Kernel nodes:
    * Use of CUDA Dynamic Parallelism is not permitted.

    * Cooperative launches are permitted as long as MPS is not in use.

  * Memcpy nodes:
    * Only copies involving device memory and/or pinned device-mapped host memory are permitted.

    * Copies involving CUDA arrays are not permitted.

    * Both operands must be accessible from the current context, and the current context must match the context of other nodes in the graph.


In the event of an error, the `result_out` and `hErrNode_out` fields will contain more information about the nature of the error. Possible error reporting includes:

  * [CUDA_GRAPH_INSTANTIATE_ERROR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg863484740f7d9f82c908d228f791cc56e625bf48bea298c8f002e84ecaa8dbf3>), if passed an invalid value or if an unexpected error occurred which is described by the return value of the function. `hErrNode_out` will be set to NULL.

  * [CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg863484740f7d9f82c908d228f791cc56bdea682421fdd0bfcd4f466ab02b1d8b>), if the graph structure is invalid. `hErrNode_out` will be set to one of the offending nodes.

  * [CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg863484740f7d9f82c908d228f791cc562c63d294c9e492a6862bf1d738a63b32>), if the graph is instantiated for device launch but contains a node of an unsupported node type, or a node which performs unsupported operations, such as use of CUDA dynamic parallelism within a kernel node. `hErrNode_out` will be set to this node.

  * [CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg863484740f7d9f82c908d228f791cc5648704cf82fdc5141cb009b92077b7a19>), if the graph is instantiated for device launch but a nodeâs context differs from that of another node. This error can also be returned if a graph is not instantiated for device launch and it contains kernels which call device-side [cudaGraphLaunch()](<../cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1920584881db959c8c74130d79019b73>) from multiple contexts. `hErrNode_out` will be set to this node.


If instantiation is successful, `result_out` will be set to [CUDA_GRAPH_INSTANTIATE_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg863484740f7d9f82c908d228f791cc56668c8105a3caeb8f4bed6b5581900bba>), and `hErrNode_out` will be set to NULL.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphExecDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7> "Destroys an executable graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphKernelNodeCopyAttributes ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â dst, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â src )


Copies attributes from source node to destination node.

######  Parameters

`dst`
    Destination node
`src`
    Source node For list of attributes see CUkernelNodeAttrID

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies attributes from source node `src` to destination node `dst`. Both node must have the same context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphKernelNodeGetAttribute ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUkernelNodeAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, [CUkernelNodeAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value_out )


Queries node attribute.

######  Parameters

`hNode`

`attr`

`value_out`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Queries attribute `attr` from node `hNode` and stores it in corresponding member of `value_out`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphKernelNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )


Returns a kernel node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`nodeParams`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of kernel node `hNode` in `nodeParams`. The `kernelParams` or `extra` array returned in `nodeParams`, as well as the argument values it points to, are owned by the node. This memory remains valid until the node is destroyed or its parameters are modified, and should not be modified directly. Use [cuGraphKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga268bf2fd520f5aa3a3d700005df6703> "Sets a kernel node's parameters.") to update the parameters of this node.

The params will contain either `kernelParams` or `extra`, according to which of these was most recently set on the node.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga268bf2fd520f5aa3a3d700005df6703> "Sets a kernel node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphKernelNodeSetAttribute ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUkernelNodeAttrID](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6f6565b334be6bb3134868e10bbdd331>)Â attr, const [CUkernelNodeAttrValue](<unionCUlaunchAttributeValue.html#unionCUlaunchAttributeValue>)*Â value )


Sets node attribute.

######  Parameters

`hNode`

`attr`

`value`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Sets attribute `attr` on node `hNode` from corresponding attribute of `value`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[CUaccessPolicyWindow](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g1838e6438f39944217e384bf2adad477>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphKernelNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_KERNEL_NODE_PARAMS](<structCUDA__KERNEL__NODE__PARAMS__v2.html#structCUDA__KERNEL__NODE__PARAMS__v2>)*Â nodeParams )


Sets a kernel node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Sets the parameters of kernel node `hNode` to `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuLaunchKernel](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15> "Launches a CUDA function CUfunction or a CUDA kernel CUkernel."), [cuGraphAddKernelNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g50d871e3bd06c1b835e52f2966ef366b> "Creates a kernel execution node and adds it to a graph."), [cuGraphKernelNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb8df3f99e8dd5e4f4a5a0f19a5518252> "Returns a kernel node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphLaunch ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Launches an executable graph in a stream.

######  Parameters

`hGraphExec`
    \- Executable graph to launch
`hStream`
    \- Stream in which to launch the graph

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Executes `hGraphExec` in `hStream`. Only one instance of `hGraphExec` may be executing at a time. Each launch is ordered behind both any previous work in `hStream` and any previous launches of `hGraphExec`. To execute a graph concurrently, it must be instantiated multiple times into multiple executable graphs.

If any allocations created by `hGraphExec` remain unfreed (from a previous launch) and `hGraphExec` was not instantiated with [CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg070bf5517d3a7915667c256eefce49561684f715bf05e39afd69aa508299a479>), the launch will fail with [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphUpload](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga7eb9849e6e4604864a482b38f25be48> "Uploads an executable graph in a stream."), [cuGraphExecDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7> "Destroys an executable graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphMemAllocNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_MEM_ALLOC_NODE_PARAMS](<structCUDA__MEM__ALLOC__NODE__PARAMS__v1.html#structCUDA__MEM__ALLOC__NODE__PARAMS__v1>)*Â params_out )


Returns a memory alloc node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`params_out`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of a memory alloc node `hNode` in `params_out`. The `poolProps` and `accessDescs` returned in `params_out`, are owned by the node. This memory remains valid until the node is destroyed. The returned parameters must not be modified.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddMemAllocNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g73a351cb71b2945a0bcb913a93f69ec9> "Creates an allocation node and adds it to a graph."), [cuGraphMemFreeNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd24d9fe5769222a2367e3f571fb2f28b> "Returns a memory free node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphMemFreeNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr_out )


Returns a memory free node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`dptr_out`
    \- Pointer to return the device address

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the address of a memory free node `hNode` in `dptr_out`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddMemFreeNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1geb7cdce5d9be2d28d9428e74eb00fa53> "Creates a memory free node and adds it to a graph."), [cuGraphMemAllocNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gee2c7d66d3d96b1470c1d1a769f250a2> "Returns a memory alloc node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphMemcpyNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â nodeParams )


Returns a memcpy node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`nodeParams`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of memcpy node `hNode` in `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga278a7ec0700c86abb0b2cfdf4d3dc1d> "Sets a memcpy node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphMemcpyNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â nodeParams )


Sets a memcpy node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets the parameters of memcpy node `hNode` to `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuGraphAddMemcpyNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g674da6ab54a677f13e0e0e8206ff5073> "Creates a memcpy node and adds it to a graph."), [cuGraphMemcpyNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g572889131dbc31720eff94b130f4005b> "Returns a memcpy node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphMemsetNodeGetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â nodeParams )


Returns a memset node's parameters.

######  Parameters

`hNode`
    \- Node to get the parameters for
`nodeParams`
    \- Pointer to return the parameters

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the parameters of memset node `hNode` in `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph."), [cuGraphMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gc27f3fd83a6e33c74519066fbaa0de67> "Sets a memset node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphMemsetNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, const [CUDA_MEMSET_NODE_PARAMS](<structCUDA__MEMSET__NODE__PARAMS__v1.html#structCUDA__MEMSET__NODE__PARAMS__v1>)*Â nodeParams )


Sets a memset node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the parameters of memset node `hNode` to `nodeParams`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbf18157f40ea2d160cb0b9e4e2b16139> "Update's a graph node's parameters."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuGraphAddMemsetNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g89dc8fc3743392777c0daa2c4aca40d3> "Creates a memset node and adds it to a graph."), [cuGraphMemsetNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18830edcfd982f952820a0d7f91b894a> "Returns a memset node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeFindInClone ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â phNode, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hOriginalNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hClonedGraph )


Finds a cloned version of a node.

######  Parameters

`phNode`
    \- Returns handle to the cloned node
`hOriginalNode`
    \- Handle to the original node
`hClonedGraph`
    \- Cloned graph to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

This function returns the node in `hClonedGraph` corresponding to `hOriginalNode` in the original graph.

`hClonedGraph` must have been cloned from `hOriginalGraph` via [cuGraphClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e> "Clones a graph."). `hOriginalNode` must have been in `hOriginalGraph` at the time of the call to [cuGraphClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e> "Clones a graph."), and the corresponding cloned node in `hClonedGraph` must not have been removed. The cloned node is then returned via `phClonedNode`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphClone](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e> "Clones a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetContainingGraph ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)*Â phGraph )


Returns the graph that contains a given graph node.

######  Parameters

`hNode`
    \- Node to query
`phGraph`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the graph that contains `hNode` in `*phGraph`. If `hNode` is in a child graph, the child graph it is in is returned.

**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.")[cuGraphNodeGetLocalId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18fd5107a28aaae1e396efcb0edaa70d> "Returns the local node id of a given graph node.")[cuGraphNodeGetToolsId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g10d4cf58921a26acce90ed1a03fcd4c1> "Returns an id used by tools to identify a given node.")[cuGraphGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0f05ae29d14198ff57d722156d60aa41> "Returns the id of a given graph.")[cuGraphExecGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a561a95ac508d0a99bccbf89aa01509> "Returns the id of a given graph exec.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetDependencies ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependencies, [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_t*Â numDependencies )


Returns a node's dependencies.

######  Parameters

`hNode`
    \- Node to query
`dependencies`
    \- Pointer to return the dependencies
`edgeData`
    \- Optional array to return edge data for each dependency
`numDependencies`
    \- See description

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns a list of `node's` dependencies. `dependencies` may be NULL, in which case this function will return the number of dependencies in `numDependencies`. Otherwise, `numDependencies` entries will be filled in. If `numDependencies` is higher than the actual number of dependencies, the remaining entries in `dependencies` will be set to NULL, and the number of nodes actually obtained will be returned in `numDependencies`.

Note that if an edge has non-zero (non-default) edge data and `edgeData` is NULL, this API will return [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>). If `edgeData` is non-NULL, then `dependencies` must be as well.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeGetDependentNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450> "Returns a node's dependent nodes."), [cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphGetRootNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd> "Returns a graph's root nodes."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphAddDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5dad91f0be4e0fde6092f15797427e2d> "Adds dependency edges to a graph."), [cuGraphRemoveDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g25048b696f56b4d6131f068074176301> "Removes dependency edges from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetDependentNodes ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â dependentNodes, [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_t*Â numDependentNodes )


Returns a node's dependent nodes.

######  Parameters

`hNode`
    \- Node to query
`dependentNodes`
    \- Pointer to return the dependent nodes
`edgeData`
    \- Optional pointer to return edge data for dependent nodes
`numDependentNodes`
    \- See description

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns a list of `node's` dependent nodes. `dependentNodes` may be NULL, in which case this function will return the number of dependent nodes in `numDependentNodes`. Otherwise, `numDependentNodes` entries will be filled in. If `numDependentNodes` is higher than the actual number of dependent nodes, the remaining entries in `dependentNodes` will be set to NULL, and the number of nodes actually obtained will be returned in `numDependentNodes`.

Note that if an edge has non-zero (non-default) edge data and `edgeData` is NULL, this API will return [CUDA_ERROR_LOSSY_QUERY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90c2195e65483c3e7f0ccbf52370c33f7>). If `edgeData` is non-NULL, then `dependentNodes` must be as well.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeGetDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9> "Returns a node's dependencies."), [cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphGetRootNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd> "Returns a graph's root nodes."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphAddDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5dad91f0be4e0fde6092f15797427e2d> "Adds dependency edges to a graph."), [cuGraphRemoveDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g25048b696f56b4d6131f068074176301> "Removes dependency edges from a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetEnabled ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned int*Â isEnabled )


Query whether a node in the given graphExec is enabled.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Node from the graph from which graphExec was instantiated
`isEnabled`
    \- Location to return the enabled status of the node

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets isEnabled to 1 if `hNode` is enabled, or 0 if `hNode` is disabled.

The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

`hNode` must not have been removed from the original graph.

Note:

  * Currently only kernel, memset and memcpy nodes are supported.

  * This function will not reflect device-side updates for device-updatable kernel nodes.


Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeSetEnabled](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g371b20eb0c0658731e38db7e68f12c78> "Enables or disables the specified node in the given graphExec."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")[cuGraphLaunch](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471> "Launches an executable graph in a stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetLocalId ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned int*Â nodeId )


Returns the local node id of a given graph node.

######  Parameters

`hNode`
    \- Node to query
`nodeId`
    \- Pointer to return the nodeId

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the node id of `hNode` in `*nodeId`. The nodeId matches that referenced by [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure."). The local nodeId and graphId together can uniquely identify the node.

**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.")[cuGraphNodeGetContainingGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbbfe267adf728f1c53aa9d99ba101b92> "Returns the graph that contains a given graph node.")[cuGraphNodeGetToolsId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g10d4cf58921a26acce90ed1a03fcd4c1> "Returns an id used by tools to identify a given node.")[cuGraphGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0f05ae29d14198ff57d722156d60aa41> "Returns the id of a given graph.")[cuGraphExecGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a561a95ac508d0a99bccbf89aa01509> "Returns the id of a given graph exec.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetToolsId ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned long long*Â toolsNodeId )


Returns an id used by tools to identify a given node.

######  Parameters

`hNode`
    \- Node to query
`toolsNodeId`


###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphDebugDotPrint](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0fb0c4d319477a0a98da005fcb0dacc4> "Write a DOT file describing graph structure.")[cuGraphNodeGetContainingGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbbfe267adf728f1c53aa9d99ba101b92> "Returns the graph that contains a given graph node.")[cuGraphNodeGetLocalId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18fd5107a28aaae1e396efcb0edaa70d> "Returns the local node id of a given graph node.")[cuGraphGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g0f05ae29d14198ff57d722156d60aa41> "Returns the id of a given graph.")[cuGraphExecGetId](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g7a561a95ac508d0a99bccbf89aa01509> "Returns the id of a given graph exec.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeGetType ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNodeType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0731a28f826922120d783d8444e154dc>)*Â type )


Returns a node's type.

######  Parameters

`hNode`
    \- Node to query
`type`
    \- Pointer to return the node type

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the node type of `hNode` in `type`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphGetNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5> "Returns a graph's nodes."), [cuGraphGetRootNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd> "Returns a graph's root nodes."), [cuGraphChildGraphNodeGetGraph](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbe9fc9267316b3778ef0db507917b4fd> "Gets a handle to the embedded graph of a child graph node."), [cuGraphKernelNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb8df3f99e8dd5e4f4a5a0f19a5518252> "Returns a kernel node's parameters."), [cuGraphKernelNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga268bf2fd520f5aa3a3d700005df6703> "Sets a kernel node's parameters."), [cuGraphHostNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2e3ea6000089fd5523c197ab5e73d5a2> "Returns a host node's parameters."), [cuGraphHostNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gae021ae8f19ee51044339db9c24dd266> "Sets a host node's parameters."), [cuGraphMemcpyNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g572889131dbc31720eff94b130f4005b> "Returns a memcpy node's parameters."), [cuGraphMemcpyNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga278a7ec0700c86abb0b2cfdf4d3dc1d> "Sets a memcpy node's parameters."), [cuGraphMemsetNodeGetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g18830edcfd982f952820a0d7f91b894a> "Returns a memset node's parameters."), [cuGraphMemsetNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gc27f3fd83a6e33c74519066fbaa0de67> "Sets a memset node's parameters.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeSetEnabled ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, unsigned int Â isEnabled )


Enables or disables the specified node in the given graphExec.

######  Parameters

`hGraphExec`
    \- The executable graph in which to set the specified node
`hNode`
    \- Node from the graph from which graphExec was instantiated
`isEnabled`
    \- Node is enabled if != 0, otherwise the node is disabled

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Sets `hNode` to be either enabled or disabled. Disabled nodes are functionally equivalent to empty nodes until they are reenabled. Existing node parameters are not affected by disabling/enabling the node.

The node is identified by the corresponding node `hNode` in the non-executable graph, from which the executable graph was instantiated.

`hNode` must not have been removed from the original graph.

The modifications only affect future launches of `hGraphExec`. Already enqueued or running launches of `hGraphExec` are not affected by this call. `hNode` is also not modified by this call.

If `hNode` is a device-updatable kernel node, the next upload/launch of `hGraphExec` will overwrite any previous device-side updates. Additionally, applying host updates to a device-updatable kernel node while it is being updated from the device will result in undefined behavior.

Note:

Currently only kernel, memset and memcpy nodes are supported.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphNodeGetEnabled](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g428f51dceec6f6211bb9c1d710925a3d> "Query whether a node in the given graphExec is enabled."), [cuGraphExecUpdate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g96efefc56df46927da7297f122adfb9f> "Check whether an executable graph can be updated with a graph and perform the update if possible."), [cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph.")[cuGraphLaunch](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471> "Launches an executable graph in a stream.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphNodeSetParams ( [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)Â hNode, [CUgraphNodeParams](<structCUgraphNodeParams.html#structCUgraphNodeParams>)*Â nodeParams )


Update's a graph node's parameters.

######  Parameters

`hNode`
    \- Node to set the parameters for
`nodeParams`
    \- Parameters to copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Sets the parameters of graph node `hNode` to `nodeParams`. The node type specified by `nodeParams->type` must match the type of `hNode`. `nodeParams` must be fully initialized and all unused bytes (reserved, padding) zeroed.

Modifying parameters is not supported for node types CU_GRAPH_NODE_TYPE_MEM_ALLOC and CU_GRAPH_NODE_TYPE_MEM_FREE.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddNode](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge01208e62f72a53367a2af903bf17d23> "Adds a node of arbitrary type to a graph."), [cuGraphExecNodeSetParams](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb318c5b61ada0e333bb12d1d33dae48b> "Update's a graph node's parameters in an instantiated graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphReleaseUserObject ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â graph, [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count )


Release a user object reference from a graph.

######  Parameters

`graph`
    \- The graph that will release the reference
`object`
    \- The user object to release a reference for
`count`
    \- The number of references to release, typically 1. Must be nonzero and not larger than INT_MAX.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Releases user object references owned by a graph.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.

**See also:**

[cuUserObjectCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g58f04e0ac0ad23d2f15ea6e9f6c8a999> "Create a user object."), [cuUserObjectRetain](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge022bcecdeca2d14cc8f28afc6a2eaf6> "Retain a reference to a user object."), [cuUserObjectRelease](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga2c16918341b8d020c9246e75658cc80> "Release a reference to a user object."), [cuGraphRetainUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gaffd130c928e56740a2a5aaeb6125c8a> "Retain a reference to a user object from a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphRemoveDependencies ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â hGraph, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â from, const [CUgraphNode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc72514a94dacc85ed0617f979211079c>)*Â to, const [CUgraphEdgeData](<structCUgraphEdgeData.html#structCUgraphEdgeData>)*Â edgeData, size_tÂ numDependencies )


Removes dependency edges from a graph.

######  Parameters

`hGraph`
    \- Graph from which to remove dependencies
`from`
    \- Array of nodes that provide the dependencies
`to`
    \- Array of dependent nodes
`edgeData`
    \- Optional array of edge data. If NULL, edge data is assumed to be default (zeroed).
`numDependencies`
    \- Number of dependencies to be removed

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

The number of `dependencies` to be removed is defined by `numDependencies`. Elements in `from` and `to` at corresponding indices define a dependency. Each node in `from` and `to` must belong to `hGraph`.

If `numDependencies` is 0, elements in `from` and `to` will be ignored. Specifying an edge that does not exist in the graph, with data matching `edgeData`, results in an error. `edgeData` is nullable, which is equivalent to passing default (zeroed) data for each edge.

Dependencies cannot be removed from graphs which contain allocation or free nodes. Any attempt to do so will return an error.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphAddDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g5dad91f0be4e0fde6092f15797427e2d> "Adds dependency edges to a graph."), [cuGraphGetEdges](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4e3183ca455aae2e832edd4034094082> "Returns a graph's dependency edges."), [cuGraphNodeGetDependencies](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd3fc7f62e46f621f59de2173e08fccc9> "Returns a node's dependencies."), [cuGraphNodeGetDependentNodes](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g61e907fa6896b5393246d1588c794450> "Returns a node's dependent nodes.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphRetainUserObject ( [CUgraph](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g69f555c38df5b3fa1ed25efef794739a>)Â graph, [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count, unsigned int Â flags )


Retain a reference to a user object from a graph.

######  Parameters

`graph`
    \- The graph to associate the reference with
`object`
    \- The user object to retain a reference for
`count`
    \- The number of references to add to the graph, typically 1. Must be nonzero and not larger than INT_MAX.
`flags`
    \- The optional flag [CU_GRAPH_USER_OBJECT_MOVE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg1649c3055c83c32f812faac63c8da0b1f9b815bff431e87e54037568c4677b9d>) transfers references from the calling thread, rather than create new references. Pass 0 to create new references.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates or moves user object references that will be owned by a CUDA graph.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.

**See also:**

[cuUserObjectCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g58f04e0ac0ad23d2f15ea6e9f6c8a999> "Create a user object."), [cuUserObjectRetain](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge022bcecdeca2d14cc8f28afc6a2eaf6> "Retain a reference to a user object."), [cuUserObjectRelease](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga2c16918341b8d020c9246e75658cc80> "Release a reference to a user object."), [cuGraphReleaseUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g232c84cc31e13e4201a421e28561eebf> "Release a user object reference from a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGraphUpload ( [CUgraphExec](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf0abeceeaa9f0a39592fe36a538ea1f0>)Â hGraphExec, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Uploads an executable graph in a stream.

######  Parameters

`hGraphExec`
    \- Executable graph to upload
`hStream`
    \- Stream in which to upload the graph

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Uploads `hGraphExec` to the device in `hStream` without executing it. Uploads of the same `hGraphExec` will be serialized. Each upload is ordered behind both any previous work in `hStream` and any previous launches of `hGraphExec`. Uses memory cached by `stream` to back the allocations owned by `hGraphExec`.

Note:

  * Graph objects are not threadsafe. [More here](<graphs-thread-safety.html#graphs-thread-safety>).

  * Note that this function may also return error codes from previous, asynchronous launches.


**See also:**

[cuGraphInstantiate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1> "Creates an executable graph from a graph."), [cuGraphLaunch](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471> "Launches an executable graph in a stream."), [cuGraphExecDestroy](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7> "Destroys an executable graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuUserObjectCreate ( [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)*Â object_out, void*Â ptr, [CUhostFn](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g262cd3570ff5d396db4e3dabede3c355>)Â destroy, unsigned int Â initialRefcount, unsigned int Â flags )


Create a user object.

######  Parameters

`object_out`
    \- Location to return the user object handle
`ptr`
    \- The pointer to pass to the destroy function
`destroy`
    \- Callback to free the user object when it is no longer in use
`initialRefcount`
    \- The initial refcount to create the object with, typically 1. The initial references are owned by the calling thread.
`flags`
    \- Currently it is required to pass [CU_USER_OBJECT_NO_DESTRUCTOR_SYNC](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg11c53cd19ee840b3b0f597d57451e94330bb85019bed1f36b8130cef76085e27>), which is the only defined flag. This indicates that the destroy callback cannot be waited on by any CUDA API. Users requiring synchronization of the callback should signal its completion manually.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Create a user object with the specified destructor callback and initial reference count. The initial references are owned by the caller.

Destructor callbacks cannot make CUDA API calls and should avoid blocking behavior, as they are executed by a shared internal thread. Another thread may be signaled to perform such actions, if it does not block forward progress of tasks scheduled through CUDA.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.

**See also:**

[cuUserObjectRetain](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge022bcecdeca2d14cc8f28afc6a2eaf6> "Retain a reference to a user object."), [cuUserObjectRelease](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga2c16918341b8d020c9246e75658cc80> "Release a reference to a user object."), [cuGraphRetainUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gaffd130c928e56740a2a5aaeb6125c8a> "Retain a reference to a user object from a graph."), [cuGraphReleaseUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g232c84cc31e13e4201a421e28561eebf> "Release a user object reference from a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuUserObjectRelease ( [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count )


Release a reference to a user object.

######  Parameters

`object`
    \- The object to release
`count`
    \- The number of references to release, typically 1. Must be nonzero and not larger than INT_MAX.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Releases user object references owned by the caller. The object's destructor is invoked if the reference count reaches zero.

It is undefined behavior to release references not owned by the caller, or to use a user object handle after all references are released.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.

**See also:**

[cuUserObjectCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g58f04e0ac0ad23d2f15ea6e9f6c8a999> "Create a user object."), [cuUserObjectRetain](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ge022bcecdeca2d14cc8f28afc6a2eaf6> "Retain a reference to a user object."), [cuGraphRetainUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gaffd130c928e56740a2a5aaeb6125c8a> "Retain a reference to a user object from a graph."), [cuGraphReleaseUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g232c84cc31e13e4201a421e28561eebf> "Release a user object reference from a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuUserObjectRetain ( [CUuserObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g2578b65c87dc98d336f99edca913e92b>)Â object, unsigned int Â count )


Retain a reference to a user object.

######  Parameters

`object`
    \- The object to retain
`count`
    \- The number of references to retain, typically 1. Must be nonzero and not larger than INT_MAX.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Retains new references to a user object. The new references are owned by the caller.

See CUDA User Objects in the CUDA C++ Programming Guide for more information on user objects.

**See also:**

[cuUserObjectCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g58f04e0ac0ad23d2f15ea6e9f6c8a999> "Create a user object."), [cuUserObjectRelease](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga2c16918341b8d020c9246e75658cc80> "Release a reference to a user object."), [cuGraphRetainUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gaffd130c928e56740a2a5aaeb6125c8a> "Retain a reference to a user object from a graph."), [cuGraphReleaseUserObject](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g232c84cc31e13e4201a421e28561eebf> "Release a user object reference from a graph."), [cuGraphCreate](<group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf> "Creates a graph.")

* * *
