# Memory Management

## 6.13.Â Memory Management

This section describes the memory management functions of the low-level CUDA driver application programming interface.

### Classes

structÂ

[CUmemDecompressParams](<structCUmemDecompressParams.html#structCUmemDecompressParams> "Structure describing the parameters that compose a single decompression operation.")

     Structure describing the parameters that compose a single decompression operation. [](<structCUmemDecompressParams.html#structCUmemDecompressParams> "Structure describing the parameters that compose a single decompression operation.")

### Enumerations

enumÂ [CUmemDecompressAlgorithm](<#group__CUDA__MEM_1g6c015495b909c100e19cd4ddafceee91>)
     Bitmasks for CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArray3DCreate](<#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pHandle, const [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>)*Â pAllocateArray )
     Creates a 3D CUDA array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArray3DGetDescriptor](<#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857>) ( [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>)*Â pArrayDescriptor, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray )
     Get a 3D CUDA array descriptor.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArrayCreate](<#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pHandle, const [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>)*Â pAllocateArray )
     Creates a 1D or 2D CUDA array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArrayDestroy](<#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray )
     Destroys a CUDA array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArrayGetDescriptor](<#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf>) ( [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>)*Â pArrayDescriptor, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray )
     Get a 1D or 2D CUDA array descriptor.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArrayGetMemoryRequirements](<#group__CUDA__MEM_1gac8761ced0fa462e4762f6528073d9f4>) ( [CUDA_ARRAY_MEMORY_REQUIREMENTS](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1>)*Â memoryRequirements, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â array, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )
     Returns the memory requirements of a CUDA array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArrayGetPlane](<#group__CUDA__MEM_1ge66ce245a1e3802f9ccc3583cec6b71f>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pPlaneArray, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray, unsigned int Â planeIdx )
     Gets a CUDA array plane from a CUDA array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuArrayGetSparseProperties](<#group__CUDA__MEM_1gf74df88a07404ee051f0e5b36647d8c7>) ( [CUDA_ARRAY_SPARSE_PROPERTIES](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1>)*Â sparseProperties, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â array )
     Returns the layout properties of a sparse CUDA array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetByPCIBusId](<#group__CUDA__MEM_1ga89cd3fa06334ba7853ed1232c5ebe2a>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â dev, const char*Â pciBusId )
     Returns a handle to a compute device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetPCIBusId](<#group__CUDA__MEM_1g85295e7d9745ab8f0aa80dd1e172acfc>) ( char*Â pciBusId, int Â len, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns a PCI Bus Id string for the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceRegisterAsyncNotification](<#group__CUDA__MEM_1g4325f3e53f7817c93b37f12da91ed199>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUasyncCallback](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g466d4731f270b66441a355ddb2c84777>)Â callbackFunc, void*Â userData, [CUasyncCallbackHandle](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0ff5c3d4645d51b02b6d11b8b0c228c5>)*Â callback )
     Registers a callback function to receive async notifications.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceUnregisterAsyncNotification](<#group__CUDA__MEM_1gc0ae698fd18cbc2c395c9140e28e83ca>) ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUasyncCallbackHandle](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0ff5c3d4645d51b02b6d11b8b0c228c5>)Â callback )
     Unregisters an async notification callback.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuIpcCloseMemHandle](<#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )
     Attempts to close memory mapped with cuIpcOpenMemHandle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuIpcGetEventHandle](<#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c>) ( [CUipcEventHandle](<structCUipcEventHandle__v1.html#structCUipcEventHandle__v1>)*Â pHandle, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )
     Gets an interprocess handle for a previously allocated event.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuIpcGetMemHandle](<#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1>) ( [CUipcMemHandle](<structCUipcMemHandle__v1.html#structCUipcMemHandle__v1>)*Â pHandle, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )
     Gets an interprocess memory handle for an existing device memory allocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuIpcOpenEventHandle](<#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e>) ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â phEvent, [CUipcEventHandle](<structCUipcEventHandle__v1.html#structCUipcEventHandle__v1>)Â handle )
     Opens an interprocess event handle for use in the current process.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuIpcOpenMemHandle](<#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pdptr, [CUipcMemHandle](<structCUipcMemHandle__v1.html#structCUipcMemHandle__v1>)Â handle, unsigned int Â Flags )
     Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAlloc](<#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize )
     Allocates device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAllocHost](<#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0>) ( void**Â pp, size_tÂ bytesize )
     Allocates page-locked host memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAllocManaged](<#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize, unsigned int Â flags )
     Allocates memory that will be automatically managed by the Unified Memory system.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemAllocPitch](<#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â pPitch, size_tÂ WidthInBytes, size_tÂ Height, unsigned int Â ElementSizeBytes )
     Allocates pitched device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemBatchDecompressAsync](<#group__CUDA__MEM_1g51e9452797573fa7597f2896182c8826>) ( [CUmemDecompressParams](<structCUmemDecompressParams.html#structCUmemDecompressParams> "Structure describing the parameters that compose a single decompression operation. ")*Â paramsArray, size_tÂ count, unsigned int Â flags, size_t*Â errorIndex, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â stream )
     Submit a batch of `count` independent decompression operations.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemFree](<#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )
     Frees device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemFreeHost](<#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c>) ( void*Â p )
     Frees page-locked host memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetAddressRange](<#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pbase, size_t*Â psize, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )
     Get information on memory allocations.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetHandleForAddressRange](<#group__CUDA__MEM_1g51e719462c04ee90a6b0f8b2a75fe031>) ( void*Â handle, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ size, [CUmemRangeHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g579b315f05d1e65a4f3de7da45013210>)Â handleType, unsigned long longÂ flags )
     Retrieve handle for an address range.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemGetInfo](<#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0>) ( size_t*Â free, size_t*Â total )
     Gets free and total memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemHostAlloc](<#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9>) ( void**Â pp, size_tÂ bytesize, unsigned int Â Flags )
     Allocates page-locked host memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemHostGetDevicePointer](<#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pdptr, void*Â p, unsigned int Â Flags )
     Passes back device pointer of mapped pinned memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemHostGetFlags](<#group__CUDA__MEM_1g42066246915fcb0400df2a17a851b35f>) ( unsigned int*Â pFlags, void*Â p )
     Passes back flags that were used for a pinned allocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemHostRegister](<#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223>) ( void*Â p, size_tÂ bytesize, unsigned int Â Flags )
     Registers an existing host memory range for use by CUDA.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemHostUnregister](<#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14>) ( void*Â p )
     Unregisters a memory range that was registered with cuMemHostRegister.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy](<#group__CUDA__MEM_1g8d0ff510f26d4b87bd3a51e731e7f698>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dst, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â src, size_tÂ ByteCount )
     Copies memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy2D](<#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27>) ( const [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>)*Â pCopy )
     Copies memory for 2D arrays.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy2DAsync](<#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274>) ( const [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>)*Â pCopy, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory for 2D arrays.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy2DUnaligned](<#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b>) ( const [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>)*Â pCopy )
     Copies memory for 2D arrays.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy3D](<#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df>) ( const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â pCopy )
     Copies memory for 3D arrays.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy3DAsync](<#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987>) ( const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â pCopy, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory for 3D arrays.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy3DBatchAsync](<#group__CUDA__MEM_1g97dd29d0e3490379a5cbdb21deb41f12>) ( size_tÂ numOps, CUDA_MEMCPY3D_BATCH_OP*Â opList, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Performs a batch of 3D memory copies asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy3DPeer](<#group__CUDA__MEM_1g11466fd70cde9329a4e16eb1f258c433>) ( const [CUDA_MEMCPY3D_PEER](<structCUDA__MEMCPY3D__PEER__v1.html#structCUDA__MEMCPY3D__PEER__v1>)*Â pCopy )
     Copies memory between contexts.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpy3DPeerAsync](<#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8>) ( const [CUDA_MEMCPY3D_PEER](<structCUDA__MEMCPY3D__PEER__v1.html#structCUDA__MEMCPY3D__PEER__v1>)*Â pCopy, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory between contexts asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyAsync](<#group__CUDA__MEM_1g5f26aaf5582ade791e5688727a178d78>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dst, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â src, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyAtoA](<#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount )
     Copies memory from Array to Array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyAtoD](<#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount )
     Copies memory from Array to Device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyAtoH](<#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4>) ( void*Â dstHost, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount )
     Copies memory from Array to Host.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyAtoHAsync](<#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01>) ( void*Â dstHost, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory from Array to Host.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyBatchAsync](<#group__CUDA__MEM_1g6f1ff58e3065df3eb4b573dba77ad31f>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dsts, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â srcs, size_t*Â sizes, size_tÂ count, [CUmemcpyAttributes](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1>)*Â attrs, size_t*Â attrsIdxs, size_tÂ numAttrs, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Performs a batch of memory copies asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyDtoA](<#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount )
     Copies memory from Device to Array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyDtoD](<#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount )
     Copies memory from Device to Device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyDtoDAsync](<#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory from Device to Device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyDtoH](<#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893>) ( void*Â dstHost, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount )
     Copies memory from Device to Host.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyDtoHAsync](<#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362>) ( void*Â dstHost, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory from Device to Host.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyHtoA](<#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, const void*Â srcHost, size_tÂ ByteCount )
     Copies memory from Host to Array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyHtoAAsync](<#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, const void*Â srcHost, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory from Host to Array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyHtoD](<#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, const void*Â srcHost, size_tÂ ByteCount )
     Copies memory from Host to Device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyHtoDAsync](<#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, const void*Â srcHost, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies memory from Host to Device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyPeer](<#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â dstContext, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â srcContext, size_tÂ ByteCount )
     Copies device memory between two contexts.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemcpyPeerAsync](<#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â dstContext, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â srcContext, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Copies device memory between two contexts asynchronously.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD16](<#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned shortÂ us, size_tÂ N )
     Initializes device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD16Async](<#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned shortÂ us, size_tÂ N, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Sets device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD2D16](<#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned shortÂ us, size_tÂ Width, size_tÂ Height )
     Initializes device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD2D16Async](<#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned shortÂ us, size_tÂ Width, size_tÂ Height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Sets device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD2D32](<#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned int Â ui, size_tÂ Width, size_tÂ Height )
     Initializes device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD2D32Async](<#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned int Â ui, size_tÂ Width, size_tÂ Height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Sets device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD2D8](<#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned char Â uc, size_tÂ Width, size_tÂ Height )
     Initializes device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD2D8Async](<#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned char Â uc, size_tÂ Width, size_tÂ Height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Sets device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD32](<#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned int Â ui, size_tÂ N )
     Initializes device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD32Async](<#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned int Â ui, size_tÂ N, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Sets device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD8](<#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned char Â uc, size_tÂ N )
     Initializes device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMemsetD8Async](<#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned char Â uc, size_tÂ N, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )
     Sets device memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMipmappedArrayCreate](<#group__CUDA__MEM_1ga5d2e311c7f9b0bc6d130af824a40bd3>) ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â pHandle, const [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>)*Â pMipmappedArrayDesc, unsigned int Â numMipmapLevels )
     Creates a CUDA mipmapped array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMipmappedArrayDestroy](<#group__CUDA__MEM_1ge0d7c768b6a6963c4d4bde5bbc74f0ad>) ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â hMipmappedArray )
     Destroys a CUDA mipmapped array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMipmappedArrayGetLevel](<#group__CUDA__MEM_1g82f276659f05be14820e99346b0f86b7>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pLevelArray, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â hMipmappedArray, unsigned int Â level )
     Gets a mipmap level of a CUDA mipmapped array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMipmappedArrayGetMemoryRequirements](<#group__CUDA__MEM_1g71b95168dd78c64cbca5b32b9cbf37e1>) ( [CUDA_ARRAY_MEMORY_REQUIREMENTS](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1>)*Â memoryRequirements, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â mipmap, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )
     Returns the memory requirements of a CUDA mipmapped array.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuMipmappedArrayGetSparseProperties](<#group__CUDA__MEM_1g55a16bd1780acb3cc94e8b88d5fe5e19>) ( [CUDA_ARRAY_SPARSE_PROPERTIES](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1>)*Â sparseProperties, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â mipmap )
     Returns the layout properties of a sparse CUDA mipmapped array.

### Enumerations

enum CUmemDecompressAlgorithm


######  Values

CU_MEM_DECOMPRESS_UNSUPPORTED = 0
    Decompression is unsupported.
CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE = 1<<0
    Deflate is supported.
CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY = 1<<1
    Snappy is supported.
CU_MEM_DECOMPRESS_ALGORITHM_LZ4 = 1<<2
    LZ4 is supported.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArray3DCreate ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pHandle, const [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>)*Â pAllocateArray )


Creates a 3D CUDA array.

######  Parameters

`pHandle`
    \- Returned array
`pAllocateArray`
    \- 3D array descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Creates a CUDA array according to the CUDA_ARRAY3D_DESCRIPTOR structure `pAllocateArray` and returns a handle to the new CUDA array in `*pHandle`. The CUDA_ARRAY3D_DESCRIPTOR is defined as:


    â    typedef struct {
                  unsigned int Width;
                  unsigned int Height;
                  unsigned int Depth;
                  [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) Format;
                  unsigned int NumChannels;
                  unsigned int Flags;
              } [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>);

where:

  * `Width`, `Height`, and `Depth` are the width, height, and depth of the CUDA array (in elements); the following types of CUDA arrays can be allocated:
    * A 1D array is allocated if `Height` and `Depth` extents are both zero.

    * A 2D array is allocated if only `Depth` extent is zero.

    * A 3D array is allocated if all three extents are non-zero.

    * A 1D layered CUDA array is allocated if only `Height` is zero and the [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flag is set. Each layer is a 1D array. The number of layers is determined by the depth extent.

    * A 2D layered CUDA array is allocated if all three extents are non-zero and the [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flag is set. Each layer is a 2D array. The number of layers is determined by the depth extent.

    * A cubemap CUDA array is allocated if all three extents are non-zero and the [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>) flag is set. `Width` must be equal to `Height`, and `Depth` must be six. A cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube. The order of the six layers in memory is the same as that listed in [CUarray_cubemap_face](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g012fda14b50e7db8798a340627c4c330>).

    * A cubemap layered CUDA array is allocated if all three extents are non-zero, and both, [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>) and [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flags are set. `Width` must be equal to `Height`, and `Depth` must be a multiple of six. A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists of a collection of cubemaps. The first six layers represent the first cubemap, the next six layers form the second cubemap, and so on.


  * Format specifies the format of the elements; [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) is defined as:

        â    typedef enum CUarray_format_enum {
                      [CU_AD_FORMAT_UNSIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e0af5f5a0ffa8e16a5c720364ccd5dac>) = 0x01,
                      [CU_AD_FORMAT_UNSIGNED_INT16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9d0f11e851e891af6f204cf05503ba525>) = 0x02,
                      [CU_AD_FORMAT_UNSIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f952b891ad5d4080db0fb2e23fe71614a0>) = 0x03,
                      [CU_AD_FORMAT_SIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9376b799ee12ce9e1de0c34cfa7839284>) = 0x08,
                      [CU_AD_FORMAT_SIGNED_INT16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f980864598b1579bd90fab79369072478f>) = 0x09,
                      [CU_AD_FORMAT_SIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f96db055c31d053bd1d5ebbaa98de2bad3>) = 0x0a,
                      [CU_AD_FORMAT_HALF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f995c97289b540ff36334722ec745f53a3>) = 0x10,
                      [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>) = 0x20,
                      [CU_AD_FORMAT_NV12](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f964889c93ccc518395eb985203735d40c>) = 0xb0,
                      [CU_AD_FORMAT_UNORM_INT8X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90dcb720ef3238f279ebd5a7eb7284137>) = 0xc0,
                      [CU_AD_FORMAT_UNORM_INT8X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e23c25eb679dd70676bd35b26041d21f>) = 0xc1,
                      [CU_AD_FORMAT_UNORM_INT8X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f939e6604652c4f7dfda35ef89bcf6a1c4>) = 0xc2,
                      [CU_AD_FORMAT_UNORM_INT16X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9a593cb744213ab457d4ebaa261879816>) = 0xc3,
                      [CU_AD_FORMAT_UNORM_INT16X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9fe334f0b162fd9ad3caad37a8c879d95>) = 0xc4,
                      [CU_AD_FORMAT_UNORM_INT16X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f965401cdeebbc53f7b02400ba14f940a4>) = 0xc5,
                      [CU_AD_FORMAT_SNORM_INT8X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f92943f83ded303df264a79ee11d1db0>) = 0xc6,
                      [CU_AD_FORMAT_SNORM_INT8X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8194990a6e17d78be0de66deffdf02f>) = 0xc7,
                      [CU_AD_FORMAT_SNORM_INT8X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9117a2e043203748187605ff8a71c2d1d>) = 0xc8,
                      [CU_AD_FORMAT_SNORM_INT16X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f95026e5e8783752bf8d3601dd4dbceb4c>) = 0xc9,
                      [CU_AD_FORMAT_SNORM_INT16X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f939f633274a07dbce442325c5d90bf294>) = 0xca,
                      [CU_AD_FORMAT_SNORM_INT16X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f99acc19038dc1e68170e485f739912d49>) = 0xcb,
                      [CU_AD_FORMAT_BC1_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9299d155257aa3c0b75634d9f9b1bfa72>) = 0x91,
                      [CU_AD_FORMAT_BC1_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9afa18b300eb91ff879532a55d5aa191b>) = 0x92,
                      [CU_AD_FORMAT_BC2_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f92b9cf00f8c6012ec679654c9f012a267>) = 0x93,
                      [CU_AD_FORMAT_BC2_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c38b5af7926b020202562d67ba7529c2>) = 0x94,
                      [CU_AD_FORMAT_BC3_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8473614347359cc74574899e2e65012>) = 0x95,
                      [CU_AD_FORMAT_BC3_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94d158239dd6c825b4bd383ed66625257>) = 0x96,
                      [CU_AD_FORMAT_BC4_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9fe7527dfa2576595eea7463a1140058c>) = 0x97,
                      [CU_AD_FORMAT_BC4_SNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90627520c6fc707d63e9d3c66d307eec6>) = 0x98,
                      [CU_AD_FORMAT_BC5_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94b524073942ab7460b68a98da955e59e>) = 0x99,
                      [CU_AD_FORMAT_BC5_SNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f907466b7b5b3d897a58fac1e9d2db163e>) = 0x9a,
                      [CU_AD_FORMAT_BC6H_UF16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94085af463b118d564873b8d275ac7912>) = 0x9b,
                      [CU_AD_FORMAT_BC6H_SF16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f938c1d137a2663d5ddca5ae6aa49f612e>) = 0x9c,
                      [CU_AD_FORMAT_BC7_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f03f9cbeee0911d3c77c08e6f5c7ff62>) = 0x9d,
                      [CU_AD_FORMAT_BC7_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9ad5b0e1cd964cbd46270223f35651677>) = 0x9e,
                      [CU_AD_FORMAT_P010](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f0efd5417115904eb086f1df0046582e>) = 0x9f,
                      [CU_AD_FORMAT_P016](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8511f36d0a010b8846c84309d8920d5>) = 0xa1,
                      [CU_AD_FORMAT_NV16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e41b4351cb805f35130636b0aafca609>) = 0xa2,
                      [CU_AD_FORMAT_P210](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9dc77f21be8b4ff4f23dcd450c3656409>) = 0xa3,
                      [CU_AD_FORMAT_P216](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9a42b3a04e2f30a93e50d7d68026f1ba9>) = 0xa4,
                      [CU_AD_FORMAT_YUY2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f961f757ee5f5c125b7be70e5b562826dc>) = 0xa5,
                      [CU_AD_FORMAT_Y210](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9796d73bbcb63216f7dd4cc4d8016b74c>) = 0xa6,
                      [CU_AD_FORMAT_Y216](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c2081981924fa204383f1ee05de74d8e>) = 0xa7,
                      [CU_AD_FORMAT_AYUV](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f99dd6d4cac84e541d2b1ad34b263bc1bc>) = 0xa8,
                      [CU_AD_FORMAT_Y410](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f91917374a5915ee6a5e1ed23c57f43b75>) = 0xa9,
                      [CU_AD_FORMAT_Y416](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f941bdcafb69e249176af2e1cc5d6178be>) = 0xb1,
                      [CU_AD_FORMAT_Y444_PLANAR8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f93af0d614c7c240194c402b6ca9b4909f>) = 0xb2,
                      [CU_AD_FORMAT_Y444_PLANAR10](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f916b3c04a1fba1562d548d4504f06a7aa>) = 0xb3,
                      [CU_AD_FORMAT_YUV444_8bit_SemiPlanar](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f945b7c15d0c8a42d569b20509e7e54e1d>) = 0xb4,
                      [CU_AD_FORMAT_YUV444_16bit_SemiPlanar](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9cb653da3339b76b267a6fa8085513017>) = 0xb5,
                      [CU_AD_FORMAT_UNORM_INT_101010_2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f93b3828154e807c69a6e0c7e0d54d31ea>) = 0x50,
                  } [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>);


  * `NumChannels` specifies the number of packed components per CUDA array element; it may be 1, 2, or 4;


  * Flags may be set to
    * [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) to enable creation of layered CUDA arrays. If this flag is set, `Depth` specifies the number of layers, not the depth of a 3D array.

    * [CUDA_ARRAY3D_SURFACE_LDST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7287c43cacf1ed05865d6bcad1a23cd9>) to enable surface references to be bound to the CUDA array. If this flag is not set, [cuSurfRefSetArray](<group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED_1g68abcde159fa897b1dfb23387926dd66> "Sets the CUDA array for a surface reference.") will fail when attempting to bind the CUDA array to a surface reference.

    * [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>) to enable creation of cubemaps. If this flag is set, `Width` must be equal to `Height`, and `Depth` must be six. If the [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flag is also set, then `Depth` must be a multiple of six.

    * [CUDA_ARRAY3D_TEXTURE_GATHER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0c929c92a91f4d9f9f49bae0131a6ccf>) to indicate that the CUDA array will be used for texture gather. Texture gather can only be performed on 2D CUDA arrays.


`Width`, `Height` and `Depth` must meet certain size requirements as listed in the following table. All values are specified in elements. Note that for brevity's sake, the full name of the device attribute is not specified. For ex., TEXTURE1D_WIDTH refers to the device attribute [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a31f7318312b520cd5bc19eb97659e8215>).

Note that 2D CUDA arrays have different size requirements if the [CUDA_ARRAY3D_TEXTURE_GATHER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0c929c92a91f4d9f9f49bae0131a6ccf>) flag is set. `Width` and `Height` must not be greater than [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3a3d02a5e777b952280c1f8b4fac477ef>) and [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a39ceeac801dbf34671de63cc7ca383935>) respectively, in that case.

**CUDA array type** |  **Valid extents that must always be met {(width range in elements), (height range), (depth range)}** |  **Valid extents with CUDA_ARRAY3D_SURFACE_LDST set {(width range in elements), (height range), (depth range)}**
---|---|---
1D  |  { (1,TEXTURE1D_WIDTH), 0, 0 }  |  { (1,SURFACE1D_WIDTH), 0, 0 }
2D  |  { (1,TEXTURE2D_WIDTH), (1,TEXTURE2D_HEIGHT), 0 }  |  { (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }
3D  |  { (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) } OR { (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE), (1,TEXTURE3D_DEPTH_ALTERNATE) }  |  { (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT), (1,SURFACE3D_DEPTH) }
1D Layered  |  { (1,TEXTURE1D_LAYERED_WIDTH), 0, (1,TEXTURE1D_LAYERED_LAYERS) }  |  { (1,SURFACE1D_LAYERED_WIDTH), 0, (1,SURFACE1D_LAYERED_LAYERS) }
2D Layered  |  { (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT), (1,TEXTURE2D_LAYERED_LAYERS) }  |  { (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT), (1,SURFACE2D_LAYERED_LAYERS) }
Cubemap  |  { (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }  |  { (1,SURFACECUBEMAP_WIDTH), (1,SURFACECUBEMAP_WIDTH), 6 }
Cubemap Layered  |  { (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_LAYERS) }  |  { (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_LAYERS) }

Here are examples of CUDA array descriptions:

Description for a CUDA array of 2048 floats:


    â    [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>) desc;
              desc.[Format](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_196078e824c96e0fe2d6d8291700b60c7>) = [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>);
              desc.[NumChannels](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_19a3c02a716777e5e74d19b35456cc49c>) = 1;
              desc.[Width](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_13de201be4776cf763f87f87d671f27ba>) = 2048;
              desc.[Height](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_104a3de12856f0f8dffb39c79a172d824>) = 0;
              desc.[Depth](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_1404d0c4db6b39d54af70c4e312ad9ea9>) = 0;

Description for a 64 x 64 CUDA array of floats:


    â    [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>) desc;
              desc.[Format](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_196078e824c96e0fe2d6d8291700b60c7>) = [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>);
              desc.[NumChannels](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_19a3c02a716777e5e74d19b35456cc49c>) = 1;
              desc.[Width](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_13de201be4776cf763f87f87d671f27ba>) = 64;
              desc.[Height](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_104a3de12856f0f8dffb39c79a172d824>) = 64;
              desc.[Depth](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_1404d0c4db6b39d54af70c4e312ad9ea9>) = 0;

Description for a `width` x `height` x `depth` CUDA array of 64-bit, 4x16-bit float16's:


    â    [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>) desc;
              desc.[Format](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_196078e824c96e0fe2d6d8291700b60c7>) = [CU_AD_FORMAT_HALF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f995c97289b540ff36334722ec745f53a3>);
              desc.[NumChannels](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_19a3c02a716777e5e74d19b35456cc49c>) = 4;
              desc.[Width](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_13de201be4776cf763f87f87d671f27ba>) = width;
              desc.[Height](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_104a3de12856f0f8dffb39c79a172d824>) = height;
              desc.[Depth](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2_1404d0c4db6b39d54af70c4e312ad9ea9>) = depth;

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMalloc3DArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g948143cf2423a072ac6a31fb635efd88>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArray3DGetDescriptor ( [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>)*Â pArrayDescriptor, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray )


Get a 3D CUDA array descriptor.

######  Parameters

`pArrayDescriptor`
    \- Returned 3D array descriptor
`hArray`
    \- 3D array to get descriptor of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Returns in `*pArrayDescriptor` a descriptor containing information on the format and dimensions of the CUDA array `hArray`. It is useful for subroutines that have been passed a CUDA array, but need to know the CUDA array parameters for validation or other purposes.

This function may be called on 1D and 2D arrays, in which case the `Height` and/or `Depth` members of the descriptor struct will be set to 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaArrayGetInfo](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g373dacf191566b0bf5e5b807517b6bf9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArrayCreate ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pHandle, const [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>)*Â pAllocateArray )


Creates a 1D or 2D CUDA array.

######  Parameters

`pHandle`
    \- Returned array
`pAllocateArray`
    \- Array descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Creates a CUDA array according to the CUDA_ARRAY_DESCRIPTOR structure `pAllocateArray` and returns a handle to the new CUDA array in `*pHandle`. The CUDA_ARRAY_DESCRIPTOR is defined as:


    â    typedef struct {
                  unsigned int Width;
                  unsigned int Height;
                  [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) Format;
                  unsigned int NumChannels;
              } [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>);

where:

  * `Width`, and `Height` are the width, and height of the CUDA array (in elements); the CUDA array is one-dimensional if height is 0, two-dimensional otherwise;

  * Format specifies the format of the elements; [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) is defined as:

        â    typedef enum CUarray_format_enum {
                      [CU_AD_FORMAT_UNSIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e0af5f5a0ffa8e16a5c720364ccd5dac>) = 0x01,
                      [CU_AD_FORMAT_UNSIGNED_INT16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9d0f11e851e891af6f204cf05503ba525>) = 0x02,
                      [CU_AD_FORMAT_UNSIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f952b891ad5d4080db0fb2e23fe71614a0>) = 0x03,
                      [CU_AD_FORMAT_SIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9376b799ee12ce9e1de0c34cfa7839284>) = 0x08,
                      [CU_AD_FORMAT_SIGNED_INT16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f980864598b1579bd90fab79369072478f>) = 0x09,
                      [CU_AD_FORMAT_SIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f96db055c31d053bd1d5ebbaa98de2bad3>) = 0x0a,
                      [CU_AD_FORMAT_HALF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f995c97289b540ff36334722ec745f53a3>) = 0x10,
                      [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>) = 0x20,
                      [CU_AD_FORMAT_NV12](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f964889c93ccc518395eb985203735d40c>) = 0xb0,
                      [CU_AD_FORMAT_UNORM_INT8X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90dcb720ef3238f279ebd5a7eb7284137>) = 0xc0,
                      [CU_AD_FORMAT_UNORM_INT8X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e23c25eb679dd70676bd35b26041d21f>) = 0xc1,
                      [CU_AD_FORMAT_UNORM_INT8X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f939e6604652c4f7dfda35ef89bcf6a1c4>) = 0xc2,
                      [CU_AD_FORMAT_UNORM_INT16X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9a593cb744213ab457d4ebaa261879816>) = 0xc3,
                      [CU_AD_FORMAT_UNORM_INT16X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9fe334f0b162fd9ad3caad37a8c879d95>) = 0xc4,
                      [CU_AD_FORMAT_UNORM_INT16X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f965401cdeebbc53f7b02400ba14f940a4>) = 0xc5,
                      [CU_AD_FORMAT_SNORM_INT8X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f92943f83ded303df264a79ee11d1db0>) = 0xc6,
                      [CU_AD_FORMAT_SNORM_INT8X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8194990a6e17d78be0de66deffdf02f>) = 0xc7,
                      [CU_AD_FORMAT_SNORM_INT8X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9117a2e043203748187605ff8a71c2d1d>) = 0xc8,
                      [CU_AD_FORMAT_SNORM_INT16X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f95026e5e8783752bf8d3601dd4dbceb4c>) = 0xc9,
                      [CU_AD_FORMAT_SNORM_INT16X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f939f633274a07dbce442325c5d90bf294>) = 0xca,
                      [CU_AD_FORMAT_SNORM_INT16X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f99acc19038dc1e68170e485f739912d49>) = 0xcb,
                      [CU_AD_FORMAT_BC1_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9299d155257aa3c0b75634d9f9b1bfa72>) = 0x91,
                      [CU_AD_FORMAT_BC1_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9afa18b300eb91ff879532a55d5aa191b>) = 0x92,
                      [CU_AD_FORMAT_BC2_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f92b9cf00f8c6012ec679654c9f012a267>) = 0x93,
                      [CU_AD_FORMAT_BC2_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c38b5af7926b020202562d67ba7529c2>) = 0x94,
                      [CU_AD_FORMAT_BC3_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8473614347359cc74574899e2e65012>) = 0x95,
                      [CU_AD_FORMAT_BC3_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94d158239dd6c825b4bd383ed66625257>) = 0x96,
                      [CU_AD_FORMAT_BC4_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9fe7527dfa2576595eea7463a1140058c>) = 0x97,
                      [CU_AD_FORMAT_BC4_SNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90627520c6fc707d63e9d3c66d307eec6>) = 0x98,
                      [CU_AD_FORMAT_BC5_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94b524073942ab7460b68a98da955e59e>) = 0x99,
                      [CU_AD_FORMAT_BC5_SNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f907466b7b5b3d897a58fac1e9d2db163e>) = 0x9a,
                      [CU_AD_FORMAT_BC6H_UF16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94085af463b118d564873b8d275ac7912>) = 0x9b,
                      [CU_AD_FORMAT_BC6H_SF16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f938c1d137a2663d5ddca5ae6aa49f612e>) = 0x9c,
                      [CU_AD_FORMAT_BC7_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f03f9cbeee0911d3c77c08e6f5c7ff62>) = 0x9d,
                      [CU_AD_FORMAT_BC7_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9ad5b0e1cd964cbd46270223f35651677>) = 0x9e,
                      [CU_AD_FORMAT_P010](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f0efd5417115904eb086f1df0046582e>) = 0x9f,
                      [CU_AD_FORMAT_P016](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8511f36d0a010b8846c84309d8920d5>) = 0xa1,
                      [CU_AD_FORMAT_NV16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e41b4351cb805f35130636b0aafca609>) = 0xa2,
                      [CU_AD_FORMAT_P210](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9dc77f21be8b4ff4f23dcd450c3656409>) = 0xa3,
                      [CU_AD_FORMAT_P216](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9a42b3a04e2f30a93e50d7d68026f1ba9>) = 0xa4,
                      [CU_AD_FORMAT_YUY2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f961f757ee5f5c125b7be70e5b562826dc>) = 0xa5,
                      [CU_AD_FORMAT_Y210](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9796d73bbcb63216f7dd4cc4d8016b74c>) = 0xa6,
                      [CU_AD_FORMAT_Y216](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c2081981924fa204383f1ee05de74d8e>) = 0xa7,
                      [CU_AD_FORMAT_AYUV](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f99dd6d4cac84e541d2b1ad34b263bc1bc>) = 0xa8,
                      [CU_AD_FORMAT_Y410](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f91917374a5915ee6a5e1ed23c57f43b75>) = 0xa9,
                      [CU_AD_FORMAT_Y416](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f941bdcafb69e249176af2e1cc5d6178be>) = 0xb1,
                      [CU_AD_FORMAT_Y444_PLANAR8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f93af0d614c7c240194c402b6ca9b4909f>) = 0xb2,
                      [CU_AD_FORMAT_Y444_PLANAR10](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f916b3c04a1fba1562d548d4504f06a7aa>) = 0xb3,
                      [CU_AD_FORMAT_YUV444_8bit_SemiPlanar](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f945b7c15d0c8a42d569b20509e7e54e1d>) = 0xb4,
                      [CU_AD_FORMAT_YUV444_16bit_SemiPlanar](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9cb653da3339b76b267a6fa8085513017>) = 0xb5,
                      [CU_AD_FORMAT_UNORM_INT_101010_2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f93b3828154e807c69a6e0c7e0d54d31ea>) = 0x50,
                      [CU_AD_FORMAT_UINT8_PACKED_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c3e98982827e44204ed4a4d41031c135>) = 0x51,
                      [CU_AD_FORMAT_UINT8_PACKED_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90cb5a5812940939b5f0eb0242a2146e7>) = 0x52,
                      [CU_AD_FORMAT_UINT8_SEMIPLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f212e89e226ee1d69ca4a47fba3c39c3>) = 0x53,
                      [CU_AD_FORMAT_UINT16_SEMIPLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e076ab1291241ef4c6c149b23321e1b5>) = 0x54,
                      [CU_AD_FORMAT_UINT8_SEMIPLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f959c089e74d94f90973118ef287d4f352>) = 0x55,
                      [CU_AD_FORMAT_UINT16_SEMIPLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f97f759c17c105bfa6fb486502fad3705e>) = 0x56,
                      [CU_AD_FORMAT_UINT8_SEMIPLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90548d7ee27a7f9401064a4d3b3dfc528>) = 0x57,
                      [CU_AD_FORMAT_UINT16_SEMIPLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90a4038e90a78ae9bf495d043c39a9e29>) = 0x58,
                      [CU_AD_FORMAT_UINT8_PLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9d0f3182f705a6615ac5299de395cace8>) = 0x59,
                      [CU_AD_FORMAT_UINT16_PLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f97a9888731c72732a6880a18ef2f082cb>) = 0x5a,
                      [CU_AD_FORMAT_UINT8_PLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f932bb01870f63cd2d9949ccb2ea235ef1>) = 0x5b,
                      [CU_AD_FORMAT_UINT16_PLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f97a9a4abfb8ce20c04ddb47925cdf3752>) = 0x5c,
                      [CU_AD_FORMAT_UINT8_PLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9922d032a80ac1229a05dae35c18c4b2e>) = 0x5d,
                      [CU_AD_FORMAT_UINT16_PLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98ff25a5b7ff451a608b2ecca340b1f71>) = 0x5e,
                 } [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>);

  * `NumChannels` specifies the number of packed components per CUDA array element; it may be 1, 2, or 4;


Here are examples of CUDA array descriptions:

Description for a CUDA array of 2048 floats:


    â    [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>) desc;
              desc.[Format](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_19ea7ac9a60d1e85a2b45a3b7287fb5e9>) = [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>);
              desc.[NumChannels](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_11c4f64c9b3497ab7d315b4fb85bc468d>) = 1;
              desc.[Width](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_140015fb3fed92224b92650450c3ea2f0>) = 2048;
              desc.[Height](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_1796dde1623e8bff79f764020d4b8f798>) = 1;

Description for a 64 x 64 CUDA array of floats:


    â    [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>) desc;
              desc.[Format](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_19ea7ac9a60d1e85a2b45a3b7287fb5e9>) = [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>);
              desc.[NumChannels](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_11c4f64c9b3497ab7d315b4fb85bc468d>) = 1;
              desc.[Width](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_140015fb3fed92224b92650450c3ea2f0>) = 64;
              desc.[Height](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_1796dde1623e8bff79f764020d4b8f798>) = 64;

Description for a `width` x `height` CUDA array of 64-bit, 4x16-bit float16's:


    â    [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>) desc;
              desc.[Format](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_19ea7ac9a60d1e85a2b45a3b7287fb5e9>) = [CU_AD_FORMAT_HALF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f995c97289b540ff36334722ec745f53a3>);
              desc.[NumChannels](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_11c4f64c9b3497ab7d315b4fb85bc468d>) = 4;
              desc.[Width](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_140015fb3fed92224b92650450c3ea2f0>) = width;
              desc.[Height](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_1796dde1623e8bff79f764020d4b8f798>) = height;

Description for a `width` x `height` CUDA array of 16-bit elements, each of which is two 8-bit unsigned chars:


    â    [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>) arrayDesc;
              desc.[Format](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_19ea7ac9a60d1e85a2b45a3b7287fb5e9>) = [CU_AD_FORMAT_UNSIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e0af5f5a0ffa8e16a5c720364ccd5dac>);
              desc.[NumChannels](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_11c4f64c9b3497ab7d315b4fb85bc468d>) = 2;
              desc.[Width](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_140015fb3fed92224b92650450c3ea2f0>) = width;
              desc.[Height](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2_1796dde1623e8bff79f764020d4b8f798>) = height;

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMallocArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g6728eb7dc25f332f50bdb16a19620d3d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArrayDestroy ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray )


Destroys a CUDA array.

######  Parameters

`hArray`
    \- Array to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ARRAY_IS_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b668b10d56232c51b67db40516cc6b5b>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Destroys the CUDA array `hArray`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaFreeArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g1b553f5f4806d67525230ac305d50900>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArrayGetDescriptor ( [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>)*Â pArrayDescriptor, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray )


Get a 1D or 2D CUDA array descriptor.

######  Parameters

`pArrayDescriptor`
    \- Returned array descriptor
`hArray`
    \- Array to get descriptor of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Returns in `*pArrayDescriptor` a descriptor containing information on the format and dimensions of the CUDA array `hArray`. It is useful for subroutines that have been passed a CUDA array, but need to know the CUDA array parameters for validation or other purposes.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaArrayGetInfo](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g373dacf191566b0bf5e5b807517b6bf9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArrayGetMemoryRequirements ( [CUDA_ARRAY_MEMORY_REQUIREMENTS](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1>)*Â memoryRequirements, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â array, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )


Returns the memory requirements of a CUDA array.

######  Parameters

`memoryRequirements`
    \- Pointer to CUDA_ARRAY_MEMORY_REQUIREMENTS
`array`
    \- CUDA array to get the memory requirements of
`device`
    \- Device to get the memory requirements for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the memory requirements of a CUDA array in `memoryRequirements` If the CUDA array is not allocated with flag [CUDA_ARRAY3D_DEFERRED_MAPPING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g854c29dbc47d04a4e42863cb87487d55>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned.

The returned value in [CUDA_ARRAY_MEMORY_REQUIREMENTS::size](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1_17a2851735a1d2c11af797f01b1d4969e>) represents the total size of the CUDA array. The returned value in [CUDA_ARRAY_MEMORY_REQUIREMENTS::alignment](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1_135c6c8106451b6313d1dffe9a28af755>) represents the alignment necessary for mapping the CUDA array.

**See also:**

[cuMipmappedArrayGetMemoryRequirements](<group__CUDA__MEM.html#group__CUDA__MEM_1g71b95168dd78c64cbca5b32b9cbf37e1> "Returns the memory requirements of a CUDA mipmapped array."), [cuMemMapArrayAsync](<group__CUDA__VA.html#group__CUDA__VA_1g5dc41a62a9feb68f2e943b438c83e5ab> "Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArrayGetPlane ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pPlaneArray, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray, unsigned int Â planeIdx )


Gets a CUDA array plane from a CUDA array.

######  Parameters

`pPlaneArray`
    \- Returned CUDA array referenced by the `planeIdx`
`hArray`
    \- Multiplanar CUDA array
`planeIdx`
    \- Plane index

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Returns in `pPlaneArray` a CUDA array that represents a single format plane of the CUDA array `hArray`.

If `planeIdx` is greater than the maximum number of planes in this array or if the array does not have a multi-planar format e.g: [CU_AD_FORMAT_NV12](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f964889c93ccc518395eb985203735d40c>), then [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note that if the `hArray` has format [CU_AD_FORMAT_NV12](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f964889c93ccc518395eb985203735d40c>), then passing in 0 for `planeIdx` returns a CUDA array of the same size as `hArray` but with one channel and [CU_AD_FORMAT_UNSIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e0af5f5a0ffa8e16a5c720364ccd5dac>) as its format. If 1 is passed for `planeIdx`, then the returned CUDA array has half the height and width of `hArray` with two channels and [CU_AD_FORMAT_UNSIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e0af5f5a0ffa8e16a5c720364ccd5dac>) as its format.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cudaArrayGetPlane](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9a851663a2b9f222b549c727adc0e079>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuArrayGetSparseProperties ( [CUDA_ARRAY_SPARSE_PROPERTIES](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1>)*Â sparseProperties, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â array )


Returns the layout properties of a sparse CUDA array.

######  Parameters

`sparseProperties`
    \- Pointer to CUDA_ARRAY_SPARSE_PROPERTIES
`array`
    \- CUDA array to get the sparse properties of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the layout properties of a sparse CUDA array in `sparseProperties` If the CUDA array is not allocated with flag [CUDA_ARRAY3D_SPARSE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8e13c9d3ef98d1f3dce95901a115abc2>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned.

If the returned value in [CUDA_ARRAY_SPARSE_PROPERTIES::flags](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_10e842bb64091fa47809112c700cb5f0a>) contains [CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0dcf4ba7e64caa5c1aa4e88caa7f659a>), then [CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1895ecb42681678271b0edba05bf1dcd9>) represents the total size of the array. Otherwise, it will be zero. Also, the returned value in [CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1edd0cca8fad1fcbb1789d537edd7e6b6>) is always zero. Note that the `array` must have been allocated using [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array.") or [cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."). For CUDA arrays obtained using [cuMipmappedArrayGetLevel](<group__CUDA__MEM.html#group__CUDA__MEM_1g82f276659f05be14820e99346b0f86b7> "Gets a mipmap level of a CUDA mipmapped array."), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned. Instead, [cuMipmappedArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1g55a16bd1780acb3cc94e8b88d5fe5e19> "Returns the layout properties of a sparse CUDA mipmapped array.") must be used to obtain the sparse properties of the entire CUDA mipmapped array to which `array` belongs to.

**See also:**

[cuMipmappedArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1g55a16bd1780acb3cc94e8b88d5fe5e19> "Returns the layout properties of a sparse CUDA mipmapped array."), [cuMemMapArrayAsync](<group__CUDA__VA.html#group__CUDA__VA_1g5dc41a62a9feb68f2e943b438c83e5ab> "Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetByPCIBusId ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â dev, const char*Â pciBusId )


Returns a handle to a compute device.

######  Parameters

`dev`
    \- Returned device handle
`pciBusId`
    \- String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where `domain`, `bus`, `device`, and `function` are all hexadecimal values

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*device` a device handle given a PCI bus ID string.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetPCIBusId](<group__CUDA__MEM.html#group__CUDA__MEM_1g85295e7d9745ab8f0aa80dd1e172acfc> "Returns a PCI Bus Id string for the device."), [cudaDeviceGetByPCIBusId](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g65f57fb8d0981ca03f6f9b20031c3e5d>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetPCIBusId ( char*Â pciBusId, int Â len, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns a PCI Bus Id string for the device.

######  Parameters

`pciBusId`
    \- Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where `domain`, `bus`, `device`, and `function` are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator.
`len`
    \- Maximum length of string to store in `name`
`dev`
    \- Device to get identifier string for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns an ASCII string identifying the device `dev` in the NULL-terminated string pointed to by `pciBusId`. `len` specifies the maximum length of the string that may be returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetByPCIBusId](<group__CUDA__MEM.html#group__CUDA__MEM_1ga89cd3fa06334ba7853ed1232c5ebe2a> "Returns a handle to a compute device."), [cudaDeviceGetPCIBusId](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gea264dad3d8c4898e0b82213c0253def>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceRegisterAsyncNotification ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUasyncCallback](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g466d4731f270b66441a355ddb2c84777>)Â callbackFunc, void*Â userData, [CUasyncCallbackHandle](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0ff5c3d4645d51b02b6d11b8b0c228c5>)*Â callback )


Registers a callback function to receive async notifications.

######  Parameters

`device`
    \- The device on which to register the callback
`callbackFunc`
    \- The function to register as a callback
`userData`
    \- A generic pointer to user data. This is passed into the callback function.
`callback`
    \- A handle representing the registered callback instance

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Registers `callbackFunc` to receive async notifications.

The `userData` parameter is passed to the callback function at async notification time. Likewise, `callback` is also passed to the callback function to distinguish between multiple registered callbacks.

The callback function being registered should be designed to return quickly (~10ms). Any long running tasks should be queued for execution on an application thread.

Callbacks may not call cuDeviceRegisterAsyncNotification or cuDeviceUnregisterAsyncNotification. Doing so will result in [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>). Async notification callbacks execute in an undefined order and may be serialized.

Returns in `*callback` a handle representing the registered callback instance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceUnregisterAsyncNotification](<group__CUDA__MEM.html#group__CUDA__MEM_1gc0ae698fd18cbc2c395c9140e28e83ca> "Unregisters an async notification callback.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceUnregisterAsyncNotification ( [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device, [CUasyncCallbackHandle](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0ff5c3d4645d51b02b6d11b8b0c228c5>)Â callback )


Unregisters an async notification callback.

######  Parameters

`device`
    \- The device from which to remove `callback`.
`callback`
    \- The callback instance to unregister from receiving async notifications.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Unregisters `callback` so that the corresponding callback function will stop receiving async notifications.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceRegisterAsyncNotification](<group__CUDA__MEM.html#group__CUDA__MEM_1g4325f3e53f7817c93b37f12da91ed199> "Registers a callback function to receive async notifications.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuIpcCloseMemHandle ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )


Attempts to close memory mapped with cuIpcOpenMemHandle.

######  Parameters

`dptr`
    \- Device pointer returned by [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.")

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Decrements the reference count of the memory returned by [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") by 1. When the reference count reaches 0, this API unmaps the memory. The original allocation in the exporting process as well as imported mappings in other processes will be unaffected.

Any resources used to enable peer access will be freed if this is the last mapping using them.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cuapiDeviceGetAttribute with [CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3348f3c29378467df5114e2e738c4b380>)

**See also:**

[cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuIpcGetEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c> "Gets an interprocess handle for a previously allocated event."), [cuIpcOpenEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e> "Opens an interprocess event handle for use in the current process."), [cuIpcGetMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1> "Gets an interprocess memory handle for an existing device memory allocation."), [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cudaIpcCloseMemHandle](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g02bb3632b5d223db6acae5f8744e2c91>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuIpcGetEventHandle ( [CUipcEventHandle](<structCUipcEventHandle__v1.html#structCUipcEventHandle__v1>)*Â pHandle, [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)Â event )


Gets an interprocess handle for a previously allocated event.

######  Parameters

`pHandle`
    \- Pointer to a user allocated CUipcEventHandle in which to return the opaque event handle
`event`
    \- Event allocated with [CU_EVENT_INTERPROCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629adab662356d24cf59f3d7de07c3cd52e>) and [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>) flags.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Takes as input a previously allocated event. This event must have been created with the [CU_EVENT_INTERPROCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629adab662356d24cf59f3d7de07c3cd52e>) and [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>) flags set. This opaque handle may be copied into other processes and opened with [cuIpcOpenEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e> "Opens an interprocess event handle for use in the current process.") to allow efficient hardware synchronization between GPU work in different processes.

After the event has been opened in the importing process, [cuEventRecord](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1> "Records an event."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event.") and [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status.") may be used in either process. Performing operations on the imported event after the exported event has been freed with [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event.") will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling [cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.") with [CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3348f3c29378467df5114e2e738c4b380>)

**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuIpcOpenEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e> "Opens an interprocess event handle for use in the current process."), [cuIpcGetMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1> "Gets an interprocess memory handle for an existing device memory allocation."), [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cuIpcCloseMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e> "Attempts to close memory mapped with cuIpcOpenMemHandle."), [cudaIpcGetEventHandle](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g89a3abe1e9a11d08c665176669109784>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuIpcGetMemHandle ( [CUipcMemHandle](<structCUipcMemHandle__v1.html#structCUipcMemHandle__v1>)*Â pHandle, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )


Gets an interprocess memory handle for an existing device memory allocation.

######  Parameters

`pHandle`
    \- Pointer to user allocated CUipcMemHandle to return the handle in.
`dptr`
    \- Base pointer to previously allocated device memory

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Takes a pointer to the base of an existing device memory allocation created with [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") and exports it for use in another process. This is a lightweight operation and may be called multiple times on an allocation without adverse effects.

If a region of memory is freed with [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory.") and a subsequent call to [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") returns memory with the same device address, [cuIpcGetMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1> "Gets an interprocess memory handle for an existing device memory allocation.") will return a unique handle for the new memory.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cuapiDeviceGetAttribute with [CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3348f3c29378467df5114e2e738c4b380>)

**See also:**

[cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuIpcGetEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c> "Gets an interprocess handle for a previously allocated event."), [cuIpcOpenEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e> "Opens an interprocess event handle for use in the current process."), [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cuIpcCloseMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e> "Attempts to close memory mapped with cuIpcOpenMemHandle."), [cudaIpcGetMemHandle](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g8a37f7dfafaca652391d0758b3667539>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuIpcOpenEventHandle ( [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>)*Â phEvent, [CUipcEventHandle](<structCUipcEventHandle__v1.html#structCUipcEventHandle__v1>)Â handle )


Opens an interprocess event handle for use in the current process.

######  Parameters

`phEvent`
    \- Returns the imported event
`handle`
    \- Interprocess handle to open

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>), [CUDA_ERROR_PEER_ACCESS_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d60abcaa3f2710f961db8c383bb95cae>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Opens an interprocess event handle exported from another process with [cuIpcGetEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c> "Gets an interprocess handle for a previously allocated event."). This function returns a [CUevent](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d740185cf0953636d4ae37f68d7559b>) that behaves like a locally created event with the [CU_EVENT_DISABLE_TIMING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5ae04079c671c8e659a3a27c7b23f629daa5463f64794c10b78c603d23c0bff2>) flag specified. This event must be freed with [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event.").

Performing operations on the imported event after the exported event has been freed with [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event.") will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cuapiDeviceGetAttribute with [CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3348f3c29378467df5114e2e738c4b380>)

**See also:**

[cuEventCreate](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db> "Creates an event."), [cuEventDestroy](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef> "Destroys an event."), [cuEventSynchronize](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c> "Waits for an event to complete."), [cuEventQuery](<group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef> "Queries an event's status."), [cuStreamWaitEvent](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f> "Make a compute stream wait on an event."), [cuIpcGetEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c> "Gets an interprocess handle for a previously allocated event."), [cuIpcGetMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1> "Gets an interprocess memory handle for an existing device memory allocation."), [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process."), [cuIpcCloseMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e> "Attempts to close memory mapped with cuIpcOpenMemHandle."), [cudaIpcOpenEventHandle](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g9691446ab0aec1d6e528357387ed87b2>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuIpcOpenMemHandle ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pdptr, [CUipcMemHandle](<structCUipcMemHandle__v1.html#structCUipcMemHandle__v1>)Â handle, unsigned int Â Flags )


Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.

######  Parameters

`pdptr`
    \- Returned device pointer
`handle`
    \- CUipcMemHandle to open
`Flags`
    \- Flags for this operation. Must be specified as [CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg871d09eefd2aacd3b10fe4f5f23b1a32567565e36fc87a2180109170a0501947>)

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_MAP_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b9a95891afee8e479ca2e89595b51a2f>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_TOO_MANY_PEERS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9168ef870793a31ef4cdd7cb6e279b34a>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Maps memory exported from another process with [cuIpcGetMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1> "Gets an interprocess memory handle for an existing device memory allocation.") into the current device address space. For contexts on different devices [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") can attempt to enable peer access between the devices as if the user called [cuCtxEnablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context."). This behavior is controlled by the [CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg871d09eefd2aacd3b10fe4f5f23b1a32567565e36fc87a2180109170a0501947>) flag. [cuDeviceCanAccessPeer](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e> "Queries if a device may directly access a peer device's memory.") can determine if a mapping is possible.

Contexts that may open CUipcMemHandles are restricted in the following way. CUipcMemHandles from each [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) in a given process may only be opened by one [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>) per [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) per other process.

If the memory handle has already been opened by the current context, the reference count on the handle is incremented by 1 and the existing device pointer is returned.

Memory returned from [cuIpcOpenMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1ga8bd126fcff919a0c996b7640f197b79> "Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.") must be freed with [cuIpcCloseMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e> "Attempts to close memory mapped with cuIpcOpenMemHandle.").

Calling [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory.") on an exported memory region before calling [cuIpcCloseMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e> "Attempts to close memory mapped with cuIpcOpenMemHandle.") in the importing context will result in undefined behavior.

IPC functionality is restricted to devices with support for unified addressing on Linux and Windows operating systems. IPC functionality on Windows is supported for compatibility purposes but not recommended as it comes with performance cost. Users can test their device for IPC functionality by calling cuapiDeviceGetAttribute with [CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3348f3c29378467df5114e2e738c4b380>)

Note:

No guarantees are made about the address returned in `*pdptr`. In particular, multiple processes may not receive the same address for the same `handle`.

**See also:**

[cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuIpcGetEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gea02eadd12483de5305878b13288a86c> "Gets an interprocess handle for a previously allocated event."), [cuIpcOpenEventHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gf1d525918b6c643b99ca8c8e42e36c2e> "Opens an interprocess event handle for use in the current process."), [cuIpcGetMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1g6f1b5be767b275f016523b2ac49ebec1> "Gets an interprocess memory handle for an existing device memory allocation."), [cuIpcCloseMemHandle](<group__CUDA__MEM.html#group__CUDA__MEM_1gd6f5d5bcf6376c6853b64635b0157b9e> "Attempts to close memory mapped with cuIpcOpenMemHandle."), [cuCtxEnablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context."), [cuDeviceCanAccessPeer](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e> "Queries if a device may directly access a peer device's memory."), [cudaIpcOpenMemHandle](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g01050a29fefde385b1042081ada4cde9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAlloc ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize )


Allocates device memory.

######  Parameters

`dptr`
    \- Returned device pointer
`bytesize`
    \- Requested allocation size in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Allocates `bytesize` bytes of linear memory on the device and returns in `*dptr` a pointer to the allocated memory. The allocated memory is suitably aligned for any kind of variable. The memory is not cleared. If `bytesize` is 0, [cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMalloc](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAllocHost ( void**Â pp, size_tÂ bytesize )


Allocates page-locked host memory.

######  Parameters

`pp`
    \- Returned pointer to host memory
`bytesize`
    \- Requested allocation size in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Allocates `bytesize` bytes of host memory that is page-locked and accessible to the device. The driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls to functions such as [cuMemcpy()](<group__CUDA__MEM.html#group__CUDA__MEM_1g8d0ff510f26d4b87bd3a51e731e7f698> "Copies memory."). Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory obtained with functions such as malloc().

On systems where [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>) is true, [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") may not page-lock the allocated memory.

Page-locking excessive amounts of memory with [cuMemAllocHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") may degrade system performance, since it reduces the amount of memory available to the system for paging. As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.

Note all host memory allocated using [cuMemAllocHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") will automatically be immediately accessible to all contexts on all devices which support unified addressing (as may be queried using [CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3dc11dd6d9f149a7bae32499f2b802c0d>)). The device pointer that may be used to access this host memory from those contexts is always equal to the returned host pointer `*pp`. See [Unified Addressing](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED>) for additional details.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMallocHost](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd5c991beb38e2b8419f50285707ae87e>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAllocManaged ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_tÂ bytesize, unsigned int Â flags )


Allocates memory that will be automatically managed by the Unified Memory system.

######  Parameters

`dptr`
    \- Returned device pointer
`bytesize`
    \- Requested allocation size in bytes
`flags`
    \- Must be one of [CU_MEM_ATTACH_GLOBAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313>) or [CU_MEM_ATTACH_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c8b59c62cab9c7a762338e5fae92e2e9c>)

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Allocates `bytesize` bytes of managed memory on the device and returns in `*dptr` a pointer to the allocated memory. If the device doesn't support allocating managed memory, [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>) is returned. Support for managed memory can be queried using the device attribute [CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a32e652d7656b5e1a381b8c430e41a055e>). The allocated memory is suitably aligned for any kind of variable. The memory is not cleared. If `bytesize` is 0, [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>). The pointer is valid on the CPU and on all GPUs in the system that support managed memory. All accesses to this pointer must obey the Unified Memory programming model.

`flags` specifies the default stream association for this allocation. `flags` must be one of [CU_MEM_ATTACH_GLOBAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313>) or [CU_MEM_ATTACH_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c8b59c62cab9c7a762338e5fae92e2e9c>). If [CU_MEM_ATTACH_GLOBAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313>) is specified, then this memory is accessible from any stream on any device. If [CU_MEM_ATTACH_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c8b59c62cab9c7a762338e5fae92e2e9c>) is specified, then the allocation should not be accessed from devices that have a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>); an explicit call to [cuStreamAttachMemAsync](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533> "Attach memory to a stream asynchronously.") will be required to enable access on such devices.

If the association is later changed via [cuStreamAttachMemAsync](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533> "Attach memory to a stream asynchronously.") to a single stream, the default association as specified during [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") is restored when that stream is destroyed. For __managed__ variables, the default association is always [CU_MEM_ATTACH_GLOBAL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg17c5d5f9b585aa2d6f121847d1a78f4c0b42aae6a29b41b734d4c0dea6c33313>). Note that destroying a stream is an asynchronous operation, and as a result, the change to default association won't happen until all work in the stream has completed.

Memory allocated with [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") should be released with [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory.").

Device memory oversubscription is possible for GPUs that have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). Managed memory on such GPUs may be evicted from device memory to host memory at any time by the Unified Memory driver in order to make room for other allocations.

In a system where all GPUs have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>), managed memory may not be populated when this API returns and instead may be populated on access. In such systems, managed memory can migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to maintain data locality and prevent excessive page faults to the extent possible. The application can also guide the driver about memory usage patterns via [cuMemAdvise](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1gaac8924b2f5a2a93f8775fb81c1a643f> "Advise about the usage of a given memory range."). The application can also explicitly migrate memory to a desired processor's memory via [cuMemPrefetchAsync](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g45c0e085febc3be8fabf5c526355b6a3> "Prefetches memory to the specified destination location.").

In a multi-GPU system where all of the GPUs have a zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) and all the GPUs have peer-to-peer support with each other, the physical storage for managed memory is created on the GPU which is active at the time [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system.") is called. All other GPUs will reference the data at reduced bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate memory among such GPUs.

In a multi-GPU system where not all GPUs have peer-to-peer support with each other and where the value of the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) is zero for at least one of those GPUs, the location chosen for physical storage of managed memory is system-dependent.

  * On Linux, the location chosen will be device memory as long as the current set of active contexts are on devices that either have peer-to-peer support with each other or have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>). If there is an active context on a GPU that does not have a non-zero value for that device attribute and it does not have peer-to-peer support with the other devices that have active contexts on them, then the location for physical storage will be 'zero-copy' or host memory. Note that this means that managed memory that is located in device memory is migrated to host memory if a new context is created on a GPU that doesn't have a non-zero value for the device attribute and does not support peer-to-peer with at least one of the other devices that has an active context. This in turn implies that context creation may fail if there is insufficient host memory to migrate all managed allocations.

  * On Windows, the physical storage is always created in 'zero-copy' or host memory. All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these circumstances, use of the environment variable CUDA_VISIBLE_DEVICES is recommended to restrict CUDA to only use those GPUs that have peer-to-peer support. Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero value to force the driver to always use device memory for physical storage. When this environment variable is set to a non-zero value, all contexts created in that process on devices that support managed memory have to be peer-to-peer compatible with each other. Context creation will fail if a context is created on a device that supports managed memory and is not peer-to-peer compatible with any of the other managed memory supporting devices on which contexts were previously created, even if those contexts have been destroyed. These environment variables are described in the CUDA programming guide under the "CUDA environment variables" section.

  * On ARM, managed memory is not available on discrete gpu with Drive PX-2.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuStreamAttachMemAsync](<group__CUDA__STREAM.html#group__CUDA__STREAM_1g6e468d680e263e7eba02a56643c50533> "Attach memory to a stream asynchronously."), [cudaMallocManaged](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gcf6b9b1019e73c5bc2b39b39fe90816e>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemAllocPitch ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â pPitch, size_tÂ WidthInBytes, size_tÂ Height, unsigned int Â ElementSizeBytes )


Allocates pitched device memory.

######  Parameters

`dptr`
    \- Returned device pointer
`pPitch`
    \- Returned pitch of allocation in bytes
`WidthInBytes`
    \- Requested allocation width in bytes
`Height`
    \- Requested allocation height in rows
`ElementSizeBytes`
    \- Size of largest reads/writes for range

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Allocates at least `WidthInBytes` * `Height` bytes of linear memory on the device and returns in `*dptr` a pointer to the allocated memory. The function may pad the allocation to ensure that corresponding pointers in any given row will continue to meet the alignment requirements for coalescing as the address is updated from row to row. `ElementSizeBytes` specifies the size of the largest reads and writes that will be performed on the memory range. `ElementSizeBytes` may be 4, 8 or 16 (since coalesced memory transactions are not possible on other data sizes). If `ElementSizeBytes` is smaller than the actual read/write size of a kernel, the kernel will run correctly, but possibly at reduced speed. The pitch returned in `*pPitch` by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") is the width in bytes of the allocation. The intended usage of pitch is as a separate parameter of the allocation, used to compute addresses within the 2D array. Given the row and column of an array element of type **T** , the address is computed as:


    â   T* pElement = (T*)((char*)BaseAddress + Row * Pitch) + Column;

The pitch returned by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") is guaranteed to work with [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") under all circumstances. For allocations of 2D arrays, it is recommended that programmers consider performing pitch allocations using [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."). Due to alignment restrictions in the hardware, this is especially true if the application will be performing 2D memory copies between different regions of device memory (whether linear memory or CUDA arrays).

The byte alignment of the pitch returned by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") is guaranteed to match or exceed the alignment requirement for texture binding with [cuTexRefSetAddress2D()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMallocPitch](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g32bd7a39135594788a542ae72217775c>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemBatchDecompressAsync ( [CUmemDecompressParams](<structCUmemDecompressParams.html#structCUmemDecompressParams> "Structure describing the parameters that compose a single decompression operation. ")*Â paramsArray, size_tÂ count, unsigned int Â flags, size_t*Â errorIndex, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â stream )


Submit a batch of `count` independent decompression operations.

######  Parameters

`paramsArray`
    The array of structures describing the independent decompression operations.
`count`
    The number of entries in `paramsArray` array.
`flags`
    Must be 0.
`errorIndex`
    The index into `paramsArray` of the decompression operation for which the error returned by this function pertains to. If `index` is SIZE_MAX and the value returned is not [CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), then the error returned by this function should be considered a general error that does not pertain to a particular decompression operation. May be `NULL`, in which case, no index will be recorded in the event of error.
`stream`
    The stream where the work will be enqueued.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Each of the `count` decompression operations is described by a single entry in the `paramsArray` array. Once the batch has been submitted, the function will return, and decompression will happen asynchronously w.r.t. the CPU. To the work completion tracking mechanisms in the CUDA driver, the batch will be considered a single unit of work and processed according to stream semantics, i.e., it is not possible to query the completion of individual decompression operations within a batch.

The memory pointed to by each of [CUmemDecompressParams.src](<structCUmemDecompressParams.html#structCUmemDecompressParams_14d390c15ec5ed068f0912e42077ae0e0>), [CUmemDecompressParams.dst](<structCUmemDecompressParams.html#structCUmemDecompressParams_139077e823122cfcfab6aeca53093c88c>), and [CUmemDecompressParams.dstActBytes](<structCUmemDecompressParams.html#structCUmemDecompressParams_1df7e9b52dd81aa6c0ed689eeab86d1a7>), must be capable of usage with the hardware decompress feature. That is, for each of said pointers, the pointer attribute [CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc2cce590e35080745e72633dfc6e0b60820cbe95f809088e8045fcb1c9857bf5>) should give a non-zero value. To ensure this, the memory backing the pointers should have been allocated using one of the following CUDA memory allocators: * [cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") * [cuMemCreate()](<group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c> "Create a CUDA memory handle representing a memory allocation of a given size described by the given properties.") with the usage flag [CU_MEM_CREATE_USAGE_HW_DECOMPRESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8b3b5d15c34f384cd3ada57fe8bb4a57>) * [cuMemAllocFromPoolAsync()](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.") from a pool that was created with the usage flag [CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gaba1d50e8fd2014843f97c2758ce9cc9>) Additionally, [CUmemDecompressParams.src](<structCUmemDecompressParams.html#structCUmemDecompressParams_14d390c15ec5ed068f0912e42077ae0e0>), [CUmemDecompressParams.dst](<structCUmemDecompressParams.html#structCUmemDecompressParams_139077e823122cfcfab6aeca53093c88c>), and [CUmemDecompressParams.dstActBytes](<structCUmemDecompressParams.html#structCUmemDecompressParams_1df7e9b52dd81aa6c0ed689eeab86d1a7>), must all be accessible from the device associated with the context where `stream` was created. For information on how to ensure this, see the documentation for the allocator of interest.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemPoolCreate](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g8aa4c143dbc20293659cd883232b95f2> "Creates a memory pool."), [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemFree ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )


Frees device memory.

######  Parameters

`dptr`
    \- Pointer to memory to free

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Frees the memory space pointed to by `dptr`, which must have been returned by a previous call to one of the following memory allocation APIs - [cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemAllocManaged()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."), [cuMemAllocAsync()](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemAllocFromPoolAsync()](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics.")

Note - This API will not perform any implict synchronization when the pointer was allocated with [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics.") or [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics."). Callers must ensure that all accesses to these pointer have completed before invoking [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."). For best performance and memory reuse, users should use [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics.") to free memory allocated via the stream ordered memory allocator. For all other pointers, this API may perform implicit synchronization.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemAllocManaged](<group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32> "Allocates memory that will be automatically managed by the Unified Memory system."), [cuMemAllocAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g13413273e84a641bce1929eae9e6501f> "Allocates memory with stream ordered semantics."), [cuMemAllocFromPoolAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1gf1dd6e1e2e8f767a5e0ea63f38ff260b> "Allocates memory from a specified pool with stream ordered semantics."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemFreeAsync](<group__CUDA__MALLOC__ASYNC.html#group__CUDA__MALLOC__ASYNC_1g41acf4131f672a2a75cd93d3241f10cf> "Frees memory with stream ordered semantics."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaFree](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemFreeHost ( void*Â p )


Frees page-locked host memory.

######  Parameters

`p`
    \- Pointer to memory to free

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Frees the memory space pointed to by `p`, which must have been returned by a previous call to [cuMemAllocHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaFreeHost](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetAddressRange ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pbase, size_t*Â psize, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr )


Get information on memory allocations.

######  Parameters

`pbase`
    \- Returned base address
`psize`
    \- Returned size of device memory allocation
`dptr`
    \- Device pointer to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the base address in `*pbase` and size in `*psize` of the allocation that contains the input pointer `dptr`. Both parameters `pbase` and `psize` are optional. If one of them is NULL, it is ignored.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetHandleForAddressRange ( void*Â handle, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ size, [CUmemRangeHandleType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g579b315f05d1e65a4f3de7da45013210>)Â handleType, unsigned long longÂ flags )


Retrieve handle for an address range.

######  Parameters

`handle`
    \- Pointer to the location where the returned handle will be stored.
`dptr`
    \- Pointer to a valid CUDA device allocation. Must be aligned to host page size.
`size`
    \- Length of the address range. Must be aligned to host page size.
`handleType`
    \- Type of handle requested (defines type and size of the `handle` output parameter)
`flags`
    \- When requesting CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD the value could be [CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75c53565b19e5c434edc5a65a6a7ab20ff810d1182d50bd1385eb543478b99f5>), otherwise 0.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Get a handle of the specified type to an address range. When requesting CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, address range obtained by a prior call to either [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory.") or [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation.") is supported if the [CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a339de64a7d3e21d22411d8dc6a2cde25b>) device attribute returns true. If the address range was obtained via [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation."), it must also be fully mapped via [cuMemMap](<group__CUDA__VA.html#group__CUDA__VA_1gff1d395423af5c5c75375516959dae56> "Maps an allocation handle to a reserved virtual address range."). Address range obtained by a prior call to either [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") or [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") is supported if the [CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3a7208113fdb76066caa5468ab3be4ce5>) device attribute returns true.

As of CUDA 13.0, querying support for address range obtained by calling [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") or [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") using the [CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a339de64a7d3e21d22411d8dc6a2cde25b>) device attribute is deprecated.

Users must ensure the `dptr` and `size` are aligned to the host page size.

The `handle` will be interpreted as a pointer to an integer to store the dma_buf file descriptor. Users must ensure the entire address range is backed and mapped when the address range is allocated by [cuMemAddressReserve](<group__CUDA__VA.html#group__CUDA__VA_1ge489256c107df2a07ddf96d80c86cd9b> "Allocate an address range reservation."). All the physical allocations backing the address range must be resident on the same device and have identical allocation properties. Users are also expected to retrieve a new handle every time the underlying physical allocation(s) corresponding to a previously queried VA range are changed.

For CUmemRangeHandleType::CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, users may set flags to [CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg75c53565b19e5c434edc5a65a6a7ab20ff810d1182d50bd1385eb543478b99f5>). Which when set on a supported platform, will give a DMA_BUF handle mapped via PCIE BAR1 or will return an error otherwise.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemGetInfo ( size_t*Â free, size_t*Â total )


Gets free and total memory.

######  Parameters

`free`
    \- Returned free memory in bytes
`total`
    \- Returned total memory in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*total` the total amount of memory available to the the current context. Returns in `*free` the amount of memory on the device that is free according to the OS. CUDA is not guaranteed to be able to allocate all of the memory that the OS reports as free. In a multi-tenet situation, free estimate returned is prone to race condition where a new allocation/free done by a different process or a different thread in the same process between the time when free memory was estimated and reported, will result in deviation in free value reported and actual free memory.

The integrated GPU on Tegra shares memory with CPU and other component of the SoC. The free and total values returned by the API excludes the SWAP memory space maintained by the OS on some platforms. The OS may move some of the memory pages into swap area as the GPU or CPU allocate or access memory. See Tegra app note on how to calculate total and free memory on Tegra.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemGetInfo](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g376b97f5ab20321ca46f7cfa9511b978>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemHostAlloc ( void**Â pp, size_tÂ bytesize, unsigned int Â Flags )


Allocates page-locked host memory.

######  Parameters

`pp`
    \- Returned pointer to host memory
`bytesize`
    \- Requested allocation size in bytes
`Flags`
    \- Flags for allocation request

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Allocates `bytesize` bytes of host memory that is page-locked and accessible to the device. The driver tracks the virtual memory ranges allocated with this function and automatically accelerates calls to functions such as [cuMemcpyHtoD()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."). Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory obtained with functions such as malloc().

On systems where [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>) is true, [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") may not page-lock the allocated memory.

Page-locking excessive amounts of memory may degrade system performance, since it reduces the amount of memory available to the system for paging. As a result, this function is best used sparingly to allocate staging areas for data exchange between host and device.

The `Flags` parameter enables different options to be specified that affect the allocation, as follows.

  * [CU_MEMHOSTALLOC_PORTABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g50f4528d46bda58b592551654a7ee0ff>): The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.


  * [CU_MEMHOSTALLOC_DEVICEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g054589ee2a0f188e664d93965d81113d>): Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.").


  * [CU_MEMHOSTALLOC_WRITECOMBINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7361580951deecace15352c97a210038>): Allocates the memory as write-combined (WC). WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by the GPU via mapped pinned memory or host->device transfers.


All of these flags are orthogonal to one another: a developer may allocate memory that is portable, mapped and/or write-combined with no restrictions.

The [CU_MEMHOSTALLOC_DEVICEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g054589ee2a0f188e664d93965d81113d>) flag may be specified on CUDA contexts for devices that do not support mapped pinned memory. The failure is deferred to [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") because the memory may be mapped into other CUDA contexts via the [CU_MEMHOSTALLOC_PORTABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g50f4528d46bda58b592551654a7ee0ff>) flag.

The memory allocated by this function must be freed with [cuMemFreeHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory.").

Note all host memory allocated using [cuMemHostAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") will automatically be immediately accessible to all contexts on all devices which support unified addressing (as may be queried using [CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3dc11dd6d9f149a7bae32499f2b802c0d>)). Unless the flag [CU_MEMHOSTALLOC_WRITECOMBINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7361580951deecace15352c97a210038>) is specified, the device pointer that may be used to access this host memory from those contexts is always equal to the returned host pointer `*pp`. If the flag [CU_MEMHOSTALLOC_WRITECOMBINED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7361580951deecace15352c97a210038>) is specified, then the function [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") must be used to query the device pointer, even if the context supports unified addressing. See [Unified Addressing](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED>) for additional details.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaHostAlloc](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemHostGetDevicePointer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pdptr, void*Â p, unsigned int Â Flags )


Passes back device pointer of mapped pinned memory.

######  Parameters

`pdptr`
    \- Returned device pointer
`p`
    \- Host pointer
`Flags`
    \- Options (must be 0)

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Passes back the device pointer `pdptr` corresponding to the mapped, pinned host buffer `p` allocated by [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.").

[cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") will fail if the [CU_MEMHOSTALLOC_DEVICEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g054589ee2a0f188e664d93965d81113d>) flag was not specified at the time the memory was allocated, or if the function is called on a GPU that does not support mapped pinned memory.

For devices that have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3904c097da7a1891f9904b3e6a49e4cdd>), the memory can also be accessed from the device using the host pointer `p`. The device pointer returned by [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") may or may not match the original host pointer `p` and depends on the devices visible to the application. If all devices visible to the application have a non-zero value for the device attribute, the device pointer returned by [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") will match the original pointer `p`. If any device visible to the application has a zero value for the device attribute, the device pointer returned by [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") will not match the original host pointer `p`, but it will be suitable for use on all devices provided Unified Virtual Addressing is enabled. In such systems, it is valid to access the memory using either pointer on devices that have a non-zero value for the device attribute. Note however that such devices should access the memory using only one of the two pointers and not both.

`Flags` provides for future releases. For now, it must be set to 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaHostGetDevicePointer](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc00502b44e5f1bdc0b424487ebb08db0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemHostGetFlags ( unsigned int*Â pFlags, void*Â p )


Passes back flags that were used for a pinned allocation.

######  Parameters

`pFlags`
    \- Returned flags word
`p`
    \- Host pointer

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Passes back the flags `pFlags` that were specified when allocating the pinned host buffer `p` allocated by [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.").

[cuMemHostGetFlags()](<group__CUDA__MEM.html#group__CUDA__MEM_1g42066246915fcb0400df2a17a851b35f> "Passes back flags that were used for a pinned allocation.") will fail if the pointer does not reside in an allocation performed by [cuMemAllocHost()](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory.") or [cuMemHostAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cudaHostGetFlags](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc470e9220559109f5088d9a01c0aeeda>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemHostRegister ( void*Â p, size_tÂ bytesize, unsigned int Â Flags )


Registers an existing host memory range for use by CUDA.

######  Parameters

`p`
    \- Host pointer to memory to page-lock
`bytesize`
    \- Size in bytes of the address range to page-lock
`Flags`
    \- Flags for allocation request

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9959a4a8475dc87812c3c64213b18dcba>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>), [CUDA_ERROR_NOT_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e954756ae7ade0dfd09faeccb513dd831b>)

###### Description

Page-locks the memory range specified by `p` and `bytesize` and maps it for the device(s) as specified by `Flags`. This memory range also is added to the same tracking mechanism as [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory.") to automatically accelerate calls to functions such as [cuMemcpyHtoD()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."). Since the memory can be accessed directly by the device, it can be read or written with much higher bandwidth than pageable memory that has not been registered. Page-locking excessive amounts of memory may degrade system performance, since it reduces the amount of memory available to the system for paging. As a result, this function is best used sparingly to register staging areas for data exchange between host and device.

On systems where [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>) is true, [cuMemHostRegister](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA.") will not page-lock the memory range specified by `ptr` but only populate unpopulated pages.

The `Flags` parameter enables different options to be specified that affect the allocation, as follows.

  * [CU_MEMHOSTREGISTER_PORTABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4f20a39f0a7bddc8ce7d644327a2e7da>): The memory returned by this call will be considered as pinned memory by all CUDA contexts, not just the one that performed the allocation.


  * [CU_MEMHOSTREGISTER_DEVICEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf1fc8645f0ab5481e7be96c80f6bfa50>): Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.").


  * [CU_MEMHOSTREGISTER_IOMEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6a86cf1a826f8da5b01f1b5cd8da2bde>): The pointer is treated as pointing to some I/O memory space, e.g. the PCI Express resource of a 3rd party device.


  * [CU_MEMHOSTREGISTER_READ_ONLY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd870d49634958b801f5c02a6ba459a1a>): The pointer is treated as pointing to memory that is considered read-only by the device. On platforms without [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a352c58d6fd1d3a72673cce199ab30cd40>), this flag is required in order to register memory mapped to the CPU as read-only. Support for the use of this flag can be queried from the device attribute [CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a33a3a9091a7991536d507dd5eff146d2b>). Using this flag with a current context associated with a device that does not have this attribute set will cause [cuMemHostRegister](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA.") to error with CUDA_ERROR_NOT_SUPPORTED.


All of these flags are orthogonal to one another: a developer may page-lock memory that is portable or mapped with no restrictions.

The [CU_MEMHOSTREGISTER_DEVICEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf1fc8645f0ab5481e7be96c80f6bfa50>) flag may be specified on CUDA contexts for devices that do not support mapped pinned memory. The failure is deferred to [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") because the memory may be mapped into other CUDA contexts via the [CU_MEMHOSTREGISTER_PORTABLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4f20a39f0a7bddc8ce7d644327a2e7da>) flag.

For devices that have a non-zero value for the device attribute [CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3904c097da7a1891f9904b3e6a49e4cdd>), the memory can also be accessed from the device using the host pointer `p`. The device pointer returned by [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") may or may not match the original host pointer `ptr` and depends on the devices visible to the application. If all devices visible to the application have a non-zero value for the device attribute, the device pointer returned by [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") will match the original pointer `ptr`. If any device visible to the application has a zero value for the device attribute, the device pointer returned by [cuMemHostGetDevicePointer()](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory.") will not match the original host pointer `ptr`, but it will be suitable for use on all devices provided Unified Virtual Addressing is enabled. In such systems, it is valid to access the memory using either pointer on devices that have a non-zero value for the device attribute. Note however that such devices should access the memory using only of the two pointers and not both.

The memory page-locked by this function must be unregistered with [cuMemHostUnregister()](<group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14> "Unregisters a memory range that was registered with cuMemHostRegister.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemHostUnregister](<group__CUDA__MEM.html#group__CUDA__MEM_1g63f450c8125359be87b7623b1c0b2a14> "Unregisters a memory range that was registered with cuMemHostRegister."), [cuMemHostGetFlags](<group__CUDA__MEM.html#group__CUDA__MEM_1g42066246915fcb0400df2a17a851b35f> "Passes back flags that were used for a pinned allocation."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cudaHostRegister](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8d5c17670f16ac4fc8fcb4181cb490c>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemHostUnregister ( void*Â p )


Unregisters a memory range that was registered with cuMemHostRegister.

######  Parameters

`p`
    \- Host pointer to memory to unregister

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e6e64a1f120336c5a90794e4d634c703>),

###### Description

Unmaps the memory range whose base address is specified by `p`, and makes it pageable again.

The base address must be the same one specified to [cuMemHostRegister()](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMemHostRegister](<group__CUDA__MEM.html#group__CUDA__MEM_1gf0a9fe11544326dabd743b7aa6b54223> "Registers an existing host memory range for use by CUDA."), [cudaHostUnregister](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g81fd4101862bbefdb42a62d60e515eea>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dst, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â src, size_tÂ ByteCount )


Copies memory.

######  Parameters

`dst`
    \- Destination unified virtual address space pointer
`src`
    \- Source unified virtual address space pointer
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies data between two pointers. `dst` and `src` are base pointers of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy. Note that this function infers the type of the transfer (host to host, host to device, device to device, or device to host) from the pointer values. This function is only allowed in contexts which support unified addressing.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8>), [cudaMemcpyToSymbol](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g4561bf9c99d91c92684a91a0bd356bfe>), [cudaMemcpyFromSymbol](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g99db510d18d37fbb0f5c075a8caf3b5f>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy2D ( const [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>)*Â pCopy )


Copies memory for 2D arrays.

######  Parameters

`pCopy`
    \- Parameters for the memory copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Perform a 2D memory copy according to the parameters specified in `pCopy`. The CUDA_MEMCPY2D structure is defined as:


    â   typedef struct CUDA_MEMCPY2D_st {
                unsigned int srcXInBytes, srcY;
                [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) srcMemoryType;
                    const void *srcHost;
                    [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) srcDevice;
                    [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) srcArray;
                    unsigned int srcPitch;

                unsigned int dstXInBytes, dstY;
                [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) dstMemoryType;
                    void *dstHost;
                    [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstDevice;
                    [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) dstArray;
                    unsigned int dstPitch;

                unsigned int WidthInBytes;
                unsigned int Height;
             } [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>);

where:

  * srcMemoryType and dstMemoryType specify the type of memory of the source and destination, respectively; CUmemorytype_enum is defined as:




    â   typedef enum CUmemorytype_enum {
                [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>) = 0x01,
                [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>) = 0x02,
                [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>) = 0x03,
                [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) = 0x04
             } [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>);

If srcMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), srcDevice and srcPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. srcArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If srcMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), srcHost and srcPitch specify the (host) base address of the source data and the bytes per row to apply. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), srcDevice and srcPitch specify the (device) base address of the source data and the bytes per row to apply. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), srcArray specifies the handle of the source data. srcHost, srcDevice and srcPitch are ignored.

If dstMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), dstHost and dstPitch specify the (host) base address of the destination data and the bytes per row to apply. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), dstDevice and dstPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. dstArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If dstMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), dstDevice and dstPitch specify the (device) base address of the destination data and the bytes per row to apply. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), dstArray specifies the handle of the destination data. dstHost, dstDevice and dstPitch are ignored.

  * srcXInBytes and srcY specify the base address of the source data for the copy.


For host pointers, the starting address is


    â  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) Start = srcDevice+srcY*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.

  * dstXInBytes and dstY specify the base address of the destination data for the copy.


For host pointers, the base address is


    â  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstStart = dstDevice+dstY*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

  * WidthInBytes and Height specify the width (in bytes) and height of the 2D copy being performed.

  * If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must be greater than or equal to WidthInBytes + dstXInBytes.


[cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") returns an error if any pitch is greater than the maximum allowed ([CU_DEVICE_ATTRIBUTE_MAX_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3c1625acc7a2db635bc1efae34030598d>)). [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") passes back pitches that always work with [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."). On intra-device memory copies (device to device, CUDA array to device, CUDA array to CUDA array), [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") may fail for pitches not computed by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."). [cuMemcpy2DUnaligned()](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays.") does not have this restriction, but may run significantly slower in the cases where [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") would have returned an error code.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy2D](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g3a58270f6775efe56c65ac47843e7cee>), [cudaMemcpy2DToArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9509226164aaa58baf0c5b8ed165df58>), [cudaMemcpy2DFromArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g0f944b3fd3c81edad0a352cf22de24f0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy2DAsync ( const [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>)*Â pCopy, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory for 2D arrays.

######  Parameters

`pCopy`
    \- Parameters for the memory copy
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Perform a 2D memory copy according to the parameters specified in `pCopy`. The CUDA_MEMCPY2D structure is defined as:


    â   typedef struct CUDA_MEMCPY2D_st {
                unsigned int srcXInBytes, srcY;
                [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) srcMemoryType;
                const void *srcHost;
                [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) srcDevice;
                [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) srcArray;
                unsigned int srcPitch;
                unsigned int dstXInBytes, dstY;
                [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) dstMemoryType;
                void *dstHost;
                [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstDevice;
                [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) dstArray;
                unsigned int dstPitch;
                unsigned int WidthInBytes;
                unsigned int Height;
             } [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>);

where:

  * srcMemoryType and dstMemoryType specify the type of memory of the source and destination, respectively; CUmemorytype_enum is defined as:




    â   typedef enum CUmemorytype_enum {
                [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>) = 0x01,
                [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>) = 0x02,
                [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>) = 0x03,
                [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) = 0x04
             } [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>);

If srcMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), srcHost and srcPitch specify the (host) base address of the source data and the bytes per row to apply. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), srcDevice and srcPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. srcArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If srcMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), srcDevice and srcPitch specify the (device) base address of the source data and the bytes per row to apply. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), srcArray specifies the handle of the source data. srcHost, srcDevice and srcPitch are ignored.

If dstMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), dstDevice and dstPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. dstArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If dstMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), dstHost and dstPitch specify the (host) base address of the destination data and the bytes per row to apply. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), dstDevice and dstPitch specify the (device) base address of the destination data and the bytes per row to apply. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), dstArray specifies the handle of the destination data. dstHost, dstDevice and dstPitch are ignored.

  * srcXInBytes and srcY specify the base address of the source data for the copy.


For host pointers, the starting address is


    â  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) Start = srcDevice+srcY*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.

  * dstXInBytes and dstY specify the base address of the destination data for the copy.


For host pointers, the base address is


    â  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstStart = dstDevice+dstY*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

  * WidthInBytes and Height specify the width (in bytes) and height of the 2D copy being performed.

  * If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must be greater than or equal to WidthInBytes + dstXInBytes.

  * If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must be greater than or equal to WidthInBytes + dstXInBytes.

  * If specified, srcHeight must be greater than or equal to Height + srcY, and dstHeight must be greater than or equal to Height \+ dstY.


[cuMemcpy2DAsync()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays.") returns an error if any pitch is greater than the maximum allowed ([CU_DEVICE_ATTRIBUTE_MAX_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3c1625acc7a2db635bc1efae34030598d>)). [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") passes back pitches that always work with [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."). On intra-device memory copies (device to device, CUDA array to device, CUDA array to CUDA array), [cuMemcpy2DAsync()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays.") may fail for pitches not computed by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpy2DAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge529b926e8fb574c2666a9a1d58b0dc1>), [cudaMemcpy2DToArrayAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g217af4b9e2de79d9252418fc661e6a6a>), [cudaMemcpy2DFromArrayAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g1c81de45e9ed5e72008a8f28e706b599>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy2DUnaligned ( const [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>)*Â pCopy )


Copies memory for 2D arrays.

######  Parameters

`pCopy`
    \- Parameters for the memory copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Perform a 2D memory copy according to the parameters specified in `pCopy`. The CUDA_MEMCPY2D structure is defined as:


    â   typedef struct CUDA_MEMCPY2D_st {
                unsigned int srcXInBytes, srcY;
                [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) srcMemoryType;
                const void *srcHost;
                [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) srcDevice;
                [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) srcArray;
                unsigned int srcPitch;
                unsigned int dstXInBytes, dstY;
                [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) dstMemoryType;
                void *dstHost;
                [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstDevice;
                [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) dstArray;
                unsigned int dstPitch;
                unsigned int WidthInBytes;
                unsigned int Height;
             } [CUDA_MEMCPY2D](<structCUDA__MEMCPY2D__v2.html#structCUDA__MEMCPY2D__v2>);

where:

  * srcMemoryType and dstMemoryType specify the type of memory of the source and destination, respectively; CUmemorytype_enum is defined as:




    â   typedef enum CUmemorytype_enum {
                [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>) = 0x01,
                [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>) = 0x02,
                [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>) = 0x03,
                [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) = 0x04
             } [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>);

If srcMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), srcDevice and srcPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. srcArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If srcMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), srcHost and srcPitch specify the (host) base address of the source data and the bytes per row to apply. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), srcDevice and srcPitch specify the (device) base address of the source data and the bytes per row to apply. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), srcArray specifies the handle of the source data. srcHost, srcDevice and srcPitch are ignored.

If dstMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), dstDevice and dstPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. dstArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If dstMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), dstHost and dstPitch specify the (host) base address of the destination data and the bytes per row to apply. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), dstDevice and dstPitch specify the (device) base address of the destination data and the bytes per row to apply. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), dstArray specifies the handle of the destination data. dstHost, dstDevice and dstPitch are ignored.

  * srcXInBytes and srcY specify the base address of the source data for the copy.


For host pointers, the starting address is


    â  void* Start = (void*)((char*)srcHost+srcY*srcPitch + srcXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) Start = srcDevice+srcY*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.

  * dstXInBytes and dstY specify the base address of the destination data for the copy.


For host pointers, the base address is


    â  void* dstStart = (void*)((char*)dstHost+dstY*dstPitch + dstXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstStart = dstDevice+dstY*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

  * WidthInBytes and Height specify the width (in bytes) and height of the 2D copy being performed.

  * If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must be greater than or equal to WidthInBytes + dstXInBytes.


[cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") returns an error if any pitch is greater than the maximum allowed ([CU_DEVICE_ATTRIBUTE_MAX_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3c1625acc7a2db635bc1efae34030598d>)). [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.") passes back pitches that always work with [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."). On intra-device memory copies (device to device, CUDA array to device, CUDA array to CUDA array), [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") may fail for pitches not computed by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."). [cuMemcpy2DUnaligned()](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays.") does not have this restriction, but may run significantly slower in the cases where [cuMemcpy2D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays.") would have returned an error code.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy2D](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g3a58270f6775efe56c65ac47843e7cee>), [cudaMemcpy2DToArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9509226164aaa58baf0c5b8ed165df58>), [cudaMemcpy2DFromArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g0f944b3fd3c81edad0a352cf22de24f0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy3D ( const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â pCopy )


Copies memory for 3D arrays.

######  Parameters

`pCopy`
    \- Parameters for the memory copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Perform a 3D memory copy according to the parameters specified in `pCopy`. The CUDA_MEMCPY3D structure is defined as:


    â        typedef struct CUDA_MEMCPY3D_st {

                      unsigned int srcXInBytes, srcY, srcZ;
                      unsigned int srcLOD;
                      [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) srcMemoryType;
                          const void *srcHost;
                          [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) srcDevice;
                          [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) srcArray;
                          unsigned int srcPitch;  // ignored when src is array
                          unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1

                      unsigned int dstXInBytes, dstY, dstZ;
                      unsigned int dstLOD;
                      [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) dstMemoryType;
                          void *dstHost;
                          [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstDevice;
                          [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) dstArray;
                          unsigned int dstPitch;  // ignored when dst is array
                          unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1

                      unsigned int WidthInBytes;
                      unsigned int Height;
                      unsigned int Depth;
                  } [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>);

where:

  * srcMemoryType and dstMemoryType specify the type of memory of the source and destination, respectively; CUmemorytype_enum is defined as:




    â   typedef enum CUmemorytype_enum {
                [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>) = 0x01,
                [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>) = 0x02,
                [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>) = 0x03,
                [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) = 0x04
             } [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>);

If srcMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), srcDevice and srcPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. srcArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If srcMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), srcHost, srcPitch and srcHeight specify the (host) base address of the source data, the bytes per row, and the height of each 2D slice of the 3D array. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), srcDevice, srcPitch and srcHeight specify the (device) base address of the source data, the bytes per row, and the height of each 2D slice of the 3D array. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), srcArray specifies the handle of the source data. srcHost, srcDevice, srcPitch and srcHeight are ignored.

If dstMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), dstDevice and dstPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. dstArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If dstMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), dstHost and dstPitch specify the (host) base address of the destination data, the bytes per row, and the height of each 2D slice of the 3D array. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), dstDevice and dstPitch specify the (device) base address of the destination data, the bytes per row, and the height of each 2D slice of the 3D array. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), dstArray specifies the handle of the destination data. dstHost, dstDevice, dstPitch and dstHeight are ignored.

  * srcXInBytes, srcY and srcZ specify the base address of the source data for the copy.


For host pointers, the starting address is


    â  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.

  * dstXInBytes, dstY and dstZ specify the base address of the destination data for the copy.


For host pointers, the base address is


    â  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

  * WidthInBytes, Height and Depth specify the width (in bytes), height and depth of the 3D copy being performed.

  * If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must be greater than or equal to WidthInBytes + dstXInBytes.

  * If specified, srcHeight must be greater than or equal to Height + srcY, and dstHeight must be greater than or equal to Height \+ dstY.


[cuMemcpy3D()](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays.") returns an error if any pitch is greater than the maximum allowed ([CU_DEVICE_ATTRIBUTE_MAX_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3c1625acc7a2db635bc1efae34030598d>)).

The srcLOD and dstLOD members of the CUDA_MEMCPY3D structure must be set to 0.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy3D](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gfec7ee5257d48c8528a709ffad48d208>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy3DAsync ( const [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>)*Â pCopy, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory for 3D arrays.

######  Parameters

`pCopy`
    \- Parameters for the memory copy
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Perform a 3D memory copy according to the parameters specified in `pCopy`. The CUDA_MEMCPY3D structure is defined as:


    â        typedef struct CUDA_MEMCPY3D_st {

                      unsigned int srcXInBytes, srcY, srcZ;
                      unsigned int srcLOD;
                      [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) srcMemoryType;
                          const void *srcHost;
                          [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) srcDevice;
                          [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) srcArray;
                          unsigned int srcPitch;  // ignored when src is array
                          unsigned int srcHeight; // ignored when src is array; may be 0 if Depth==1

                      unsigned int dstXInBytes, dstY, dstZ;
                      unsigned int dstLOD;
                      [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>) dstMemoryType;
                          void *dstHost;
                          [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstDevice;
                          [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) dstArray;
                          unsigned int dstPitch;  // ignored when dst is array
                          unsigned int dstHeight; // ignored when dst is array; may be 0 if Depth==1

                      unsigned int WidthInBytes;
                      unsigned int Height;
                      unsigned int Depth;
                  } [CUDA_MEMCPY3D](<structCUDA__MEMCPY3D__v2.html#structCUDA__MEMCPY3D__v2>);

where:

  * srcMemoryType and dstMemoryType specify the type of memory of the source and destination, respectively; CUmemorytype_enum is defined as:




    â   typedef enum CUmemorytype_enum {
                [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>) = 0x01,
                [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>) = 0x02,
                [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>) = 0x03,
                [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>) = 0x04
             } [CUmemorytype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8a114cc994ad2e865c44ef3838eaec72>);

If srcMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), srcDevice and srcPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. srcArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If srcMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), srcHost, srcPitch and srcHeight specify the (host) base address of the source data, the bytes per row, and the height of each 2D slice of the 3D array. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), srcDevice, srcPitch and srcHeight specify the (device) base address of the source data, the bytes per row, and the height of each 2D slice of the 3D array. srcArray is ignored.

If srcMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), srcArray specifies the handle of the source data. srcHost, srcDevice, srcPitch and srcHeight are ignored.

If dstMemoryType is [CU_MEMORYTYPE_UNIFIED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727a47ca2de6db5cf82084ad80ce66aa71>), dstDevice and dstPitch specify the (unified virtual address space) base address of the source data and the bytes per row to apply. dstArray is ignored. This value may be used only if unified addressing is supported in the calling context.

If dstMemoryType is [CU_MEMORYTYPE_HOST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec727f98a88f26eec8490bfc180c5a73e101>), dstHost and dstPitch specify the (host) base address of the destination data, the bytes per row, and the height of each 2D slice of the 3D array. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72ec7e15ba4b111a26adb3487023707299>), dstDevice and dstPitch specify the (device) base address of the destination data, the bytes per row, and the height of each 2D slice of the 3D array. dstArray is ignored.

If dstMemoryType is [CU_MEMORYTYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg8a114cc994ad2e865c44ef3838eaec72d7f97cd13a156651767607456fe25b66>), dstArray specifies the handle of the destination data. dstHost, dstDevice, dstPitch and dstHeight are ignored.

  * srcXInBytes, srcY and srcZ specify the base address of the source data for the copy.


For host pointers, the starting address is


    â  void* Start = (void*)((char*)srcHost+(srcZ*srcHeight+srcY)*srcPitch + srcXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) Start = srcDevice+(srcZ*srcHeight+srcY)*srcPitch+srcXInBytes;

For CUDA arrays, srcXInBytes must be evenly divisible by the array element size.

  * dstXInBytes, dstY and dstZ specify the base address of the destination data for the copy.


For host pointers, the base address is


    â  void* dstStart = (void*)((char*)dstHost+(dstZ*dstHeight+dstY)*dstPitch + dstXInBytes);

For device pointers, the starting address is


    â  [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) dstStart = dstDevice+(dstZ*dstHeight+dstY)*dstPitch+dstXInBytes;

For CUDA arrays, dstXInBytes must be evenly divisible by the array element size.

  * WidthInBytes, Height and Depth specify the width (in bytes), height and depth of the 3D copy being performed.

  * If specified, srcPitch must be greater than or equal to WidthInBytes + srcXInBytes, and dstPitch must be greater than or equal to WidthInBytes + dstXInBytes.

  * If specified, srcHeight must be greater than or equal to Height + srcY, and dstHeight must be greater than or equal to Height \+ dstY.


[cuMemcpy3DAsync()](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays.") returns an error if any pitch is greater than the maximum allowed ([CU_DEVICE_ATTRIBUTE_MAX_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3c1625acc7a2db635bc1efae34030598d>)).

The srcLOD and dstLOD members of the CUDA_MEMCPY3D structure must be set to 0.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpy3DAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g785bd0963e476a740533382a67674641>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy3DBatchAsync ( size_tÂ numOps, CUDA_MEMCPY3D_BATCH_OP*Â opList, unsigned long longÂ flags, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Performs a batch of 3D memory copies asynchronously.

######  Parameters

`numOps`
    \- Total number of memcpy operations.
`opList`
    \- Array of size `numOps` containing the actual memcpy operations.
`flags`
    \- Flags for future use, must be zero now.
`hStream`
    \- The stream to enqueue the operations in. Must not be default NULL stream.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Performs a batch of memory copies. The batch as a whole executes in stream order but copies within a batch are not guaranteed to execute in any specific order. Note that this means specifying any dependent copies within a batch will result in undefined behavior.

Performs memory copies as specified in the `opList` array. The length of this array is specified in `numOps`. Each entry in this array describes a copy operation. This includes among other things, the source and destination operands for the copy as specified in CUDA_MEMCPY3D_BATCH_OP::src and CUDA_MEMCPY3D_BATCH_OP::dst respectively. The source and destination operands of a copy can either be a pointer or a CUDA array. The width, height and depth of a copy is specified in CUDA_MEMCPY3D_BATCH_OP::extent. The width, height and depth of a copy are specified in elements and must not be zero. For pointer-to-pointer copies, the element size is considered to be 1. For pointer to CUDA array or vice versa copies, the element size is determined by the CUDA array. For CUDA array to CUDA array copies, the element size of the two CUDA arrays must match.

For a given operand, if CUmemcpy3DOperand::type is specified as [CU_MEMCPY_OPERAND_TYPE_POINTER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg507c7c67580a9b998fd8a59ce883c7d4c8f57f3e6c8b3863359d15cf89aaa593>), then CUmemcpy3DOperand::op::ptr will be used. The CUmemcpy3DOperand::op::ptr::ptr field must contain the pointer where the copy should begin. The CUmemcpy3DOperand::op::ptr::rowLength field specifies the length of each row in elements and must either be zero or be greater than or equal to the width of the copy specified in CUDA_MEMCPY3D_BATCH_OP::extent::width. The CUmemcpy3DOperand::op::ptr::layerHeight field specifies the height of each layer and must either be zero or be greater than or equal to the height of the copy specified in CUDA_MEMCPY3D_BATCH_OP::extent::height. When either of these values is zero, that aspect of the operand is considered to be tightly packed according to the copy extent. For managed memory pointers on devices where [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) is true or system-allocated pageable memory on devices where [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>) is true, the CUmemcpy3DOperand::op::ptr::locHint field can be used to hint the location of the operand.

If an operand's type is specified as [CU_MEMCPY_OPERAND_TYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg507c7c67580a9b998fd8a59ce883c7d416fd927c15c167e4e734e657f9aa323b>), then CUmemcpy3DOperand::op::array will be used. The CUmemcpy3DOperand::op::array::array field specifies the CUDA array and CUmemcpy3DOperand::op::array::offset specifies the 3D offset into that array where the copy begins.

The [CUmemcpyAttributes::srcAccessOrder](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1_1d152922b22834808ca5714f688400761>) indicates the source access ordering to be observed for copies associated with the attribute. If the source access order is set to [CU_MEMCPY_SRC_ACCESS_ORDER_STREAM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg88fcbe39c3714884fcd1ca9d3b9e425183a1e3dd6ef4b91364f645443d183c70>), then the source will be accessed in stream order. If the source access order is set to [CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg88fcbe39c3714884fcd1ca9d3b9e4251d05487493fc2a9ba99d4ccdc834a4266>) then it indicates that access to the source pointer can be out of stream order and all accesses must be complete before the API call returns. This flag is suited for ephemeral sources (ex., stack variables) when it's known that no prior operations in the stream can be accessing the memory and also that the lifetime of the memory is limited to the scope that the source variable was declared in. Specifying this flag allows the driver to optimize the copy and removes the need for the user to synchronize the stream after the API call. If the source access order is set to [CU_MEMCPY_SRC_ACCESS_ORDER_ANY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg88fcbe39c3714884fcd1ca9d3b9e4251f9cb69c7dc969e7f9350e5749b310314>) then it indicates that access to the source pointer can be out of stream order and the accesses can happen even after the API call returns. This flag is suited for host pointers allocated outside CUDA (ex., via malloc) when it's known that no prior operations in the stream can be accessing the memory. Specifying this flag allows the driver to optimize the copy on certain platforms. Each memcopy operation in `opList` must have a valid srcAccessOrder setting, otherwise this API will return [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

The [CUmemcpyAttributes::flags](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1_10c9ad8770c38c992894a6870991127e4>) field can be used to specify certain flags for copies. Setting the [CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg0badb1f4f792793fdcf4a78593b4655a4c2b776afc2c1570cc93d48984b4d794>) flag indicates that the associated copies should preferably overlap with any compute work. Note that this flag is a hint and can be ignored depending on the platform and other parameters of the copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy3DPeer ( const [CUDA_MEMCPY3D_PEER](<structCUDA__MEMCPY3D__PEER__v1.html#structCUDA__MEMCPY3D__PEER__v1>)*Â pCopy )


Copies memory between contexts.

######  Parameters

`pCopy`
    \- Parameters for the memory copy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Perform a 3D memory copy according to the parameters specified in `pCopy`. See the definition of the CUDA_MEMCPY3D_PEER structure for documentation of its parameters.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a> "Copies device memory between two contexts."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2> "Copies device memory between two contexts asynchronously."), [cuMemcpy3DPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8> "Copies memory between contexts asynchronously."), [cudaMemcpy3DPeer](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1geeab4601354962a5968eefc8b79ec2dd>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpy3DPeerAsync ( const [CUDA_MEMCPY3D_PEER](<structCUDA__MEMCPY3D__PEER__v1.html#structCUDA__MEMCPY3D__PEER__v1>)*Â pCopy, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory between contexts asynchronously.

######  Parameters

`pCopy`
    \- Parameters for the memory copy
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Perform a 3D memory copy according to the parameters specified in `pCopy`. See the definition of the CUDA_MEMCPY3D_PEER structure for documentation of its parameters.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a> "Copies device memory between two contexts."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2> "Copies device memory between two contexts asynchronously."), [cuMemcpy3DPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8> "Copies memory between contexts asynchronously."), [cudaMemcpy3DPeerAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7386b2845149b48c87f82ea017690aa8>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dst, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â src, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory asynchronously.

######  Parameters

`dst`
    \- Destination unified virtual address space pointer
`src`
    \- Source unified virtual address space pointer
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies data between two pointers. `dst` and `src` are base pointers of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy. Note that this function infers the type of the transfer (host to host, host to device, device to device, or device to host) from the pointer values. This function is only allowed in contexts which support unified addressing.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpyAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79>), [cudaMemcpyToSymbolAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd00b41ade29161aafbf6ff8aee3d6eb5>), [cudaMemcpyFromSymbolAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g2d9f7a440f1e522555dfe994245a5946>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyAtoA ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount )


Copies memory from Array to Array.

######  Parameters

`dstArray`
    \- Destination array
`dstOffset`
    \- Offset in bytes of destination array
`srcArray`
    \- Source array
`srcOffset`
    \- Offset in bytes of source array
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from one 1D CUDA array to another. `dstArray` and `srcArray` specify the handles of the destination and source CUDA arrays for the copy, respectively. `dstOffset` and `srcOffset` specify the destination and source offsets in bytes into the CUDA arrays. `ByteCount` is the number of bytes to be copied. The size of the elements in the CUDA arrays need not be the same format, but the elements must be the same size; and count must be evenly divisible by that size.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpyArrayToArray](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1g5daffa65811c6be7eba1ec3c6c19ddb0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyAtoD ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount )


Copies memory from Array to Device.

######  Parameters

`dstDevice`
    \- Destination device pointer
`srcArray`
    \- Source array
`srcOffset`
    \- Offset in bytes of source array
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from one 1D CUDA array to device memory. `dstDevice` specifies the base pointer of the destination and must be naturally aligned with the CUDA array elements. `srcArray` and `srcOffset` specify the CUDA array handle and the offset in bytes into the array where the copy is to begin. `ByteCount` specifies the number of bytes to copy and must be evenly divisible by the array element size.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpyFromArray](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1g6fbe8ed786061afaeaf79dc17eef15e9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyAtoH ( void*Â dstHost, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount )


Copies memory from Array to Host.

######  Parameters

`dstHost`
    \- Destination device pointer
`srcArray`
    \- Source array
`srcOffset`
    \- Offset in bytes of source array
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from one 1D CUDA array to host memory. `dstHost` specifies the base pointer of the destination. `srcArray` and `srcOffset` specify the CUDA array handle and starting offset in bytes of the source data. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpyFromArray](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1g6fbe8ed786061afaeaf79dc17eef15e9>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyAtoHAsync ( void*Â dstHost, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â srcArray, size_tÂ srcOffset, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory from Array to Host.

######  Parameters

`dstHost`
    \- Destination pointer
`srcArray`
    \- Source array
`srcOffset`
    \- Offset in bytes of source array
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies from one 1D CUDA array to host memory. `dstHost` specifies the base pointer of the destination. `srcArray` and `srcOffset` specify the CUDA array handle and starting offset in bytes of the source data. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpyFromArrayAsync](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1gfa22cfe6148b4c82593ecf3582f1dc33>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyBatchAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dsts, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â srcs, size_t*Â sizes, size_tÂ count, [CUmemcpyAttributes](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1>)*Â attrs, size_t*Â attrsIdxs, size_tÂ numAttrs, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Performs a batch of memory copies asynchronously.

######  Parameters

`dsts`
    \- Array of destination pointers.
`srcs`
    \- Array of memcpy source pointers.
`sizes`
    \- Array of sizes for memcpy operations.
`count`
    \- Size of `dsts`, `srcs` and `sizes` arrays
`attrs`
    \- Array of memcpy attributes.
`attrsIdxs`
    \- Array of indices to specify which copies each entry in the `attrs` array applies to. The attributes specified in attrs[k] will be applied to copies starting from attrsIdxs[k] through attrsIdxs[k+1] \- 1. Also attrs[numAttrs-1] will apply to copies starting from attrsIdxs[numAttrs-1] through count - 1.
`numAttrs`
    \- Size of `attrs` and `attrsIdxs` arrays.
`hStream`
    \- The stream to enqueue the operations in. Must not be legacy NULL stream.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Performs a batch of memory copies. The batch as a whole executes in stream order but copies within a batch are not guaranteed to execute in any specific order. This API only supports pointer-to-pointer copies. For copies involving CUDA arrays, please see [cuMemcpy3DBatchAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g97dd29d0e3490379a5cbdb21deb41f12> "Performs a batch of 3D memory copies asynchronously.").

Performs memory copies from source buffers specified in `srcs` to destination buffers specified in `dsts`. The size of each copy is specified in `sizes`. All three arrays must be of the same length as specified by `count`. Since there are no ordering guarantees for copies within a batch, specifying any dependent copies within a batch will result in undefined behavior.

Every copy in the batch has to be associated with a set of attributes specified in the `attrs` array. Each entry in this array can apply to more than one copy. This can be done by specifying in the `attrsIdxs` array, the index of the first copy that the corresponding entry in the `attrs` array applies to. Both `attrs` and `attrsIdxs` must be of the same length as specified by `numAttrs`. For example, if a batch has 10 copies listed in dst/src/sizes, the first 6 of which have one set of attributes and the remaining 4 another, then `numAttrs` will be 2, `attrsIdxs` will be {0, 6} and `attrs` will contains the two sets of attributes. Note that the first entry in `attrsIdxs` must always be 0. Also, each entry must be greater than the previous entry and the last entry should be less than `count`. Furthermore, `numAttrs` must be lesser than or equal to `count`.

The [CUmemcpyAttributes::srcAccessOrder](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1_1d152922b22834808ca5714f688400761>) indicates the source access ordering to be observed for copies associated with the attribute. If the source access order is set to [CU_MEMCPY_SRC_ACCESS_ORDER_STREAM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg88fcbe39c3714884fcd1ca9d3b9e425183a1e3dd6ef4b91364f645443d183c70>), then the source will be accessed in stream order. If the source access order is set to [CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg88fcbe39c3714884fcd1ca9d3b9e4251d05487493fc2a9ba99d4ccdc834a4266>) then it indicates that access to the source pointer can be out of stream order and all accesses must be complete before the API call returns. This flag is suited for ephemeral sources (ex., stack variables) when it's known that no prior operations in the stream can be accessing the memory and also that the lifetime of the memory is limited to the scope that the source variable was declared in. Specifying this flag allows the driver to optimize the copy and removes the need for the user to synchronize the stream after the API call. If the source access order is set to [CU_MEMCPY_SRC_ACCESS_ORDER_ANY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg88fcbe39c3714884fcd1ca9d3b9e4251f9cb69c7dc969e7f9350e5749b310314>) then it indicates that access to the source pointer can be out of stream order and the accesses can happen even after the API call returns. This flag is suited for host pointers allocated outside CUDA (ex., via malloc) when it's known that no prior operations in the stream can be accessing the memory. Specifying this flag allows the driver to optimize the copy on certain platforms. Each memcpy operation in the batch must have a valid CUmemcpyAttributes corresponding to it including the appropriate srcAccessOrder setting, otherwise the API will return [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>).

The [CUmemcpyAttributes::srcLocHint](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1_12b2ea7d3968f58243d6d94f99bab6a55>) and [CUmemcpyAttributes::dstLocHint](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1_15ef6cafc9e673f19d946cd5517dd05ec>) allows applications to specify hint locations for operands of a copy when the operand doesn't have a fixed location. That is, these hints are only applicable for managed memory pointers on devices where [CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a333110e44c9cb6ead02f03ff6f6fd495e>) is true or system-allocated pageable memory on devices where [CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a35fdcdbe1dfc3ad5ec428c279e0efb9cd>) is true. For other cases, these hints are ignored.

The [CUmemcpyAttributes::flags](<structCUmemcpyAttributes__v1.html#structCUmemcpyAttributes__v1_10c9ad8770c38c992894a6870991127e4>) field can be used to specify certain flags for copies. Setting the [CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg0badb1f4f792793fdcf4a78593b4655a4c2b776afc2c1570cc93d48984b4d794>) flag indicates that the associated copies should preferably overlap with any compute work. Note that this flag is a hint and can be ignored depending on the platform and other parameters of the copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyDtoA ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount )


Copies memory from Device to Array.

######  Parameters

`dstArray`
    \- Destination array
`dstOffset`
    \- Offset in bytes of destination array
`srcDevice`
    \- Source device pointer
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from device memory to a 1D CUDA array. `dstArray` and `dstOffset` specify the CUDA array handle and starting index of the destination data. `srcDevice` specifies the base pointer of the source. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpyToArray](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1g15b5d20cedf31dd13801c6015da0e828>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyDtoD ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount )


Copies memory from Device to Device.

######  Parameters

`dstDevice`
    \- Destination device pointer
`srcDevice`
    \- Source device pointer
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from device memory to device memory. `dstDevice` and `srcDevice` are the base pointers of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8>), [cudaMemcpyToSymbol](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g4561bf9c99d91c92684a91a0bd356bfe>), [cudaMemcpyFromSymbol](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g99db510d18d37fbb0f5c075a8caf3b5f>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyDtoDAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory from Device to Device.

######  Parameters

`dstDevice`
    \- Destination device pointer
`srcDevice`
    \- Source device pointer
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies from device memory to device memory. `dstDevice` and `srcDevice` are the base pointers of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpyAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79>), [cudaMemcpyToSymbolAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd00b41ade29161aafbf6ff8aee3d6eb5>), [cudaMemcpyFromSymbolAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g2d9f7a440f1e522555dfe994245a5946>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyDtoH ( void*Â dstHost, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount )


Copies memory from Device to Host.

######  Parameters

`dstHost`
    \- Destination host pointer
`srcDevice`
    \- Source device pointer
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from device to host memory. `dstHost` and `srcDevice` specify the base pointers of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8>), [cudaMemcpyFromSymbol](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g99db510d18d37fbb0f5c075a8caf3b5f>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyDtoHAsync ( void*Â dstHost, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory from Device to Host.

######  Parameters

`dstHost`
    \- Destination host pointer
`srcDevice`
    \- Source device pointer
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies from device to host memory. `dstHost` and `srcDevice` specify the base pointers of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpyAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79>), [cudaMemcpyFromSymbolAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g2d9f7a440f1e522555dfe994245a5946>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyHtoA ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, const void*Â srcHost, size_tÂ ByteCount )


Copies memory from Host to Array.

######  Parameters

`dstArray`
    \- Destination array
`dstOffset`
    \- Offset in bytes of destination array
`srcHost`
    \- Source host pointer
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from host memory to a 1D CUDA array. `dstArray` and `dstOffset` specify the CUDA array handle and starting offset in bytes of the destination data. `pSrc` specifies the base address of the source. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpyToArray](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1g15b5d20cedf31dd13801c6015da0e828>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyHtoAAsync ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â dstArray, size_tÂ dstOffset, const void*Â srcHost, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory from Host to Array.

######  Parameters

`dstArray`
    \- Destination array
`dstOffset`
    \- Offset in bytes of destination array
`srcHost`
    \- Source host pointer
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies from host memory to a 1D CUDA array. `dstArray` and `dstOffset` specify the CUDA array handle and starting offset in bytes of the destination data. `srcHost` specifies the base address of the source. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpyToArrayAsync](<../cuda-runtime-api/group__CUDART__MEMORY__DEPRECATED.html#group__CUDART__MEMORY__DEPRECATED_1g92f0eaaaa772fd428dfc7c7ba699d272>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyHtoD ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, const void*Â srcHost, size_tÂ ByteCount )


Copies memory from Host to Device.

######  Parameters

`dstDevice`
    \- Destination device pointer
`srcHost`
    \- Source host pointer
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from host memory to device memory. `dstDevice` and `srcHost` are the base addresses of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemcpy](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8>), [cudaMemcpyToSymbol](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g4561bf9c99d91c92684a91a0bd356bfe>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyHtoDAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, const void*Â srcHost, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies memory from Host to Device.

######  Parameters

`dstDevice`
    \- Destination device pointer
`srcHost`
    \- Source host pointer
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies from host memory to device memory. `dstDevice` and `srcHost` are the base addresses of the destination and source, respectively. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.

  * Memory regions requested must be either entirely registered with CUDA, or in the case of host pageable transfers, not registered at all. Memory regions spanning over allocations that are both registered and not registered with CUDA are not supported and will return CUDA_ERROR_INVALID_VALUE.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemcpyAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79>), [cudaMemcpyToSymbolAsync](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd00b41ade29161aafbf6ff8aee3d6eb5>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyPeer ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â dstContext, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â srcContext, size_tÂ ByteCount )


Copies device memory between two contexts.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstContext`
    \- Destination context
`srcDevice`
    \- Source device pointer
`srcContext`
    \- Source context
`ByteCount`
    \- Size of memory copy in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Copies from device memory in one context to device memory in another context. `dstDevice` is the base device pointer of the destination memory and `dstContext` is the destination context. `srcDevice` is the base device pointer of the source memory and `srcContext` is the source pointer. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [synchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-sync>) behavior for most use cases.


**See also:**

[cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpy3DPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1g11466fd70cde9329a4e16eb1f258c433> "Copies memory between contexts."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2> "Copies device memory between two contexts asynchronously."), [cuMemcpy3DPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8> "Copies memory between contexts asynchronously."), [cudaMemcpyPeer](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g88fd1245b2cb10d2d30c74900b7dfb9c>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemcpyPeerAsync ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â dstContext, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â srcDevice, [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â srcContext, size_tÂ ByteCount, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Copies device memory between two contexts asynchronously.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstContext`
    \- Destination context
`srcDevice`
    \- Source device pointer
`srcContext`
    \- Source context
`ByteCount`
    \- Size of memory copy in bytes
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Copies from device memory in one context to device memory in another context. `dstDevice` is the base device pointer of the destination memory and `dstContext` is the destination context. `srcDevice` is the base device pointer of the source memory and `srcContext` is the source pointer. `ByteCount` specifies the number of bytes to copy.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * This function exhibits [asynchronous](<api-sync-behavior.html#api-sync-behavior__memcpy-async>) behavior for most use cases.

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1ge1f5c7771544fee150ada8853c7cbf4a> "Copies device memory between two contexts."), [cuMemcpy3DPeer](<group__CUDA__MEM.html#group__CUDA__MEM_1g11466fd70cde9329a4e16eb1f258c433> "Copies memory between contexts."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpy3DPeerAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gc4e4bfd9f627d3aa3695979e058f1bb8> "Copies memory between contexts asynchronously."), [cudaMemcpyPeerAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gbfde4ace9ff4823f4ac45e5c6bdcd2ee>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD16 ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned shortÂ us, size_tÂ N )


Initializes device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`us`
    \- Value to set
`N`
    \- Number of elements

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the memory range of `N` 16-bit values to the specified value `us`. The `dstDevice` pointer must be two byte aligned.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD16Async ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned shortÂ us, size_tÂ N, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Sets device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`us`
    \- Value to set
`N`
    \- Number of elements
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the memory range of `N` 16-bit values to the specified value `us`. The `dstDevice` pointer must be two byte aligned.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemsetAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7c9761e21d9f0999fd136c51e7b9b2a0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD2D16 ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned shortÂ us, size_tÂ Width, size_tÂ Height )


Initializes device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstPitch`
    \- Pitch of destination device pointer(Unused if `Height` is 1)
`us`
    \- Value to set
`Width`
    \- Width of row
`Height`
    \- Number of rows

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the 2D memory range of `Width` 16-bit values to the specified value `us`. `Height` specifies the number of rows to set, and `dstPitch` specifies the number of bytes between each row. The `dstDevice` pointer and `dstPitch` offset must be two byte aligned. This function performs fastest when the pitch is one that has been passed back by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset2D](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g120112b2bd627c7a896390efadc4d2c1>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD2D16Async ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned shortÂ us, size_tÂ Width, size_tÂ Height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Sets device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstPitch`
    \- Pitch of destination device pointer(Unused if `Height` is 1)
`us`
    \- Value to set
`Width`
    \- Width of row
`Height`
    \- Number of rows
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the 2D memory range of `Width` 16-bit values to the specified value `us`. `Height` specifies the number of rows to set, and `dstPitch` specifies the number of bytes between each row. The `dstDevice` pointer and `dstPitch` offset must be two byte aligned. This function performs fastest when the pitch is one that has been passed back by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset2DAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8fdcc53996ff49c570f4b5ead0256ef0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD2D32 ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned int Â ui, size_tÂ Width, size_tÂ Height )


Initializes device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstPitch`
    \- Pitch of destination device pointer(Unused if `Height` is 1)
`ui`
    \- Value to set
`Width`
    \- Width of row
`Height`
    \- Number of rows

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the 2D memory range of `Width` 32-bit values to the specified value `ui`. `Height` specifies the number of rows to set, and `dstPitch` specifies the number of bytes between each row. The `dstDevice` pointer and `dstPitch` offset must be four byte aligned. This function performs fastest when the pitch is one that has been passed back by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset2D](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g120112b2bd627c7a896390efadc4d2c1>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD2D32Async ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned int Â ui, size_tÂ Width, size_tÂ Height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Sets device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstPitch`
    \- Pitch of destination device pointer(Unused if `Height` is 1)
`ui`
    \- Value to set
`Width`
    \- Width of row
`Height`
    \- Number of rows
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the 2D memory range of `Width` 32-bit values to the specified value `ui`. `Height` specifies the number of rows to set, and `dstPitch` specifies the number of bytes between each row. The `dstDevice` pointer and `dstPitch` offset must be four byte aligned. This function performs fastest when the pitch is one that has been passed back by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset2DAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8fdcc53996ff49c570f4b5ead0256ef0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD2D8 ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned char Â uc, size_tÂ Width, size_tÂ Height )


Initializes device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstPitch`
    \- Pitch of destination device pointer(Unused if `Height` is 1)
`uc`
    \- Value to set
`Width`
    \- Width of row
`Height`
    \- Number of rows

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the 2D memory range of `Width` 8-bit values to the specified value `uc`. `Height` specifies the number of rows to set, and `dstPitch` specifies the number of bytes between each row. This function performs fastest when the pitch is one that has been passed back by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset2D](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g120112b2bd627c7a896390efadc4d2c1>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD2D8Async ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, size_tÂ dstPitch, unsigned char Â uc, size_tÂ Width, size_tÂ Height, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Sets device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`dstPitch`
    \- Pitch of destination device pointer(Unused if `Height` is 1)
`uc`
    \- Value to set
`Width`
    \- Width of row
`Height`
    \- Number of rows
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the 2D memory range of `Width` 8-bit values to the specified value `uc`. `Height` specifies the number of rows to set, and `dstPitch` specifies the number of bytes between each row. This function performs fastest when the pitch is one that has been passed back by [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.").

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset2DAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g8fdcc53996ff49c570f4b5ead0256ef0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD32 ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned int Â ui, size_tÂ N )


Initializes device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`ui`
    \- Value to set
`N`
    \- Number of elements

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the memory range of `N` 32-bit values to the specified value `ui`. The `dstDevice` pointer must be four byte aligned.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD32Async ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned int Â ui, size_tÂ N, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Sets device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`ui`
    \- Value to set
`N`
    \- Number of elements
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the memory range of `N` 32-bit values to the specified value `ui`. The `dstDevice` pointer must be four byte aligned.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cudaMemsetAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7c9761e21d9f0999fd136c51e7b9b2a0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD8 ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned char Â uc, size_tÂ N )


Initializes device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`uc`
    \- Value to set
`N`
    \- Number of elements

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the memory range of `N` 8-bit values to the specified value `uc`.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gaef08a7ccd61112f94e82f2b30d43627> "Sets device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemset](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gf7338650f7683c51ee26aadc6973c63a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMemsetD8Async ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dstDevice, unsigned char Â uc, size_tÂ N, [CUstream](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gb946c7f02e09efd788a204718015d88a>)Â hStream )


Sets device memory.

######  Parameters

`dstDevice`
    \- Destination device pointer
`uc`
    \- Value to set
`N`
    \- Number of elements
`hStream`
    \- Stream identifier

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets the memory range of `N` 8-bit values to the specified value `uc`.

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * See also [memset synchronization details](<api-sync-behavior.html#api-sync-behavior__memset>).

  * This function uses standard [default stream](<stream-sync-behavior.html#stream-sync-behavior__default-stream>) semantics.


**See also:**

[cuArray3DCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1gc2322c70b38c2984536c90ed118bb1d7> "Creates a 3D CUDA array."), [cuArray3DGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1gb58549f2f3f390b9e0e7c8f3acd53857> "Get a 3D CUDA array descriptor."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cuArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1g982878affbbc023de84874faac838b0b> "Destroys a CUDA array."), [cuArrayGetDescriptor](<group__CUDA__MEM.html#group__CUDA__MEM_1g661fe823dbd37bf11f82a71bd4762acf> "Get a 1D or 2D CUDA array descriptor."), [cuMemAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), [cuMemAllocHost](<group__CUDA__MEM.html#group__CUDA__MEM_1gdd8311286d2c2691605362c689bc64e0> "Allocates page-locked host memory."), [cuMemAllocPitch](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory."), [cuMemcpy2D](<group__CUDA__MEM.html#group__CUDA__MEM_1g27f885b30c34cc20a663a671dbf6fc27> "Copies memory for 2D arrays."), [cuMemcpy2DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g4acf155faeb969d9d21f5433d3d0f274> "Copies memory for 2D arrays."), [cuMemcpy2DUnaligned](<group__CUDA__MEM.html#group__CUDA__MEM_1g2fa285d47fd7020e596bfeab3deb651b> "Copies memory for 2D arrays."), [cuMemcpy3D](<group__CUDA__MEM.html#group__CUDA__MEM_1g4b5238975579f002c0199a3800ca44df> "Copies memory for 3D arrays."), [cuMemcpy3DAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g79f4f3fde6ae0f529568d881d9e11987> "Copies memory for 3D arrays."), [cuMemcpyAtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gf81b218c984a31436ec9e23a85fb604a> "Copies memory from Array to Array."), [cuMemcpyAtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g825b3f037f7f51382cae991bae8173fd> "Copies memory from Array to Device."), [cuMemcpyAtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1gf7ad1edb2539cccc352c6b8b76f657f4> "Copies memory from Array to Host."), [cuMemcpyAtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g64cbd2e60436699aebdd0bdbf14d0f01> "Copies memory from Array to Host."), [cuMemcpyDtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1gad6827247af91600b56ce6e2ddb802e1> "Copies memory from Device to Array."), [cuMemcpyDtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g1725774abf8b51b91945f3336b778c8b> "Copies memory from Device to Device."), [cuMemcpyDtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g39ea09ba682b8eccc9c3e0c04319b5c8> "Copies memory from Device to Device."), [cuMemcpyDtoH](<group__CUDA__MEM.html#group__CUDA__MEM_1g3480368ee0208a98f75019c9a8450893> "Copies memory from Device to Host."), [cuMemcpyDtoHAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362> "Copies memory from Device to Host."), [cuMemcpyHtoA](<group__CUDA__MEM.html#group__CUDA__MEM_1g57d3d780d165ecc0e3b3ce08e141cd89> "Copies memory from Host to Array."), [cuMemcpyHtoAAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1gb5c4863f64f132b4bc2661818b3fd188> "Copies memory from Host to Array."), [cuMemcpyHtoD](<group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169> "Copies memory from Host to Device."), [cuMemcpyHtoDAsync](<group__CUDA__MEM.html#group__CUDA__MEM_1g1572263fe2597d7ba4f6964597a354a3> "Copies memory from Host to Device."), [cuMemFree](<group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a> "Frees device memory."), [cuMemFreeHost](<group__CUDA__MEM.html#group__CUDA__MEM_1g62e0fdbe181dab6b1c90fa1a51c7b92c> "Frees page-locked host memory."), [cuMemGetAddressRange](<group__CUDA__MEM.html#group__CUDA__MEM_1g64fee5711274a2a0573a789c94d8299b> "Get information on memory allocations."), [cuMemGetInfo](<group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0> "Gets free and total memory."), [cuMemHostAlloc](<group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9> "Allocates page-locked host memory."), [cuMemHostGetDevicePointer](<group__CUDA__MEM.html#group__CUDA__MEM_1g57a39e5cba26af4d06be67fc77cc62f0> "Passes back device pointer of mapped pinned memory."), [cuMemsetD2D8](<group__CUDA__MEM.html#group__CUDA__MEM_1ge88b13e646e2be6ba0e0475ef5205974> "Initializes device memory."), [cuMemsetD2D8Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g3f7b6924a3e49c3265b328f534102e97> "Sets device memory."), [cuMemsetD2D16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7f561a15a66144fa9f6ab5350edc8a30> "Initializes device memory."), [cuMemsetD2D16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g64ee197befac3d74d9fefedcf6ef6b10> "Sets device memory."), [cuMemsetD2D32](<group__CUDA__MEM.html#group__CUDA__MEM_1g74b359b2d026bfeb7c795b5038d07523> "Initializes device memory."), [cuMemsetD2D32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g8a78d3147ac93fac955052c815d9ea3c> "Sets device memory."), [cuMemsetD8](<group__CUDA__MEM.html#group__CUDA__MEM_1g6e582bf866e9e2fb014297bfaf354d7b> "Initializes device memory."), [cuMemsetD16](<group__CUDA__MEM.html#group__CUDA__MEM_1g7d805e610054392a4d11e8a8bf5eb35c> "Initializes device memory."), [cuMemsetD16Async](<group__CUDA__MEM.html#group__CUDA__MEM_1gf731438877dd8ec875e4c43d848c878c> "Sets device memory."), [cuMemsetD32](<group__CUDA__MEM.html#group__CUDA__MEM_1g983e8d8759acd1b64326317481fbf132> "Initializes device memory."), [cuMemsetD32Async](<group__CUDA__MEM.html#group__CUDA__MEM_1g58229da5d30f1c0cdf667b320ec2c0f5> "Sets device memory."), [cudaMemsetAsync](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7c9761e21d9f0999fd136c51e7b9b2a0>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMipmappedArrayCreate ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â pHandle, const [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>)*Â pMipmappedArrayDesc, unsigned int Â numMipmapLevels )


Creates a CUDA mipmapped array.

######  Parameters

`pHandle`
    \- Returned mipmapped array
`pMipmappedArrayDesc`
    \- mipmapped array descriptor
`numMipmapLevels`
    \- Number of mipmap levels

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Creates a CUDA mipmapped array according to the CUDA_ARRAY3D_DESCRIPTOR structure `pMipmappedArrayDesc` and returns a handle to the new CUDA mipmapped array in `*pHandle`. `numMipmapLevels` specifies the number of mipmap levels to be allocated. This value is clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].

The CUDA_ARRAY3D_DESCRIPTOR is defined as:


    â    typedef struct {
                  unsigned int Width;
                  unsigned int Height;
                  unsigned int Depth;
                  [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) Format;
                  unsigned int NumChannels;
                  unsigned int Flags;
              } [CUDA_ARRAY3D_DESCRIPTOR](<structCUDA__ARRAY3D__DESCRIPTOR__v2.html#structCUDA__ARRAY3D__DESCRIPTOR__v2>);

where:

  * `Width`, `Height`, and `Depth` are the width, height, and depth of the CUDA array (in elements); the following types of CUDA arrays can be allocated:
    * A 1D mipmapped array is allocated if `Height` and `Depth` extents are both zero.

    * A 2D mipmapped array is allocated if only `Depth` extent is zero.

    * A 3D mipmapped array is allocated if all three extents are non-zero.

    * A 1D layered CUDA mipmapped array is allocated if only `Height` is zero and the [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flag is set. Each layer is a 1D array. The number of layers is determined by the depth extent.

    * A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and the [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flag is set. Each layer is a 2D array. The number of layers is determined by the depth extent.

    * A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>) flag is set. `Width` must be equal to `Height`, and `Depth` must be six. A cubemap is a special type of 2D layered CUDA array, where the six layers represent the six faces of a cube. The order of the six layers in memory is the same as that listed in [CUarray_cubemap_face](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g012fda14b50e7db8798a340627c4c330>).

    * A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero, and both, [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>) and [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flags are set. `Width` must be equal to `Height`, and `Depth` must be a multiple of six. A cubemap layered CUDA array is a special type of 2D layered CUDA array that consists of a collection of cubemaps. The first six layers represent the first cubemap, the next six layers form the second cubemap, and so on.


  * Format specifies the format of the elements; [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) is defined as:

        â    typedef enum CUarray_format_enum {
                      [CU_AD_FORMAT_UNSIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e0af5f5a0ffa8e16a5c720364ccd5dac>) = 0x01,
                      [CU_AD_FORMAT_UNSIGNED_INT16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9d0f11e851e891af6f204cf05503ba525>) = 0x02,
                      [CU_AD_FORMAT_UNSIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f952b891ad5d4080db0fb2e23fe71614a0>) = 0x03,
                      [CU_AD_FORMAT_SIGNED_INT8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9376b799ee12ce9e1de0c34cfa7839284>) = 0x08,
                      [CU_AD_FORMAT_SIGNED_INT16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f980864598b1579bd90fab79369072478f>) = 0x09,
                      [CU_AD_FORMAT_SIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f96db055c31d053bd1d5ebbaa98de2bad3>) = 0x0a,
                      [CU_AD_FORMAT_HALF](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f995c97289b540ff36334722ec745f53a3>) = 0x10,
                      [CU_AD_FORMAT_FLOAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98140f3b0de3d87bdbf26964c24840f3c>) = 0x20,
                      [CU_AD_FORMAT_NV12](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f964889c93ccc518395eb985203735d40c>) = 0xb0,
                      [CU_AD_FORMAT_UNORM_INT8X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90dcb720ef3238f279ebd5a7eb7284137>) = 0xc0,
                      [CU_AD_FORMAT_UNORM_INT8X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e23c25eb679dd70676bd35b26041d21f>) = 0xc1,
                      [CU_AD_FORMAT_UNORM_INT8X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f939e6604652c4f7dfda35ef89bcf6a1c4>) = 0xc2,
                      [CU_AD_FORMAT_UNORM_INT16X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9a593cb744213ab457d4ebaa261879816>) = 0xc3,
                      [CU_AD_FORMAT_UNORM_INT16X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9fe334f0b162fd9ad3caad37a8c879d95>) = 0xc4,
                      [CU_AD_FORMAT_UNORM_INT16X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f965401cdeebbc53f7b02400ba14f940a4>) = 0xc5,
                      [CU_AD_FORMAT_SNORM_INT8X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f92943f83ded303df264a79ee11d1db0>) = 0xc6,
                      [CU_AD_FORMAT_SNORM_INT8X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8194990a6e17d78be0de66deffdf02f>) = 0xc7,
                      [CU_AD_FORMAT_SNORM_INT8X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9117a2e043203748187605ff8a71c2d1d>) = 0xc8,
                      [CU_AD_FORMAT_SNORM_INT16X1](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f95026e5e8783752bf8d3601dd4dbceb4c>) = 0xc9,
                      [CU_AD_FORMAT_SNORM_INT16X2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f939f633274a07dbce442325c5d90bf294>) = 0xca,
                      [CU_AD_FORMAT_SNORM_INT16X4](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f99acc19038dc1e68170e485f739912d49>) = 0xcb,
                      [CU_AD_FORMAT_BC1_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9299d155257aa3c0b75634d9f9b1bfa72>) = 0x91,
                      [CU_AD_FORMAT_BC1_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9afa18b300eb91ff879532a55d5aa191b>) = 0x92,
                      [CU_AD_FORMAT_BC2_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f92b9cf00f8c6012ec679654c9f012a267>) = 0x93,
                      [CU_AD_FORMAT_BC2_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c38b5af7926b020202562d67ba7529c2>) = 0x94,
                      [CU_AD_FORMAT_BC3_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8473614347359cc74574899e2e65012>) = 0x95,
                      [CU_AD_FORMAT_BC3_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94d158239dd6c825b4bd383ed66625257>) = 0x96,
                      [CU_AD_FORMAT_BC4_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9fe7527dfa2576595eea7463a1140058c>) = 0x97,
                      [CU_AD_FORMAT_BC4_SNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90627520c6fc707d63e9d3c66d307eec6>) = 0x98,
                      [CU_AD_FORMAT_BC5_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94b524073942ab7460b68a98da955e59e>) = 0x99,
                      [CU_AD_FORMAT_BC5_SNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f907466b7b5b3d897a58fac1e9d2db163e>) = 0x9a,
                      [CU_AD_FORMAT_BC6H_UF16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f94085af463b118d564873b8d275ac7912>) = 0x9b,
                      [CU_AD_FORMAT_BC6H_SF16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f938c1d137a2663d5ddca5ae6aa49f612e>) = 0x9c,
                      [CU_AD_FORMAT_BC7_UNORM](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f03f9cbeee0911d3c77c08e6f5c7ff62>) = 0x9d,
                      [CU_AD_FORMAT_BC7_UNORM_SRGB](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9ad5b0e1cd964cbd46270223f35651677>) = 0x9e,
                      [CU_AD_FORMAT_P010](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f0efd5417115904eb086f1df0046582e>) = 0x9f,
                      [CU_AD_FORMAT_P016](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9b8511f36d0a010b8846c84309d8920d5>) = 0xa1,
                      [CU_AD_FORMAT_NV16](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e41b4351cb805f35130636b0aafca609>) = 0xa2,
                      [CU_AD_FORMAT_P210](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9dc77f21be8b4ff4f23dcd450c3656409>) = 0xa3,
                      [CU_AD_FORMAT_P216](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9a42b3a04e2f30a93e50d7d68026f1ba9>) = 0xa4,
                      [CU_AD_FORMAT_YUY2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f961f757ee5f5c125b7be70e5b562826dc>) = 0xa5,
                      [CU_AD_FORMAT_Y210](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9796d73bbcb63216f7dd4cc4d8016b74c>) = 0xa6,
                      [CU_AD_FORMAT_Y216](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c2081981924fa204383f1ee05de74d8e>) = 0xa7,
                      [CU_AD_FORMAT_AYUV](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f99dd6d4cac84e541d2b1ad34b263bc1bc>) = 0xa8,
                      [CU_AD_FORMAT_Y410](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f91917374a5915ee6a5e1ed23c57f43b75>) = 0xa9,
                      [CU_AD_FORMAT_Y416](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f941bdcafb69e249176af2e1cc5d6178be>) = 0xb1,
                      [CU_AD_FORMAT_Y444_PLANAR8](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f93af0d614c7c240194c402b6ca9b4909f>) = 0xb2,
                      [CU_AD_FORMAT_Y444_PLANAR10](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f916b3c04a1fba1562d548d4504f06a7aa>) = 0xb3,
                      [CU_AD_FORMAT_YUV444_8bit_SemiPlanar](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f945b7c15d0c8a42d569b20509e7e54e1d>) = 0xb4,
                      [CU_AD_FORMAT_YUV444_16bit_SemiPlanar](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9cb653da3339b76b267a6fa8085513017>) = 0xb5,
                      [CU_AD_FORMAT_UNORM_INT_101010_2](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f93b3828154e807c69a6e0c7e0d54d31ea>) = 0x50,
                      [CU_AD_FORMAT_UINT8_PACKED_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9c3e98982827e44204ed4a4d41031c135>) = 0x51,
                      [CU_AD_FORMAT_UINT8_PACKED_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90cb5a5812940939b5f0eb0242a2146e7>) = 0x52,
                      [CU_AD_FORMAT_UINT8_SEMIPLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9f212e89e226ee1d69ca4a47fba3c39c3>) = 0x53,
                      [CU_AD_FORMAT_UINT16_SEMIPLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9e076ab1291241ef4c6c149b23321e1b5>) = 0x54,
                      [CU_AD_FORMAT_UINT8_SEMIPLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f959c089e74d94f90973118ef287d4f352>) = 0x55,
                      [CU_AD_FORMAT_UINT16_SEMIPLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f97f759c17c105bfa6fb486502fad3705e>) = 0x56,
                      [CU_AD_FORMAT_UINT8_SEMIPLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90548d7ee27a7f9401064a4d3b3dfc528>) = 0x57,
                      [CU_AD_FORMAT_UINT16_SEMIPLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f90a4038e90a78ae9bf495d043c39a9e29>) = 0x58,
                      [CU_AD_FORMAT_UINT8_PLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9d0f3182f705a6615ac5299de395cace8>) = 0x59,
                      [CU_AD_FORMAT_UINT16_PLANAR_420](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f97a9888731c72732a6880a18ef2f082cb>) = 0x5a,
                      [CU_AD_FORMAT_UINT8_PLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f932bb01870f63cd2d9949ccb2ea235ef1>) = 0x5b,
                      [CU_AD_FORMAT_UINT16_PLANAR_422](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f97a9a4abfb8ce20c04ddb47925cdf3752>) = 0x5c,
                      [CU_AD_FORMAT_UINT8_PLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f9922d032a80ac1229a05dae35c18c4b2e>) = 0x5d,
                      [CU_AD_FORMAT_UINT16_PLANAR_444](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f98ff25a5b7ff451a608b2ecca340b1f71>) = 0x5e,
                  } [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>);


  * `NumChannels` specifies the number of packed components per CUDA array element; it may be 1, 2, or 4;


  * Flags may be set to
    * [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) to enable creation of layered CUDA mipmapped arrays. If this flag is set, `Depth` specifies the number of layers, not the depth of a 3D array.

    * [CUDA_ARRAY3D_SURFACE_LDST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7287c43cacf1ed05865d6bcad1a23cd9>) to enable surface references to be bound to individual mipmap levels of the CUDA mipmapped array. If this flag is not set, [cuSurfRefSetArray](<group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED_1g68abcde159fa897b1dfb23387926dd66> "Sets the CUDA array for a surface reference.") will fail when attempting to bind a mipmap level of the CUDA mipmapped array to a surface reference.

    * [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>) to enable creation of mipmapped cubemaps. If this flag is set, `Width` must be equal to `Height`, and `Depth` must be six. If the [CUDA_ARRAY3D_LAYERED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge4adf555c51852623a3dea962ab8ee85>) flag is also set, then `Depth` must be a multiple of six.

    * [CUDA_ARRAY3D_TEXTURE_GATHER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0c929c92a91f4d9f9f49bae0131a6ccf>) to indicate that the CUDA mipmapped array will be used for texture gather. Texture gather can only be performed on 2D CUDA mipmapped arrays.


`Width`, `Height` and `Depth` must meet certain size requirements as listed in the following table. All values are specified in elements. Note that for brevity's sake, the full name of the device attribute is not specified. For ex., TEXTURE1D_MIPMAPPED_WIDTH refers to the device attribute [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a373cf80abc3969a6a6a9be1b9f36d7f18>).

**CUDA array type** |  **Valid extents that must always be met {(width range in elements), (height range), (depth range)}** |  **Valid extents with CUDA_ARRAY3D_SURFACE_LDST set {(width range in elements), (height range), (depth range)}**
---|---|---
1D  |  { (1,TEXTURE1D_MIPMAPPED_WIDTH), 0, 0 }  |  { (1,SURFACE1D_WIDTH), 0, 0 }
2D  |  { (1,TEXTURE2D_MIPMAPPED_WIDTH), (1,TEXTURE2D_MIPMAPPED_HEIGHT), 0 }  |  { (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }
3D  |  { (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) } OR { (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE), (1,TEXTURE3D_DEPTH_ALTERNATE) }  |  { (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT), (1,SURFACE3D_DEPTH) }
1D Layered  |  { (1,TEXTURE1D_LAYERED_WIDTH), 0, (1,TEXTURE1D_LAYERED_LAYERS) }  |  { (1,SURFACE1D_LAYERED_WIDTH), 0, (1,SURFACE1D_LAYERED_LAYERS) }
2D Layered  |  { (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT), (1,TEXTURE2D_LAYERED_LAYERS) }  |  { (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT), (1,SURFACE2D_LAYERED_LAYERS) }
Cubemap  |  { (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }  |  { (1,SURFACECUBEMAP_WIDTH), (1,SURFACECUBEMAP_WIDTH), 6 }
Cubemap Layered  |  { (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_LAYERS) }  |  { (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_LAYERS) }

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMipmappedArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1ge0d7c768b6a6963c4d4bde5bbc74f0ad> "Destroys a CUDA mipmapped array."), [cuMipmappedArrayGetLevel](<group__CUDA__MEM.html#group__CUDA__MEM_1g82f276659f05be14820e99346b0f86b7> "Gets a mipmap level of a CUDA mipmapped array."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cudaMallocMipmappedArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g9abd550dd3f655473d2640dc85be9774>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMipmappedArrayDestroy ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â hMipmappedArray )


Destroys a CUDA mipmapped array.

######  Parameters

`hMipmappedArray`
    \- Mipmapped array to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_ARRAY_IS_MAPPED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b668b10d56232c51b67db40516cc6b5b>), [CUDA_ERROR_CONTEXT_IS_DESTROYED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9b27ac43f7ce8446f5c9636dd73fb2139>)

###### Description

Destroys the CUDA mipmapped array `hMipmappedArray`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMipmappedArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1ga5d2e311c7f9b0bc6d130af824a40bd3> "Creates a CUDA mipmapped array."), [cuMipmappedArrayGetLevel](<group__CUDA__MEM.html#group__CUDA__MEM_1g82f276659f05be14820e99346b0f86b7> "Gets a mipmap level of a CUDA mipmapped array."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cudaFreeMipmappedArray](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g904669241eac5bdbfb410eb4124e4924>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMipmappedArrayGetLevel ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â pLevelArray, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â hMipmappedArray, unsigned int Â level )


Gets a mipmap level of a CUDA mipmapped array.

######  Parameters

`pLevelArray`
    \- Returned mipmap level CUDA array
`hMipmappedArray`
    \- CUDA mipmapped array
`level`
    \- Mipmap level

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

Returns in `*pLevelArray` a CUDA array that represents a single mipmap level of the CUDA mipmapped array `hMipmappedArray`.

If `level` is greater than the maximum number of levels in this mipmapped array, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuMipmappedArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1ga5d2e311c7f9b0bc6d130af824a40bd3> "Creates a CUDA mipmapped array."), [cuMipmappedArrayDestroy](<group__CUDA__MEM.html#group__CUDA__MEM_1ge0d7c768b6a6963c4d4bde5bbc74f0ad> "Destroys a CUDA mipmapped array."), [cuArrayCreate](<group__CUDA__MEM.html#group__CUDA__MEM_1g4192ff387a81c3bd5ed8c391ed62ca24> "Creates a 1D or 2D CUDA array."), [cudaGetMipmappedArrayLevel](<../cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g7086e6f81e6dda1ddf4cdb6c1764094a>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMipmappedArrayGetMemoryRequirements ( [CUDA_ARRAY_MEMORY_REQUIREMENTS](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1>)*Â memoryRequirements, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â mipmap, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â device )


Returns the memory requirements of a CUDA mipmapped array.

######  Parameters

`memoryRequirements`
    \- Pointer to CUDA_ARRAY_MEMORY_REQUIREMENTS
`mipmap`
    \- CUDA mipmapped array to get the memory requirements of
`device`
    \- Device to get the memory requirements for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the memory requirements of a CUDA mipmapped array in `memoryRequirements` If the CUDA mipmapped array is not allocated with flag [CUDA_ARRAY3D_DEFERRED_MAPPING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g854c29dbc47d04a4e42863cb87487d55>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned.

The returned value in [CUDA_ARRAY_MEMORY_REQUIREMENTS::size](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1_17a2851735a1d2c11af797f01b1d4969e>) represents the total size of the CUDA mipmapped array. The returned value in [CUDA_ARRAY_MEMORY_REQUIREMENTS::alignment](<structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1.html#structCUDA__ARRAY__MEMORY__REQUIREMENTS__v1_135c6c8106451b6313d1dffe9a28af755>) represents the alignment necessary for mapping the CUDA mipmapped array.

**See also:**

[cuArrayGetMemoryRequirements](<group__CUDA__MEM.html#group__CUDA__MEM_1gac8761ced0fa462e4762f6528073d9f4> "Returns the memory requirements of a CUDA array."), [cuMemMapArrayAsync](<group__CUDA__VA.html#group__CUDA__VA_1g5dc41a62a9feb68f2e943b438c83e5ab> "Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuMipmappedArrayGetSparseProperties ( [CUDA_ARRAY_SPARSE_PROPERTIES](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1>)*Â sparseProperties, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â mipmap )


Returns the layout properties of a sparse CUDA mipmapped array.

######  Parameters

`sparseProperties`
    \- Pointer to CUDA_ARRAY_SPARSE_PROPERTIES
`mipmap`
    \- CUDA mipmapped array to get the sparse properties of

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the sparse array layout properties in `sparseProperties` If the CUDA mipmapped array is not allocated with flag [CUDA_ARRAY3D_SPARSE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8e13c9d3ef98d1f3dce95901a115abc2>)[CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned.

For non-layered CUDA mipmapped arrays, [CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1895ecb42681678271b0edba05bf1dcd9>) returns the size of the mip tail region. The mip tail region includes all mip levels whose width, height or depth is less than that of the tile. For layered CUDA mipmapped arrays, if [CUDA_ARRAY_SPARSE_PROPERTIES::flags](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_10e842bb64091fa47809112c700cb5f0a>) contains [CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g0dcf4ba7e64caa5c1aa4e88caa7f659a>), then [CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1895ecb42681678271b0edba05bf1dcd9>) specifies the size of the mip tail of all layers combined. Otherwise, [CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1895ecb42681678271b0edba05bf1dcd9>) specifies mip tail size per layer. The returned value of [CUDA_ARRAY_SPARSE_PROPERTIES::miptailFirstLevel](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1edd0cca8fad1fcbb1789d537edd7e6b6>) is valid only if [CUDA_ARRAY_SPARSE_PROPERTIES::miptailSize](<structCUDA__ARRAY__SPARSE__PROPERTIES__v1.html#structCUDA__ARRAY__SPARSE__PROPERTIES__v1_1895ecb42681678271b0edba05bf1dcd9>) is non-zero.

**See also:**

[cuArrayGetSparseProperties](<group__CUDA__MEM.html#group__CUDA__MEM_1gf74df88a07404ee051f0e5b36647d8c7> "Returns the layout properties of a sparse CUDA array."), [cuMemMapArrayAsync](<group__CUDA__VA.html#group__CUDA__VA_1g5dc41a62a9feb68f2e943b438c83e5ab> "Maps or unmaps subregions of sparse CUDA arrays and sparse CUDA mipmapped arrays.")

* * *
