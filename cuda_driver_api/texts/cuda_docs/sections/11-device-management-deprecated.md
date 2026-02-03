# Device Management (Deprecated)

## 6.6.Â Device Management [DEPRECATED]

This section describes the device management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceComputeCapability](<#group__CUDA__DEVICE__DEPRECATED_1gdc50ce6a6e0a593158d4ccb3567e0545>) ( int*Â major, int*Â minor, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns the compute capability of the device.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetProperties](<#group__CUDA__DEVICE__DEPRECATED_1ged20a6d946d0217b3b1e0a40df6a43a6>) ( [CUdevprop](<structCUdevprop__v1.html#structCUdevprop__v1>)*Â prop, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )
     Returns properties for a selected device.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceComputeCapability ( int*Â major, int*Â minor, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns the compute capability of the device.

######  Parameters

`major`
    \- Major revision number
`minor`
    \- Minor revision number
`dev`
    \- Device handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000003>)

This function was deprecated as of CUDA 5.0 and its functionality superseded by [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.").

###### Description

Returns in `*major` and `*minor` the major and minor revision numbers that define the compute capability of the device `dev`.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetProperties ( [CUdevprop](<structCUdevprop__v1.html#structCUdevprop__v1>)*Â prop, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev )


Returns properties for a selected device.

######  Parameters

`prop`
    \- Returned properties of device
`dev`
    \- Device to get properties for

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000002>)

This function was deprecated as of CUDA 5.0 and replaced by [cuDeviceGetAttribute()](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device.").

###### Description

Returns in `*prop` the properties of device `dev`. The CUdevprop structure is defined as:


    â     typedef struct CUdevprop_st {
               int maxThreadsPerBlock;
               int maxThreadsDim[3];
               int maxGridSize[3];
               int sharedMemPerBlock;
               int totalConstantMemory;
               int SIMDWidth;
               int memPitch;
               int regsPerBlock;
               int clockRate;
               int textureAlign
            } [CUdevprop](<structCUdevprop__v1.html#structCUdevprop__v1>);

where:

  * maxThreadsPerBlock is the maximum number of threads per block;

  * maxThreadsDim[3] is the maximum sizes of each dimension of a block;

  * maxGridSize[3] is the maximum sizes of each dimension of a grid;

  * sharedMemPerBlock is the total amount of shared memory available per block in bytes;

  * totalConstantMemory is the total amount of constant memory available on the device in bytes;

  * SIMDWidth is the warp size;

  * memPitch is the maximum pitch allowed by the memory copy functions that involve memory regions allocated through [cuMemAllocPitch()](<group__CUDA__MEM.html#group__CUDA__MEM_1gcbe9b033f6c4de80f63cc6e58ed9a45a> "Allocates pitched device memory.");

  * regsPerBlock is the total number of registers available per block;

  * clockRate is the clock frequency in kilohertz;

  * textureAlign is the alignment requirement; texture base addresses that are aligned to textureAlign bytes do not need an offset applied to texture fetches.


Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetAttribute](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266> "Returns information about the device."), [cuDeviceGetCount](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74> "Returns the number of compute-capable devices."), [cuDeviceGetName](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gef75aa30df95446a845f2a7b9fffbb7f> "Returns an identifier string for the device."), [cuDeviceGetUuid](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g987b46b884c101ed5be414ab4d9e60e4> "Return an UUID for the device."), [cuDeviceGet](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g8bdd1cc7201304b01357b8034f6587cb> "Returns a handle to a compute device."), [cuDeviceTotalMem](<group__CUDA__DEVICE.html#group__CUDA__DEVICE_1gc6a0d6551335a3780f9f3c967a0fde5d> "Returns the total amount of memory on the device.")

* * *
