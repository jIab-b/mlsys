# Version Management

## 6.4.Â Version Management

This section describes the version management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDriverGetVersion](<#group__CUDA__VERSION_1g8b7a10395392e049006e61bcdc8ebe71>) ( int*Â driverVersion )
     Returns the latest CUDA version supported by driver.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDriverGetVersion ( int*Â driverVersion )


Returns the latest CUDA version supported by driver.

######  Parameters

`driverVersion`
    \- Returns the CUDA driver version

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*driverVersion` the version of CUDA supported by the driver. The version is returned as (1000 * major + 10 * minor). For example, CUDA 9.2 would be represented by 9020.

This function automatically returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if `driverVersion` is NULL.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaDriverGetVersion](<../cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g8a06ee14a0551606b7c780084d5564ab>), [cudaRuntimeGetVersion](<../cuda-runtime-api/group__CUDART____VERSION.html#group__CUDART____VERSION_1g0e3952c7802fd730432180f1f4a6cdc6>)

* * *
