# Occupancy

## 6.25.Â Occupancy

This section describes the occupancy calculation functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyAvailableDynamicSMemPerBlock](<#group__CUDA__OCCUPANCY_1gae02af6a9df9e1bbd51941af631bce69>) ( size_t*Â dynamicSmemSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, int Â numBlocks, int Â blockSize )
     Returns dynamic shared memory available per block when launching `numBlocks` blocks on SM.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyMaxActiveBlocksPerMultiprocessor](<#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98>) ( int*Â numBlocks, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, int Â blockSize, size_tÂ dynamicSMemSize )
     Returns occupancy of a function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags](<#group__CUDA__OCCUPANCY_1g8f1da4d4983e5c3025447665423ae2c2>) ( int*Â numBlocks, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, int Â blockSize, size_tÂ dynamicSMemSize, unsigned int Â flags )
     Returns occupancy of a function.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyMaxActiveClusters](<#group__CUDA__OCCUPANCY_1g4f52cbf144d74ed20351a594dc26386b>) ( int*Â numClusters, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, const [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>)*Â config )
     Given the kernel function (`func`) and launch configuration (`config`), return the maximum number of clusters that could co-exist on the target device in `*numClusters`.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyMaxPotentialBlockSize](<#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03>) ( int*Â minGridSize, int*Â blockSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, [CUoccupancyB2DSize](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6cb31f1273726f5567051e3e21607a45>)Â blockSizeToDynamicSMemSize, size_tÂ dynamicSMemSize, int Â blockSizeLimit )
     Suggest a launch configuration with reasonable occupancy.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyMaxPotentialBlockSizeWithFlags](<#group__CUDA__OCCUPANCY_1g04c0bb65630f82d9b99a5ca0203ee5aa>) ( int*Â minGridSize, int*Â blockSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, [CUoccupancyB2DSize](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6cb31f1273726f5567051e3e21607a45>)Â blockSizeToDynamicSMemSize, size_tÂ dynamicSMemSize, int Â blockSizeLimit, unsigned int Â flags )
     Suggest a launch configuration with reasonable occupancy.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuOccupancyMaxPotentialClusterSize](<#group__CUDA__OCCUPANCY_1gd6f60814c1e3440145115ade3730365f>) ( int*Â clusterSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, const [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>)*Â config )
     Given the kernel function (`func`) and launch configuration (`config`), return the maximum cluster size in `*clusterSize`.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyAvailableDynamicSMemPerBlock ( size_t*Â dynamicSmemSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, int Â numBlocks, int Â blockSize )


Returns dynamic shared memory available per block when launching `numBlocks` blocks on SM.

######  Parameters

`dynamicSmemSize`
    \- Returned maximum dynamic shared memory
`func`
    \- Kernel function for which occupancy is calculated
`numBlocks`
    \- Number of blocks to fit on SM
`blockSize`
    \- Size of the blocks

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*dynamicSmemSize` the maximum size of dynamic shared memory to allow `numBlocks` blocks per SM.

Note that the API can also be used with context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will be the current context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyMaxActiveBlocksPerMultiprocessor ( int*Â numBlocks, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, int Â blockSize, size_tÂ dynamicSMemSize )


Returns occupancy of a function.

######  Parameters

`numBlocks`
    \- Returned occupancy
`func`
    \- Kernel for which occupancy is calculated
`blockSize`
    \- Block size the kernel is intended to be launched with
`dynamicSMemSize`
    \- Per-block dynamic shared memory usage intended, in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*numBlocks` the number of the maximum active blocks per streaming multiprocessor.

Note that the API can also be used with context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will be the current context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaOccupancyMaxActiveBlocksPerMultiprocessor](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g5a5d67a3c907371559ba692195e8a38c>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int*Â numBlocks, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, int Â blockSize, size_tÂ dynamicSMemSize, unsigned int Â flags )


Returns occupancy of a function.

######  Parameters

`numBlocks`
    \- Returned occupancy
`func`
    \- Kernel for which occupancy is calculated
`blockSize`
    \- Block size the kernel is intended to be launched with
`dynamicSMemSize`
    \- Per-block dynamic shared memory usage intended, in bytes
`flags`
    \- Requested behavior for the occupancy calculator

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*numBlocks` the number of the maximum active blocks per streaming multiprocessor.

The `Flags` parameter controls how special cases are handled. The valid flags are:

  * [CU_OCCUPANCY_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg40caa223198d058e073116b6a55eb895996ff3316265d9adf180fc54aa6c4b85>), which maintains the default behavior as [cuOccupancyMaxActiveBlocksPerMultiprocessor](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gcc6e1094d05cba2cee17fe33ddd04a98> "Returns occupancy of a function.");


  * [CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg40caa223198d058e073116b6a55eb8955f3f3738f84d2fd569e4b574350d09bb>), which suppresses the default behavior on platform where global caching affects occupancy. On such platforms, if caching is enabled, but per-block SM resource usage would result in zero occupancy, the occupancy calculator will calculate the occupancy as if caching is disabled. Setting [CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg40caa223198d058e073116b6a55eb8955f3f3738f84d2fd569e4b574350d09bb>) makes the occupancy calculator to return 0 in such cases. More information can be found about this feature in the "Unified L1/Texture Cache" section of the Maxwell tuning guide.


Note that the API can also be with launch context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will be the current context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g603b86b20b37823253ff89fe8688ba83>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyMaxActiveClusters ( int*Â numClusters, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, const [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>)*Â config )


Given the kernel function (`func`) and launch configuration (`config`), return the maximum number of clusters that could co-exist on the target device in `*numClusters`.

######  Parameters

`numClusters`
    \- Returned maximum number of clusters that could co-exist on the target device
`func`
    \- Kernel function for which maximum number of clusters are calculated
`config`
    \- Launch configuration for the given kernel function

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_CLUSTER_SIZE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e927b783c2a2b1e62e8a9edc9044c70e66>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

If the function has required cluster size already set (see [cudaFuncGetAttributes](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0e78e02c6d12ebddd4577ac6ebadf494>) / [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function.")), the cluster size from config must either be unspecified or match the required size. Without required sizes, the cluster size must be specified in config, else the function will return an error.

Note that various attributes of the kernel function may affect occupancy calculation. Runtime environment may affect how the hardware schedules the clusters, so the calculated occupancy is not guaranteed to be achievable.

Note that the API can also be used with context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will either be taken from the specified stream `config->hStream` or the current context in case of NULL stream.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaFuncGetAttributes](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0e78e02c6d12ebddd4577ac6ebadf494>), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyMaxPotentialBlockSize ( int*Â minGridSize, int*Â blockSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, [CUoccupancyB2DSize](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6cb31f1273726f5567051e3e21607a45>)Â blockSizeToDynamicSMemSize, size_tÂ dynamicSMemSize, int Â blockSizeLimit )


Suggest a launch configuration with reasonable occupancy.

######  Parameters

`minGridSize`
    \- Returned minimum grid size needed to achieve the maximum occupancy
`blockSize`
    \- Returned maximum block size that can achieve the maximum occupancy
`func`
    \- Kernel for which launch configuration is calculated
`blockSizeToDynamicSMemSize`
    \- A function that calculates how much per-block dynamic shared memory `func` uses based on the block size
`dynamicSMemSize`
    \- Dynamic shared memory usage intended, in bytes
`blockSizeLimit`
    \- The maximum block size `func` is designed to handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

Returns in `*blockSize` a reasonable block size that can achieve the maximum occupancy (or, the maximum number of active warps with the fewest blocks per multiprocessor), and in `*minGridSize` the minimum grid size to achieve the maximum occupancy.

If `blockSizeLimit` is 0, the configurator will use the maximum block size permitted by the device / function instead.

If per-block dynamic shared memory allocation is not needed, the user should leave both `blockSizeToDynamicSMemSize` and `dynamicSMemSize` as 0.

If per-block dynamic shared memory allocation is needed, then if the dynamic shared memory size is constant regardless of block size, the size should be passed through `dynamicSMemSize`, and `blockSizeToDynamicSMemSize` should be NULL.

Otherwise, if the per-block dynamic shared memory size varies with different block sizes, the user needs to provide a unary function through `blockSizeToDynamicSMemSize` that computes the dynamic shared memory needed by `func` for any given block size. `dynamicSMemSize` is ignored. An example signature is:


    â    // Take block size, returns dynamic shared memory needed
              size_t blockToSmem(int blockSize);

Note that the API can also be used with context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will be the current context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaOccupancyMaxPotentialBlockSize](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gee5334618ed4bb0871e4559a77643fc1>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyMaxPotentialBlockSizeWithFlags ( int*Â minGridSize, int*Â blockSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, [CUoccupancyB2DSize](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6cb31f1273726f5567051e3e21607a45>)Â blockSizeToDynamicSMemSize, size_tÂ dynamicSMemSize, int Â blockSizeLimit, unsigned int Â flags )


Suggest a launch configuration with reasonable occupancy.

######  Parameters

`minGridSize`
    \- Returned minimum grid size needed to achieve the maximum occupancy
`blockSize`
    \- Returned maximum block size that can achieve the maximum occupancy
`func`
    \- Kernel for which launch configuration is calculated
`blockSizeToDynamicSMemSize`
    \- A function that calculates how much per-block dynamic shared memory `func` uses based on the block size
`dynamicSMemSize`
    \- Dynamic shared memory usage intended, in bytes
`blockSizeLimit`
    \- The maximum block size `func` is designed to handle
`flags`
    \- Options

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

An extended version of [cuOccupancyMaxPotentialBlockSize](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03> "Suggest a launch configuration with reasonable occupancy."). In addition to arguments passed to [cuOccupancyMaxPotentialBlockSize](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03> "Suggest a launch configuration with reasonable occupancy."), [cuOccupancyMaxPotentialBlockSizeWithFlags](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1g04c0bb65630f82d9b99a5ca0203ee5aa> "Suggest a launch configuration with reasonable occupancy.") also takes a `Flags` parameter.

The `Flags` parameter controls how special cases are handled. The valid flags are:

  * [CU_OCCUPANCY_DEFAULT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg40caa223198d058e073116b6a55eb895996ff3316265d9adf180fc54aa6c4b85>), which maintains the default behavior as [cuOccupancyMaxPotentialBlockSize](<group__CUDA__OCCUPANCY.html#group__CUDA__OCCUPANCY_1gf179c4ab78962a8468e41c3f57851f03> "Suggest a launch configuration with reasonable occupancy.");


  * [CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg40caa223198d058e073116b6a55eb8955f3f3738f84d2fd569e4b574350d09bb>), which suppresses the default behavior on platform where global caching affects occupancy. On such platforms, the launch configurations that produces maximal occupancy might not support global caching. Setting [CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg40caa223198d058e073116b6a55eb8955f3f3738f84d2fd569e4b574350d09bb>) guarantees that the the produced launch configuration is global caching compatible at a potential cost of occupancy. More information can be found about this feature in the "Unified L1/Texture Cache" section of the Maxwell tuning guide.


Note that the API can also be used with context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will be the current context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaOccupancyMaxPotentialBlockSizeWithFlags](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gd0524825c5c01bbc9a5e29e890745800>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuOccupancyMaxPotentialClusterSize ( int*Â clusterSize, [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)Â func, const [CUlaunchConfig](<structCUlaunchConfig.html#structCUlaunchConfig>)*Â config )


Given the kernel function (`func`) and launch configuration (`config`), return the maximum cluster size in `*clusterSize`.

######  Parameters

`clusterSize`
    \- Returned maximum cluster size that can be launched for the given kernel function and launch configuration
`func`
    \- Kernel function for which maximum cluster size is calculated
`config`
    \- Launch configuration for the given kernel function

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### Description

The cluster dimensions in `config` are ignored. If func has a required cluster size set (see [cudaFuncGetAttributes](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0e78e02c6d12ebddd4577ac6ebadf494>) / [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function.")),`*clusterSize` will reflect the required cluster size.

By default this function will always return a value that's portable on future hardware. A higher value may be returned if the kernel function allows non-portable cluster sizes.

This function will respect the compile time launch bounds.

Note that the API can also be used with context-less kernel [CUkernel](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g612028921e5736db673e4307589989ed>) by querying the handle using [cuLibraryGetKernel()](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g15336d865f5abd63e3dc6004d5bc037a> "Returns a kernel handle.") and then passing it to the API by casting to [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>). Here, the context to use for calculations will either be taken from the specified stream `config->hStream` or the current context in case of NULL stream.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cudaFuncGetAttributes](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0e78e02c6d12ebddd4577ac6ebadf494>), [cuFuncGetAttribute](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b> "Returns information about a function.")

* * *
