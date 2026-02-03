# Module Management

## 6.10.Â Module Management

This section describes the module management functions of the low-level CUDA driver application programming interface.

### Enumerations

enumÂ [CUmoduleLoadingMode](<#group__CUDA__MODULE_1g0e7bf7ad0d578861e15678997f74f789>)


### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLinkAddData](<#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77>) ( CUlinkStateÂ state, [CUjitInputType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc78e5cb421c428676861189048888958>)Â type, void*Â data, size_tÂ size, const char*Â name, unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues )
     Add an input to a pending linker invocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLinkAddFile](<#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c>) ( CUlinkStateÂ state, [CUjitInputType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc78e5cb421c428676861189048888958>)Â type, const char*Â path, unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues )
     Add a file input to a pending linker invocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLinkComplete](<#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716>) ( CUlinkStateÂ state, void**Â cubinOut, size_t*Â sizeOut )
     Complete a pending linker invocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLinkCreate](<#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d>) ( unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues, CUlinkState*Â stateOut )
     Creates a pending JIT linker invocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuLinkDestroy](<#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9>) ( CUlinkStateÂ state )
     Destroys state for a JIT linker invocation.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleEnumerateFunctions](<#group__CUDA__MODULE_1g6bdb22a7d9cacf7df5bda2a18082ec50>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)*Â functions, unsigned int Â numFunctions, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â mod )
     Returns the function handles within a module.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleGetFunction](<#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf>) ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)*Â hfunc, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )
     Returns a function handle.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleGetFunctionCount](<#group__CUDA__MODULE_1gecc8fb61eca765cb0f1eb32f00cf3b49>) ( unsigned int*Â count, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â mod )
     Returns the number of functions within a module.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleGetGlobal](<#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â bytes, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )
     Returns a global pointer from a module.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleGetLoadingMode](<#group__CUDA__MODULE_1g96de378a738ec46d9277c9c9df8f6fd6>) ( [CUmoduleLoadingMode](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g0e7bf7ad0d578861e15678997f74f789>)*Â mode )
     Query lazy loading mode.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleLoad](<#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const char*Â fname )
     Loads a compute module.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleLoadData](<#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const void*Â image )
     Load a module's data.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleLoadDataEx](<#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const void*Â image, unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues )
     Load a module's data with options.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleLoadFatBinary](<#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const void*Â fatCubin )
     Load a module's data.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleUnload](<#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b>) ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod )
     Unloads a module.

### Enumerations

enum CUmoduleLoadingMode


CUDA Lazy Loading status

######  Values

CU_MODULE_EAGER_LOADING = 0x1
    Lazy Kernel Loading is not enabled
CU_MODULE_LAZY_LOADING = 0x2
    Lazy Kernel Loading is enabled

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLinkAddData ( CUlinkStateÂ state, [CUjitInputType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc78e5cb421c428676861189048888958>)Â type, void*Â data, size_tÂ size, const char*Â name, unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues )


Add an input to a pending linker invocation.

######  Parameters

`state`
    A pending linker action.
`type`
    The type of the input data.
`data`
    The input data. PTX must be NULL-terminated.
`size`
    The length of the input data.
`name`
    An optional name for this input in log messages.
`numOptions`
    Size of options.
`options`
    Options to be applied only for this input (overrides options from [cuLinkCreate](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d> "Creates a pending JIT linker invocation.")).
`optionValues`
    Array of option values, each cast to void *.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>)

###### Description

Ownership of `data` is retained by the caller. No reference is retained to any inputs after this call returns.

This method accepts only compiler options, which are used if the data must be compiled from PTX, and does not accept any of [CU_JIT_WALL_TIME](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997df326f19e67f00768bff8849a8dc5def5>), [CU_JIT_INFO_LOG_BUFFER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d0cfba79ec44e0480a78b132ff158c0ae>), [CU_JIT_ERROR_LOG_BUFFER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997da770301a8ec6fdfb9abcbd60100beb3c>), [CU_JIT_TARGET_FROM_CUCONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d8e9b9dfcc1ad8fb0f89bbfb04f46e988>), or [CU_JIT_TARGET](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d3bf12ddeaa2a9d64db4d20de537cd596>).

Note:

For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted

**See also:**

[cuLinkCreate](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d> "Creates a pending JIT linker invocation."), [cuLinkAddFile](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c> "Add a file input to a pending linker invocation."), [cuLinkComplete](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716> "Complete a pending linker invocation."), [cuLinkDestroy](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9> "Destroys state for a JIT linker invocation.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLinkAddFile ( CUlinkStateÂ state, [CUjitInputType](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc78e5cb421c428676861189048888958>)Â type, const char*Â path, unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues )


Add a file input to a pending linker invocation.

######  Parameters

`state`
    A pending linker action
`type`
    The type of the input data
`path`
    Path to the input file
`numOptions`
    Size of options
`options`
    Options to be applied only for this input (overrides options from [cuLinkCreate](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d> "Creates a pending JIT linker invocation."))
`optionValues`
    Array of option values, each cast to void *

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_FILE_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e93c20ccb3e24d5bf65625b1212fd8f4aa>)[CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_IMAGE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90b7bd1dd2fb3491c588ce569c02d1a2f>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>)

###### Description

No reference is retained to any inputs after this call returns.

This method accepts only compiler options, which are used if the input must be compiled from PTX, and does not accept any of [CU_JIT_WALL_TIME](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997df326f19e67f00768bff8849a8dc5def5>), [CU_JIT_INFO_LOG_BUFFER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d0cfba79ec44e0480a78b132ff158c0ae>), [CU_JIT_ERROR_LOG_BUFFER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997da770301a8ec6fdfb9abcbd60100beb3c>), [CU_JIT_TARGET_FROM_CUCONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d8e9b9dfcc1ad8fb0f89bbfb04f46e988>), or [CU_JIT_TARGET](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d3bf12ddeaa2a9d64db4d20de537cd596>).

This method is equivalent to invoking [cuLinkAddData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77> "Add an input to a pending linker invocation.") on the contents of the file.

Note:

For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted

**See also:**

[cuLinkCreate](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d> "Creates a pending JIT linker invocation."), [cuLinkAddData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77> "Add an input to a pending linker invocation."), [cuLinkComplete](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716> "Complete a pending linker invocation."), [cuLinkDestroy](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9> "Destroys state for a JIT linker invocation.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLinkComplete ( CUlinkStateÂ state, void**Â cubinOut, size_t*Â sizeOut )


Complete a pending linker invocation.

######  Parameters

`state`
    A pending linker invocation
`cubinOut`
    On success, this will point to the output image
`sizeOut`
    Optional parameter to receive the size of the generated image

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>)

###### Description

Completes the pending linker action and returns the cubin image for the linked device code, which can be used with [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."). The cubin is owned by `state`, so it should be loaded before `state` is destroyed via [cuLinkDestroy](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9> "Destroys state for a JIT linker invocation."). This call does not destroy `state`.

**See also:**

[cuLinkCreate](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d> "Creates a pending JIT linker invocation."), [cuLinkAddData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77> "Add an input to a pending linker invocation."), [cuLinkAddFile](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c> "Add a file input to a pending linker invocation."), [cuLinkDestroy](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9> "Destroys state for a JIT linker invocation."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLinkCreate ( unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues, CUlinkState*Â stateOut )


Creates a pending JIT linker invocation.

######  Parameters

`numOptions`
    Size of options arrays
`options`
    Array of linker and compiler options
`optionValues`
    Array of option values, each cast to void *
`stateOut`
    On success, this will contain a CUlinkState to specify and complete this action

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>)

###### Description

If the call is successful, the caller owns the returned CUlinkState, which should eventually be destroyed with [cuLinkDestroy](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9> "Destroys state for a JIT linker invocation."). The device code machine size (32 or 64 bit) will match the calling application.

Both linker and compiler options may be specified. Compiler options will be applied to inputs to this linker action which must be compiled from PTX. The options [CU_JIT_WALL_TIME](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997df326f19e67f00768bff8849a8dc5def5>), [CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d8dc284de594cc1504db521869ad47cd9>), and [CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg5527fa8030d5cabedc781a04dbd1997d4e5c6eb78ba970a0b1683ac040044811>) will accumulate data until the CUlinkState is destroyed.

The data passed in via [cuLinkAddData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77> "Add an input to a pending linker invocation.") and [cuLinkAddFile](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c> "Add a file input to a pending linker invocation.") will be treated as relocatable (-rdc=true to nvcc) when linking the final cubin during [cuLinkComplete](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716> "Complete a pending linker invocation.") and will have similar consequences as offline relocatable device code linking.

`optionValues` must remain valid for the life of the CUlinkState if output options are used. No other references to inputs are maintained after this call returns.

Note:

For LTO-IR input, only LTO-IR compiled with toolkits prior to CUDA 12.0 will be accepted

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuLinkAddData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77> "Add an input to a pending linker invocation."), [cuLinkAddFile](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c> "Add a file input to a pending linker invocation."), [cuLinkComplete](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716> "Complete a pending linker invocation."), [cuLinkDestroy](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9> "Destroys state for a JIT linker invocation.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuLinkDestroy ( CUlinkStateÂ state )


Destroys state for a JIT linker invocation.

######  Parameters

`state`
    State object for the linker invocation

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>)

###### Description

**See also:**

[cuLinkCreate](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d> "Creates a pending JIT linker invocation.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleEnumerateFunctions ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)*Â functions, unsigned int Â numFunctions, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â mod )


Returns the function handles within a module.

######  Parameters

`functions`
    \- Buffer where the function handles are returned to
`numFunctions`
    \- Maximum number of function handles may be returned to the buffer
`mod`
    \- Module to query from

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `functions` a maximum number of `numFunctions` function handles within `mod`. When function loading mode is set to LAZY the function retrieved may be partially loaded. The loading state of a function can be queried using cuFunctionIsLoaded. CUDA APIs may load the function automatically when called with partially loaded function handle which may incur additional latency. Alternatively, cuFunctionLoad can be used to explicitly load a function. The returned function handles become invalid when the module is unloaded.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetFunctionCount](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gecc8fb61eca765cb0f1eb32f00cf3b49> "Returns the number of functions within a module."), [cuFuncIsLoaded](<group__CUDA__EXEC.html#group__CUDA__EXEC_1gfc6fed4bbe6c35e0445a49396774aa96> "Returns if the function is loaded."), [cuFuncLoad](<group__CUDA__EXEC.html#group__CUDA__EXEC_1g3b67024e8875bfd155534785708093ab> "Loads a function.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleGetFunction ( [CUfunction](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gba6128b948022f495706d93bc2cea9c8>)*Â hfunc, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )


Returns a function handle.

######  Parameters

`hfunc`
    \- Returned function handle
`hmod`
    \- Module to retrieve function from
`name`
    \- Name of function to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `*hfunc` the handle of the function of name `name` located in module `hmod`. If no function of that name exists, [cuModuleGetFunction()](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle.") returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleGetFunctionCount ( unsigned int*Â count, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â mod )


Returns the number of functions within a module.

######  Parameters

`count`
    \- Number of functions found within the module
`mod`
    \- Module to query

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_HANDLE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e90fd2e63260c6317ba943af0f7e4b8d21>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `count` the number of functions in `mod`.

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleGetGlobal ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â dptr, size_t*Â bytes, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )


Returns a global pointer from a module.

######  Parameters

`dptr`
    \- Returned global device pointer
`bytes`
    \- Returned global size in bytes
`hmod`
    \- Module to retrieve global from
`name`
    \- Name of global to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### Description

Returns in `*dptr` and `*bytes` the base pointer and size of the global of name `name` located in module `hmod`. If no variable of that name exists, [cuModuleGetGlobal()](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module.") returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>). One of the parameters `dptr` or `bytes` (not both) can be NULL in which case it is ignored.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module."), [cudaGetSymbolAddress](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g3a0f010e70a3343db18227cec9615177>), [cudaGetSymbolSize](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g0561c8ffee270bff0bbb7deb81ad865c>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleGetLoadingMode ( [CUmoduleLoadingMode](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g0e7bf7ad0d578861e15678997f74f789>)*Â mode )


Query lazy loading mode.

######  Parameters

`mode`
    \- Returns the lazy loading mode

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>),

###### Description

Returns lazy loading mode Module loading mode is controlled by CUDA_MODULE_LOADING env variable

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."),

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleLoad ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const char*Â fname )


Loads a compute module.

######  Parameters

`module`
    \- Returned module
`fname`
    \- Filename of module to load

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_FILE_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e93c20ccb3e24d5bf65625b1212fd8f4aa>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>), [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95ab6c0086a6130b5b895ff15ce841ee6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>)

###### Description

Takes a filename `fname` and loads the corresponding module `module` into the current context. The CUDA driver API does not attempt to lazily allocate the resources needed by a module; if the memory for functions and data (constant and global) needed by the module cannot be allocated, [cuModuleLoad()](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module.") fails. The file should be a cubin file as output by **nvcc** , or a PTX file either as output by **nvcc** or handwritten, or a fatbin file as output by **nvcc** from toolchain 4.0 or later, or a Tile IR file.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleLoadData ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const void*Â image )


Load a module's data.

######  Parameters

`module`
    \- Returned module
`image`
    \- Module data to load

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>), [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95ab6c0086a6130b5b895ff15ce841ee6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>)

###### Description

Takes a pointer `image` and loads the corresponding module `module` into the current context. The `image` may be a cubin or fatbin as output by **nvcc** , or a NULL-terminated PTX, either as output by **nvcc** or hand-written, or Tile IR data.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleLoadDataEx ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const void*Â image, unsigned int Â numOptions, [CUjit_option](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g5527fa8030d5cabedc781a04dbd1997d>)*Â options, void**Â optionValues )


Load a module's data with options.

######  Parameters

`module`
    \- Returned module
`image`
    \- Module data to load
`numOptions`
    \- Number of options
`options`
    \- Options for JIT
`optionValues`
    \- Option values for JIT

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>), [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95ab6c0086a6130b5b895ff15ce841ee6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>)

###### Description

Takes a pointer `image` and loads the corresponding module `module` into the current context. The `image` may be a cubin or fatbin as output by **nvcc** , or a NULL-terminated PTX, either as output by **nvcc** or hand-written, or Tile IR data.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleLoadFatBinary ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)*Â module, const void*Â fatCubin )


Load a module's data.

######  Parameters

`module`
    \- Returned module
`fatCubin`
    \- Fat binary to load

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_INVALID_PTX](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a5bb7f216af3efbea2116ff18253b1a3>), [CUDA_ERROR_UNSUPPORTED_PTX_VERSION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98381092e26bfe660cef4a755bb549610>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_NO_BINARY_FOR_GPU](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e94ef545ed5f461db9351f98de98497abf>), [CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95ab6c0086a6130b5b895ff15ce841ee6>), [CUDA_ERROR_SHARED_OBJECT_INIT_FAILED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d8a149ebc98aa90f6417e531fa645043>), [CUDA_ERROR_JIT_COMPILER_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e91e1b93d0f27e74d6a9e9e16f410542c6>)

###### Description

Takes a pointer `fatCubin` and loads the corresponding module `module` into the current context. The pointer represents a fat binary object, which is a collection of different cubin and/or PTX files, all representing the same device code, but compiled and optimized for different architectures.

Prior to CUDA 4.0, there was no documented API for constructing and using fat binary objects by programmers. Starting with CUDA 4.0, fat binary objects can be constructed by providing the -fatbin option to **nvcc**. More information can be found in the **nvcc** document.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleUnload ( [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod )


Unloads a module.

######  Parameters

`hmod`
    \- Module to unload

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>)

###### Description

Unloads a module `hmod` from the current context. Attempting to unload a module which was obtained from the Library Management API such as [cuLibraryGetModule](<group__CUDA__LIBRARY.html#group__CUDA__LIBRARY_1g0d439597c77b64cf247de33f0609a5d8> "Returns a module handle.") will return [CUDA_ERROR_NOT_PERMITTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9e23d0197c490ec332a43e55b167968a3>).

Note:

  * Note that this function may also return error codes from previous, asynchronous launches.

  * Use of the handle after this call is undefined behavior.


**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data.")

* * *
