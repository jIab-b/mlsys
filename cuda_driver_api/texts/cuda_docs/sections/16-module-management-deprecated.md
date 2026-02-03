# Module Management (Deprecated)

## 6.11.Â Module Management [DEPRECATED]

This section describes the deprecated module management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleGetSurfRef](<#group__CUDA__MODULE__DEPRECATED_1g3c9cccfdfa65d6cf492b7ce1b93a4596>) ( [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>)*Â pSurfRef, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )
     Returns a handle to a surface reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuModuleGetTexRef](<#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)*Â pTexRef, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )
     Returns a handle to a texture reference.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleGetSurfRef ( [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>)*Â pSurfRef, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )


Returns a handle to a surface reference.

######  Parameters

`pSurfRef`
    \- Returned surface reference
`hmod`
    \- Module to retrieve surface reference from
`name`
    \- Name of surface reference to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000009>)

###### Description

Returns in `*pSurfRef` the handle of the surface reference of name `name` in the module `hmod`. If no surface reference of that name exists, [cuModuleGetSurfRef()](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g3c9cccfdfa65d6cf492b7ce1b93a4596> "Returns a handle to a surface reference.") returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>).

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetTexRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuModuleGetTexRef ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)*Â pTexRef, [CUmodule](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9e4ef4dcfba4662b2299acb8d049a1ef>)Â hmod, const char*Â name )


Returns a handle to a texture reference.

######  Parameters

`pTexRef`
    \- Returned texture reference
`hmod`
    \- Module to retrieve texture reference from
`name`
    \- Name of texture reference to retrieve

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000008>)

###### Description

Returns in `*pTexRef` the handle of the texture reference of name `name` in the module `hmod`. If no texture reference of that name exists, [cuModuleGetTexRef()](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g9965d238143354d573ef5789057be561> "Returns a handle to a texture reference.") returns [CUDA_ERROR_NOT_FOUND](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9af1cadbeb21d3a78115ca211ba44c053>). This texture reference handle should not be destroyed, since it will be destroyed when the module is unloaded.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuModuleGetFunction](<group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf> "Returns a function handle."), [cuModuleGetGlobal](<group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab> "Returns a global pointer from a module."), [cuModuleGetSurfRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g3c9cccfdfa65d6cf492b7ce1b93a4596> "Returns a handle to a surface reference."), [cuModuleLoad](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3> "Loads a compute module."), [cuModuleLoadData](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b> "Load a module's data."), [cuModuleLoadDataEx](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12> "Load a module's data with options."), [cuModuleLoadFatBinary](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g13a2292b6819f8f86127768334436c3b> "Load a module's data."), [cuModuleUnload](<group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b> "Unloads a module.")

* * *
