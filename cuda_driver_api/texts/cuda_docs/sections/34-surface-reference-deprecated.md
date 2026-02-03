# Surface Reference (Deprecated)

## 6.27.Â Surface Reference Management [DEPRECATED]

This section describes the surface reference management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuSurfRefGetArray](<#group__CUDA__SURFREF__DEPRECATED_1g9e46d47dce3ff21a0c6485c8613e391c>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â phArray, [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>)Â hSurfRef )
     Passes back the CUDA array bound to a surface reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuSurfRefSetArray](<#group__CUDA__SURFREF__DEPRECATED_1g68abcde159fa897b1dfb23387926dd66>) ( [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>)Â hSurfRef, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray, unsigned int Â Flags )
     Sets the CUDA array for a surface reference.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuSurfRefGetArray ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â phArray, [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>)Â hSurfRef )


Passes back the CUDA array bound to a surface reference.

######  Parameters

`phArray`
    \- Surface reference handle
`hSurfRef`
    \- Surface reference handle

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000050>)

###### Description

Returns in `*phArray` the CUDA array bound to the surface reference `hSurfRef`, or returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if the surface reference is not bound to any CUDA array.

**See also:**

[cuModuleGetSurfRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g3c9cccfdfa65d6cf492b7ce1b93a4596> "Returns a handle to a surface reference."), [cuSurfRefSetArray](<group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED_1g68abcde159fa897b1dfb23387926dd66> "Sets the CUDA array for a surface reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuSurfRefSetArray ( [CUsurfref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7b99472b414f10b2c04dd2530dc7ea76>)Â hSurfRef, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray, unsigned int Â Flags )


Sets the CUDA array for a surface reference.

######  Parameters

`hSurfRef`
    \- Surface reference handle
`hArray`
    \- CUDA array handle
`Flags`
    \- set to 0

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000049>)

###### Description

Sets the CUDA array `hArray` to be read and written by the surface reference `hSurfRef`. Any previous CUDA array state associated with the surface reference is superseded by this function. `Flags` must be set to 0. The [CUDA_ARRAY3D_SURFACE_LDST](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7287c43cacf1ed05865d6bcad1a23cd9>) flag must have been set for the CUDA array. Any CUDA array previously bound to `hSurfRef` is unbound.

**See also:**

[cuModuleGetSurfRef](<group__CUDA__MODULE__DEPRECATED.html#group__CUDA__MODULE__DEPRECATED_1g3c9cccfdfa65d6cf492b7ce1b93a4596> "Returns a handle to a surface reference."), [cuSurfRefGetArray](<group__CUDA__SURFREF__DEPRECATED.html#group__CUDA__SURFREF__DEPRECATED_1g9e46d47dce3ff21a0c6485c8613e391c> "Passes back the CUDA array bound to a surface reference.")

* * *
