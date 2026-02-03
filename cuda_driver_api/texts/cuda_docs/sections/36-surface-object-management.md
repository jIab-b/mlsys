# Surface Object Management

## 6.29.Â Surface Object Management

This section describes the surface object management functions of the low-level CUDA driver application programming interface. The surface object API is only supported on devices of compute capability 3.0 or higher.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuSurfObjectCreate](<#group__CUDA__SURFOBJECT_1g6bc972c90c9590c9f720b2754e6d079d>) ( [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>)*Â pSurfObject, const [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc )
     Creates a surface object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuSurfObjectDestroy](<#group__CUDA__SURFOBJECT_1g4c4ec48d203d1e0bb71750ddc4d7aef3>) ( [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>)Â surfObject )
     Destroys a surface object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuSurfObjectGetResourceDesc](<#group__CUDA__SURFOBJECT_1g2472b7ea0b7e74600ed3d6c244b7ba21>) ( [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc, [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>)Â surfObject )
     Returns a surface object's resource descriptor.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuSurfObjectCreate ( [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>)*Â pSurfObject, const [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc )


Creates a surface object.

######  Parameters

`pSurfObject`
    \- Surface object to create
`pResDesc`
    \- Resource descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a surface object and returns it in `pSurfObject`. `pResDesc` describes the data to perform surface load/stores on. [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) must be [CU_RESOURCE_TYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f68171f299e8447a926051e13d613d77b1>) and CUDA_RESOURCE_DESC::res::array::hArray must be set to a valid CUDA array handle. [CUDA_RESOURCE_DESC::flags](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1857d9251dec54700bd8cd071accc3bdf>) must be set to zero.

Surface objects are only supported on devices of compute capability 3.0 or higher. Additionally, a surface object is an opaque value, and, as such, should only be accessed through CUDA API calls.

**See also:**

[cuSurfObjectDestroy](<group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT_1g4c4ec48d203d1e0bb71750ddc4d7aef3> "Destroys a surface object."), [cudaCreateSurfaceObject](<../cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1g958899474ab2c5f40d233b524d6c5a01>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuSurfObjectDestroy ( [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>)Â surfObject )


Destroys a surface object.

######  Parameters

`surfObject`
    \- Surface object to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Destroys the surface object specified by `surfObject`.

**See also:**

[cuSurfObjectCreate](<group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT_1g6bc972c90c9590c9f720b2754e6d079d> "Creates a surface object."), [cudaDestroySurfaceObject](<../cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1g9fab66c3a39b9f8f52b718eea794ad60>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuSurfObjectGetResourceDesc ( [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc, [CUsurfObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4acc685a8412637d05668e30e984e220>)Â surfObject )


Returns a surface object's resource descriptor.

######  Parameters

`pResDesc`
    \- Resource descriptor
`surfObject`
    \- Surface object

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the resource descriptor for the surface object specified by `surfObject`.

**See also:**

[cuSurfObjectCreate](<group__CUDA__SURFOBJECT.html#group__CUDA__SURFOBJECT_1g6bc972c90c9590c9f720b2754e6d079d> "Creates a surface object."), [cudaGetSurfaceObjectResourceDesc](<../cuda-runtime-api/group__CUDART__SURFACE__OBJECT.html#group__CUDART__SURFACE__OBJECT_1gd7087318f73ae605645d6721d51486bd>)

* * *
