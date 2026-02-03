# Error Handling

## 6.2.Â Error Handling

This section describes the error handling functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGetErrorName](<#group__CUDA__ERROR_1g2c4ac087113652bb3d1f95bf2513c468>) ( [CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â error, const char**Â pStr )
     Gets the string representation of an error code enum name.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuGetErrorString](<#group__CUDA__ERROR_1g72758fcaf05b5c7fac5c25ead9445ada>) ( [CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â error, const char**Â pStr )
     Gets the string description of an error code.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGetErrorName ( [CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â error, const char**Â pStr )


Gets the string representation of an error code enum name.

######  Parameters

`error`
    \- Error code to convert to string
`pStr`
    \- Address of the string pointer.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets `*pStr` to the address of a NULL-terminated string representation of the name of the enum error code `error`. If the error code is not recognized, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned and `*pStr` will be set to the NULL address.

**See also:**

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>), [cudaGetErrorName](<../cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1gb3de7da2f23736878270026dcfc70075>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuGetErrorString ( [CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â error, const char**Â pStr )


Gets the string description of an error code.

######  Parameters

`error`
    \- Error code to convert to string
`pStr`
    \- Address of the string pointer.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Sets `*pStr` to the address of a NULL-terminated string description of the error code `error`. If the error code is not recognized, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) will be returned and `*pStr` will be set to the NULL address.

**See also:**

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>), [cudaGetErrorString](<../cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g4bc9e35a618dfd0877c29c8ee45148f1>)

* * *
