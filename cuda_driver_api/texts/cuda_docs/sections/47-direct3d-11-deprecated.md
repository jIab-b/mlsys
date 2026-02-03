# Direct3D 11 (Deprecated)

## 6.43.1.Â Direct3D 11 Interoperability [DEPRECATED]

## [[Direct3D 11 Interoperability](<group__CUDA__D3D11.html#group__CUDA__D3D11>)]

This section describes deprecated Direct3D 11 interoperability functionality.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D11CtxCreate](<#group__CUDA__D3D11__DEPRECATED_1gc81ef881bbd18ff5527f37e8a76fd761>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, unsigned int Â Flags, ID3D11Device*Â pD3DDevice )
     Create a CUDA context for interoperability with Direct3D 11.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D11CtxCreateOnDevice](<#group__CUDA__D3D11__DEPRECATED_1g9f5d54aa09837416a552b16d52479a02>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, ID3D11Device*Â pD3DDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â cudaDevice )
     Create a CUDA context for interoperability with Direct3D 11.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuD3D11GetDirect3DDevice](<#group__CUDA__D3D11__DEPRECATED_1g0b929512b51e56cfef54f2616bd33ed8>) ( ID3D11Device**Â ppD3DDevice )
     Get the Direct3D 11 device against which the current CUDA context was created.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D11CtxCreate ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)*Â pCudaDevice, unsigned int Â Flags, ID3D11Device*Â pD3DDevice )


Create a CUDA context for interoperability with Direct3D 11.

######  Parameters

`pCtx`
    \- Returned newly created CUDA context
`pCudaDevice`
    \- Returned pointer to the device on which the context was created
`Flags`
    \- Context creation flags (see [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") for details)
`pD3DDevice`
    \- Direct3D device to create interoperability context with

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000103>)

This function is deprecated as of CUDA 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with a D3D11 device in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D11GetDevice](<group__CUDA__D3D11.html#group__CUDA__D3D11_1ga1f1648cdf3bd5aef7a55af6dc1f42cd> "Gets the CUDA device corresponding to a display adapter."), [cuGraphicsD3D11RegisterResource](<group__CUDA__D3D11.html#group__CUDA__D3D11_1g4c02792aa87c3acc255b9de15b0509da> "Register a Direct3D 11 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D11CtxCreateOnDevice ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)*Â pCtx, unsigned int Â flags, ID3D11Device*Â pD3DDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â cudaDevice )


Create a CUDA context for interoperability with Direct3D 11.

######  Parameters

`pCtx`
    \- Returned newly created CUDA context
`flags`
    \- Context creation flags (see [cuCtxCreate()](<group__CUDA__CTX.html#group__CUDA__CTX_1g77e9fb578caefca5ed15b4acebf35265> "Create a CUDA context.") for details)
`pD3DDevice`
    \- Direct3D device to create interoperability context with
`cudaDevice`
    \- The CUDA device on which to create the context. This device must be among the devices returned when querying CU_D3D11_DEVICES_ALL from [cuD3D11GetDevices](<group__CUDA__D3D11.html#group__CUDA__D3D11_1g7fca109b0dba2050b58f6bac627ff441> "Gets the CUDA devices corresponding to a Direct3D 11 device.").

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>), [CUDA_ERROR_OUT_OF_MEMORY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9264c50688ed110e8476b591befe60c02>), [CUDA_ERROR_UNKNOWN](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9c5a6ab0245179d297f1fa56ed0097183>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000104>)

This function is deprecated as of CUDA 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with a D3D11 device in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D11GetDevices](<group__CUDA__D3D11.html#group__CUDA__D3D11_1g7fca109b0dba2050b58f6bac627ff441> "Gets the CUDA devices corresponding to a Direct3D 11 device."), [cuGraphicsD3D11RegisterResource](<group__CUDA__D3D11.html#group__CUDA__D3D11_1g4c02792aa87c3acc255b9de15b0509da> "Register a Direct3D 11 resource for access by CUDA.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuD3D11GetDirect3DDevice ( ID3D11Device**Â ppD3DDevice )


Get the Direct3D 11 device against which the current CUDA context was created.

######  Parameters

`ppD3DDevice`
    \- Returned Direct3D device corresponding to CUDA context

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000105>)

This function is deprecated as of CUDA 5.0.

###### Description

This function is deprecated and should no longer be used. It is no longer necessary to associate a CUDA context with a D3D11 device in order to achieve maximum interoperability performance.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuD3D11GetDevice](<group__CUDA__D3D11.html#group__CUDA__D3D11_1ga1f1648cdf3bd5aef7a55af6dc1f42cd> "Gets the CUDA device corresponding to a display adapter.")

* * *
