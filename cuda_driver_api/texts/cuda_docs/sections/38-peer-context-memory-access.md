# Peer Context Memory Access

## 6.31.Â Peer Context Memory Access

This section describes the direct peer context memory access functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxDisablePeerAccess](<#group__CUDA__PEER__ACCESS_1g5b4b6936ea868d4954ce4d841a3b4810>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â peerContext )
     Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuCtxEnablePeerAccess](<#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a>) ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â peerContext, unsigned int Â Flags )
     Enables direct access to memory allocations in a peer context.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceCanAccessPeer](<#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e>) ( int*Â canAccessPeer, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â peerDev )
     Queries if a device may directly access a peer device's memory.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetP2PAtomicCapabilities](<#group__CUDA__PEER__ACCESS_1gfd989876c8fd3291b520c0b561d5282d>) ( unsigned int*Â capabilities, const [CUatomicOperation *](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfd7012de0abfe50cee089f3f00d6dcf3>)*Â operations, unsigned int Â count, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â srcDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dstDevice )
     Queries details about atomic operations supported between two devices.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuDeviceGetP2PAttribute](<#group__CUDA__PEER__ACCESS_1g4c55c60508f8eba4546b51f2ee545393>) ( int*Â value, [CUdevice_P2PAttribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g578d7cf687ce20f7e99468e8c14e22de>)Â attrib, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â srcDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dstDevice )
     Queries attributes of the link between two devices.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxDisablePeerAccess ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â peerContext )


Disables direct access to memory allocations in a peer context and unregisters any registered allocations.

######  Parameters

`peerContext`
    \- Peer context to disable direct access to

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_PEER_ACCESS_NOT_ENABLED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95e985a1735204ae8455e9eec402d46c3>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>),

###### Description

Returns [CUDA_ERROR_PEER_ACCESS_NOT_ENABLED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e95e985a1735204ae8455e9eec402d46c3>) if direct peer access has not yet been enabled from `peerContext` to the current context.

Returns [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) if there is no current context, or if `peerContext` is not a valid context.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceCanAccessPeer](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e> "Queries if a device may directly access a peer device's memory."), [cuCtxEnablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context."), [cudaDeviceDisablePeerAccess](<../cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g9663734ad02653207ad6836053bf572e>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuCtxEnablePeerAccess ( [CUcontext](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f5bd81658f866613785b3a0bb7d7d9>)Â peerContext, unsigned int Â Flags )


Enables direct access to memory allocations in a peer context.

######  Parameters

`peerContext`
    \- Peer context to enable direct access to from the current context
`Flags`
    \- Reserved for future use and must be set to 0

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e947c18606be796573aa2957402fa89a9c>), [CUDA_ERROR_TOO_MANY_PEERS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9168ef870793a31ef4cdd7cb6e279b34a>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_PEER_ACCESS_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d60abcaa3f2710f961db8c383bb95cae>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

If both the current context and `peerContext` are on devices which support unified addressing (as may be queried using [CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3dc11dd6d9f149a7bae32499f2b802c0d>)) and same major compute capability, then on success all allocations from `peerContext` will immediately be accessible by the current context. See [Unified Addressing](<group__CUDA__UNIFIED.html#group__CUDA__UNIFIED>) for additional details.

Note that access granted by this call is unidirectional and that in order to access memory from the current context in `peerContext`, a separate symmetric call to [cuCtxEnablePeerAccess()](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context.") is required.

Note that there are both device-wide and system-wide limitations per system configuration, as noted in the CUDA Programming Guide under the section "Peer-to-Peer Memory Access".

Returns [CUDA_ERROR_PEER_ACCESS_UNSUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9d60abcaa3f2710f961db8c383bb95cae>) if [cuDeviceCanAccessPeer()](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e> "Queries if a device may directly access a peer device's memory.") indicates that the [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) of the current context cannot directly access memory from the [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>) of `peerContext`.

Returns [CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e947c18606be796573aa2957402fa89a9c>) if direct access of `peerContext` from the current context has already been enabled.

Returns [CUDA_ERROR_TOO_MANY_PEERS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9168ef870793a31ef4cdd7cb6e279b34a>) if direct peer access is not possible because hardware resources required for peer access have been exhausted.

Returns [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>) if there is no current context, `peerContext` is not a valid context, or if the current context is `peerContext`.

Returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if `Flags` is not 0.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceCanAccessPeer](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e> "Queries if a device may directly access a peer device's memory."), [cuCtxDisablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g5b4b6936ea868d4954ce4d841a3b4810> "Disables direct access to memory allocations in a peer context and unregisters any registered allocations."), [cudaDeviceEnablePeerAccess](<../cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g2b0adabf90db37e5cfddc92cbb2589f3>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceCanAccessPeer ( int*Â canAccessPeer, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dev, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â peerDev )


Queries if a device may directly access a peer device's memory.

######  Parameters

`canAccessPeer`
    \- Returned access capability
`dev`
    \- Device from which allocations on `peerDev` are to be directly accessed.
`peerDev`
    \- Device on which the allocations to be directly accessed by `dev` reside.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>)

###### Description

Returns in `*canAccessPeer` a value of 1 if contexts on `dev` are capable of directly accessing memory from contexts on `peerDev` and 0 otherwise. If direct access of `peerDev` from `dev` is possible, then access may be enabled on two specific contexts by calling [cuCtxEnablePeerAccess()](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context.").

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxEnablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context."), [cuCtxDisablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g5b4b6936ea868d4954ce4d841a3b4810> "Disables direct access to memory allocations in a peer context and unregisters any registered allocations."), [cudaDeviceCanAccessPeer](<../cuda-runtime-api/group__CUDART__PEER.html#group__CUDART__PEER_1g4db0d04e44995d5c1c34be4ecc863f22>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetP2PAtomicCapabilities ( unsigned int*Â capabilities, const [CUatomicOperation *](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfd7012de0abfe50cee089f3f00d6dcf3>)*Â operations, unsigned int Â count, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â srcDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dstDevice )


Queries details about atomic operations supported between two devices.

######  Parameters

`capabilities`
    \- Returned capability details of each requested operation
`operations`
    \- Requested operations
`count`
    \- Count of requested operations and size of capabilities
`srcDevice`
    \- The source device of the target link
`dstDevice`
    \- The destination device of the target link

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*capabilities` the details about requested atomic `*operations` over the the link between `srcDevice` and `dstDevice`. The allocated size of `*operations` and `*capabilities` must be `count`.

For each [CUatomicOperation](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfd7012de0abfe50cee089f3f00d6dcf3>) in `*operations`, the corresponding result in `*capabilities` will be a bitmask indicating which of [CUatomicOperationCapability](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf8fbc0f84fd6461c5611b3935b26e22c>) the link supports natively.

Returns [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>) if `srcDevice` or `dstDevice` are not valid or if they represent the same device.

Returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if `*capabilities` or `*operations` is NULL, if `count` is 0, or if any of `*operations` is not valid.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuDeviceGetP2PAttribute](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g4c55c60508f8eba4546b51f2ee545393> "Queries attributes of the link between two devices."), [cudaDeviceGetP2PAttribute](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gc63e5bf168e53b2daf71904eab048fa9>), [cudaDeviceGetP2PAtomicCapabilities](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1ga608cadee2598ca942b362db73267c2b>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuDeviceGetP2PAttribute ( int*Â value, [CUdevice_P2PAttribute](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g578d7cf687ce20f7e99468e8c14e22de>)Â attrib, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â srcDevice, [CUdevice](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g41ca2a24a242b36ef2ca77330b5fb72a>)Â dstDevice )


Queries attributes of the link between two devices.

######  Parameters

`value`
    \- Returned value of the requested attribute
`attrib`
    \- The requested attribute of the link between `srcDevice` and `dstDevice`.
`srcDevice`
    \- The source device of the target link.
`dstDevice`
    \- The destination device of the target link.

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns in `*value` the value of the requested attribute `attrib` of the link between `srcDevice` and `dstDevice`. The supported attributes are:

  * [CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg578d7cf687ce20f7e99468e8c14e22de193d16e6c0ee3a975c184b32586f9fdc>): A relative value indicating the performance of the link between two devices.

  * [CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg578d7cf687ce20f7e99468e8c14e22dec7e28aec0cd03c462a49d00d1b145f46>) P2P: 1 if P2P Access is enable.

  * [CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg578d7cf687ce20f7e99468e8c14e22de810416263d10b9917ac99d35058d6236>): 1 if all CUDA-valid atomic operations over the link are supported.

  * [CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg578d7cf687ce20f7e99468e8c14e22de83a83eb1e8d535b6b8ecf00f509f4097>): 1 if cudaArray can be accessed over the link.

  * [CU_DEVICE_P2P_ATTRIBUTE_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg578d7cf687ce20f7e99468e8c14e22de4a3bdcbd230a998ee38f8ec70bd902a0>): 1 if some CUDA-valid atomic operations over the link are supported. Information about specific operations can be retrieved with [cuDeviceGetP2PAtomicCapabilities](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1gfd989876c8fd3291b520c0b561d5282d> "Queries details about atomic operations supported between two devices.").


Returns [CUDA_ERROR_INVALID_DEVICE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e96f047e7215788ca96c81af92a04bfb6c>) if `srcDevice` or `dstDevice` are not valid or if they represent the same device.

Returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if `attrib` is not valid or if `value` is a null pointer.

Note:

Note that this function may also return error codes from previous, asynchronous launches.

**See also:**

[cuCtxEnablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a> "Enables direct access to memory allocations in a peer context."), [cuCtxDisablePeerAccess](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g5b4b6936ea868d4954ce4d841a3b4810> "Disables direct access to memory allocations in a peer context and unregisters any registered allocations."), [cuDeviceCanAccessPeer](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e> "Queries if a device may directly access a peer device's memory."), [cuDeviceGetP2PAtomicCapabilities](<group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1gfd989876c8fd3291b520c0b561d5282d> "Queries details about atomic operations supported between two devices."), [cudaDeviceGetP2PAttribute](<../cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1gc63e5bf168e53b2daf71904eab048fa9>)

* * *
