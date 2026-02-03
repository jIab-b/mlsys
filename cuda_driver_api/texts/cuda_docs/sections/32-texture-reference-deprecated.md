# Texture Reference (Deprecated)

## 6.26.Â Texture Reference Management [DEPRECATED]

This section describes the deprecated texture reference management functions of the low-level CUDA driver application programming interface.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefCreate](<#group__CUDA__TEXREF__DEPRECATED_1g3b7632ddefba6033dc44bc149793619b>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)*Â pTexRef )
     Creates a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefDestroy](<#group__CUDA__TEXREF__DEPRECATED_1g80c407e5759db31015f50fea94c10fa1>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Destroys a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetAddress](<#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488>) ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pdptr, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the address associated with a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetAddressMode](<#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67>) ( [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>)*Â pam, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, int Â dim )
     Gets the addressing mode used by a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetArray](<#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087>) ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â phArray, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the array bound to a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetBorderColor](<#group__CUDA__TEXREF__DEPRECATED_1g04303cad6225620089ad34ffb50caf48>) ( float*Â pBorderColor, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the border color used by a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetFilterMode](<#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d>) ( [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)*Â pfm, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the filter-mode used by a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetFlags](<#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94>) ( unsigned int*Â pFlags, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the flags used by a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetFormat](<#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8>) ( [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>)*Â pFormat, int*Â pNumChannels, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the format used by a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetMaxAnisotropy](<#group__CUDA__TEXREF__DEPRECATED_1g9e101cc5a0dcab4a9a7c709ab9ecfd1c>) ( int*Â pmaxAniso, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the maximum anisotropy for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetMipmapFilterMode](<#group__CUDA__TEXREF__DEPRECATED_1ge2726d645a4d84df974f9da2f5a85b11>) ( [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)*Â pfm, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the mipmap filtering mode for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetMipmapLevelBias](<#group__CUDA__TEXREF__DEPRECATED_1g46dca9c5a96a5494b60499fe81c15f82>) ( float*Â pbias, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the mipmap level bias for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetMipmapLevelClamp](<#group__CUDA__TEXREF__DEPRECATED_1g7e0b66c45535bd2b753d9860f212d848>) ( float*Â pminMipmapLevelClamp, float*Â pmaxMipmapLevelClamp, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the min/max mipmap level clamps for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefGetMipmappedArray](<#group__CUDA__TEXREF__DEPRECATED_1g3bc191d80a7a6e1cf7405a00fde9131a>) ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â phMipmappedArray, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )
     Gets the mipmapped array bound to a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetAddress](<#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b>) ( size_t*Â ByteOffset, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ bytes )
     Binds an address as a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetAddress2D](<#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, const [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>)*Â desc, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ Pitch )
     Binds an address as a 2D texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetAddressMode](<#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, int Â dim, [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>)Â am )
     Sets the addressing mode for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetArray](<#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray, unsigned int Â Flags )
     Binds an array as a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetBorderColor](<#group__CUDA__TEXREF__DEPRECATED_1g1db39c355bedd9e7ffb00e2011784dea>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, float*Â pBorderColor )
     Sets the border color for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetFilterMode](<#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)Â fm )
     Sets the filtering mode for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetFlags](<#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, unsigned int Â Flags )
     Sets the flags for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetFormat](<#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>)Â fmt, int Â NumPackedComponents )
     Sets the format for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetMaxAnisotropy](<#group__CUDA__TEXREF__DEPRECATED_1g2b144345d6089ec4053c334fb7d04490>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, unsigned int Â maxAniso )
     Sets the maximum anisotropy for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetMipmapFilterMode](<#group__CUDA__TEXREF__DEPRECATED_1g82a54190706dd35d8923966b60f320eb>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)Â fm )
     Sets the mipmap filtering mode for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetMipmapLevelBias](<#group__CUDA__TEXREF__DEPRECATED_1g6d208de7a968f051fc54224883b1994c>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, float Â bias )
     Sets the mipmap level bias for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetMipmapLevelClamp](<#group__CUDA__TEXREF__DEPRECATED_1g9b39decf969353890454895e988e9018>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, float Â minMipmapLevelClamp, float Â maxMipmapLevelClamp )
     Sets the mipmap min/max mipmap level clamps for a texture reference.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexRefSetMipmappedArray](<#group__CUDA__TEXREF__DEPRECATED_1gb35f38ee0738f00c988db5c1ed8c38ea>) ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â hMipmappedArray, unsigned int Â Flags )
     Binds a mipmapped array to a texture reference.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefCreate ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)*Â pTexRef )


Creates a texture reference.

######  Parameters

`pTexRef`
    \- Returned texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000047>)

###### Description

Creates a texture reference and returns its handle in `*pTexRef`. Once created, the application must call [cuTexRefSetArray()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference.") or [cuTexRefSetAddress()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference.") to associate the reference with allocated memory. Other texture reference functions are used to specify the format and interpretation (addressing, filtering, etc.) to be used when the memory is read through this texture reference.

**See also:**

[cuTexRefDestroy](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g80c407e5759db31015f50fea94c10fa1> "Destroys a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefDestroy ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Destroys a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000048>)

###### Description

Destroys the texture reference specified by `hTexRef`.

**See also:**

[cuTexRefCreate](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g3b7632ddefba6033dc44bc149793619b> "Creates a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetAddress ( [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)*Â pdptr, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the address associated with a texture reference.

######  Parameters

`pdptr`
    \- Returned device address
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000035>)

###### Description

Returns in `*pdptr` the base address bound to the texture reference `hTexRef`, or returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if the texture reference is not bound to any device memory range.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetAddressMode ( [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>)*Â pam, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, int Â dim )


Gets the addressing mode used by a texture reference.

######  Parameters

`pam`
    \- Returned addressing mode
`hTexRef`
    \- Texture reference
`dim`
    \- Dimension

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000038>)

###### Description

Returns in `*pam` the addressing mode corresponding to the dimension `dim` of the texture reference `hTexRef`. Currently, the only valid value for `dim` are 0 and 1.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetArray ( [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)*Â phArray, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the array bound to a texture reference.

######  Parameters

`phArray`
    \- Returned array
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000036>)

###### Description

Returns in `*phArray` the CUDA array bound to the texture reference `hTexRef`, or returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if the texture reference is not bound to any CUDA array.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetBorderColor ( float*Â pBorderColor, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the border color used by a texture reference.

######  Parameters

`pBorderColor`
    \- Returned Type and Value of RGBA color
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000045>)

###### Description

Returns in `pBorderColor`, values of the RGBA color used by the texture reference `hTexRef`. The color value is of type float and holds color components in the following sequence: pBorderColor[0] holds 'R' component pBorderColor[1] holds 'G' component pBorderColor[2] holds 'B' component pBorderColor[3] holds 'A' component

**See also:**

[cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetBorderColor](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g1db39c355bedd9e7ffb00e2011784dea> "Sets the border color for a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetFilterMode ( [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)*Â pfm, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the filter-mode used by a texture reference.

######  Parameters

`pfm`
    \- Returned filtering mode
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000039>)

###### Description

Returns in `*pfm` the filtering mode of the texture reference `hTexRef`.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetFlags ( unsigned int*Â pFlags, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the flags used by a texture reference.

######  Parameters

`pFlags`
    \- Returned flags
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000046>)

###### Description

Returns in `*pFlags` the flags of the texture reference `hTexRef`.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetFormat ( [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>)*Â pFormat, int*Â pNumChannels, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the format used by a texture reference.

######  Parameters

`pFormat`
    \- Returned format
`pNumChannels`
    \- Returned number of components
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000040>)

###### Description

Returns in `*pFormat` and `*pNumChannels` the format and number of components of the CUDA array bound to the texture reference `hTexRef`. If `pFormat` or `pNumChannels` is NULL, it will be ignored.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetMaxAnisotropy ( int*Â pmaxAniso, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the maximum anisotropy for a texture reference.

######  Parameters

`pmaxAniso`
    \- Returned maximum anisotropy
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000044>)

###### Description

Returns the maximum anisotropy in `pmaxAniso` that's used when reading memory through the texture reference `hTexRef`.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetMipmapFilterMode ( [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)*Â pfm, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the mipmap filtering mode for a texture reference.

######  Parameters

`pfm`
    \- Returned mipmap filtering mode
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000041>)

###### Description

Returns the mipmap filtering mode in `pfm` that's used when reading memory through the texture reference `hTexRef`.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetMipmapLevelBias ( float*Â pbias, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the mipmap level bias for a texture reference.

######  Parameters

`pbias`
    \- Returned mipmap level bias
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000042>)

###### Description

Returns the mipmap level bias in `pBias` that's added to the specified mipmap level when reading memory through the texture reference `hTexRef`.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetMipmapLevelClamp ( float*Â pminMipmapLevelClamp, float*Â pmaxMipmapLevelClamp, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the min/max mipmap level clamps for a texture reference.

######  Parameters

`pminMipmapLevelClamp`
    \- Returned mipmap min level clamp
`pmaxMipmapLevelClamp`
    \- Returned mipmap max level clamp
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000043>)

###### Description

Returns the min/max mipmap level clamps in `pminMipmapLevelClamp` and `pmaxMipmapLevelClamp` that's used when reading memory through the texture reference `hTexRef`.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefGetMipmappedArray ( [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)*Â phMipmappedArray, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef )


Gets the mipmapped array bound to a texture reference.

######  Parameters

`phMipmappedArray`
    \- Returned mipmapped array
`hTexRef`
    \- Texture reference

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000037>)

###### Description

Returns in `*phMipmappedArray` the CUDA mipmapped array bound to the texture reference `hTexRef`, or returns [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) if the texture reference is not bound to any CUDA mipmapped array.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetAddress ( size_t*Â ByteOffset, [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ bytes )


Binds an address as a texture reference.

######  Parameters

`ByteOffset`
    \- Returned byte offset
`hTexRef`
    \- Texture reference to bind
`dptr`
    \- Device pointer to bind
`bytes`
    \- Size of memory to bind in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000024>)

###### Description

Binds a linear address range to the texture reference `hTexRef`. Any previous address or CUDA array state associated with the texture reference is superseded by this function. Any memory previously bound to `hTexRef` is unbound.

Since the hardware enforces an alignment requirement on texture base addresses, [cuTexRefSetAddress()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference.") passes back a byte offset in `*ByteOffset` that must be applied to texture fetches in order to read from the desired memory. This offset must be divided by the texel size and passed to kernels that read from the texture so they can be applied to the tex1Dfetch() function.

If the device memory pointer was returned from [cuMemAlloc()](<group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467> "Allocates device memory."), the offset is guaranteed to be 0 and NULL may be passed as the `ByteOffset` parameter.

The total number of elements (or texels) in the linear address range cannot exceed [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3335584a4bc5128e2a5ae9a4417f5b758>). The number of elements is computed as (`bytes` / bytesPerElement), where bytesPerElement is determined from the data format and number of components set using [cuTexRefSetFormat()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference.").

**See also:**

[cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetAddress2D ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, const [CUDA_ARRAY_DESCRIPTOR](<structCUDA__ARRAY__DESCRIPTOR__v2.html#structCUDA__ARRAY__DESCRIPTOR__v2>)*Â desc, [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>)Â dptr, size_tÂ Pitch )


Binds an address as a 2D texture reference.

######  Parameters

`hTexRef`
    \- Texture reference to bind
`desc`
    \- Descriptor of CUDA array
`dptr`
    \- Device pointer to bind
`Pitch`
    \- Line pitch in bytes

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000025>)

###### Description

Binds a linear address range to the texture reference `hTexRef`. Any previous address or CUDA array state associated with the texture reference is superseded by this function. Any memory previously bound to `hTexRef` is unbound.

Using a tex2D() function inside a kernel requires a call to either [cuTexRefSetArray()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference.") to bind the corresponding texture reference to an array, or [cuTexRefSetAddress2D()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference.") to bind the texture reference to linear memory.

Function calls to [cuTexRefSetFormat()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference.") cannot follow calls to [cuTexRefSetAddress2D()](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference.") for the same texture reference.

It is required that `dptr` be aligned to the appropriate hardware-specific texture alignment. You can query this value using the device attribute [CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a39531a2b5f533e749109e9e0189f38196>). If an unaligned `dptr` is supplied, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

`Pitch` has to be aligned to the hardware-specific texture pitch alignment. This value can be queried using the device attribute [CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3551f1067be9a6187d75da5fcda7960d0>). If an unaligned `Pitch` is supplied, [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

Width and Height, which are specified in elements (or texels), cannot exceed [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3afe638125896be2c465876a4955d699e>) and [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3ff1c959cba47edce2374f66f161489c4>) respectively. `Pitch`, which is specified in bytes, cannot exceed [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3ad66050d059c337dd9635bfb7574f3d7>).

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetAddressMode ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, int Â dim, [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>)Â am )


Sets the addressing mode for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`dim`
    \- Dimension
`am`
    \- Addressing mode to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000027>)

###### Description

Specifies the addressing mode `am` for the given dimension `dim` of the texture reference `hTexRef`. If `dim` is zero, the addressing mode is applied to the first parameter of the functions used to fetch from the texture; if `dim` is 1, the second, and so on. [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>) is defined as:


    â   typedef enum CUaddress_mode_enum {
                [CU_TR_ADDRESS_MODE_WRAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff760888787d43274db1dfed07818895b04197fcb>) = 0,
                [CU_TR_ADDRESS_MODE_CLAMP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088878ed20ebe21592443f61ecc06d61f32f4>) = 1,
                [CU_TR_ADDRESS_MODE_MIRROR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088872490f71f92668604bd10d49d77d198b8>) = 2,
                [CU_TR_ADDRESS_MODE_BORDER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff7608887ccee8c7882028f0865e8a2e542524fa4>) = 3
             } [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>);

Note that this call has no effect if `hTexRef` is bound to linear memory. Also, if the flag, [CU_TRSF_NORMALIZED_COORDINATES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7a19eb49fd506ecded6e8f314298d486>), is not set, the only supported address mode is [CU_TR_ADDRESS_MODE_CLAMP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088878ed20ebe21592443f61ecc06d61f32f4>).

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetArray ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>)Â hArray, unsigned int Â Flags )


Binds an array as a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference to bind
`hArray`
    \- Array to bind
`Flags`
    \- Options (must be [CU_TRSA_OVERRIDE_FORMAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f0c76f9c215b3bdeca06456bec3e68>))

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000022>)

###### Description

Binds the CUDA array `hArray` to the texture reference `hTexRef`. Any previous address or CUDA array state associated with the texture reference is superseded by this function. `Flags` must be set to [CU_TRSA_OVERRIDE_FORMAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f0c76f9c215b3bdeca06456bec3e68>). Any CUDA array previously bound to `hTexRef` is unbound.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetBorderColor ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, float*Â pBorderColor )


Sets the border color for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`pBorderColor`
    \- RGBA color

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000033>)

###### Description

Specifies the value of the RGBA color via the `pBorderColor` to the texture reference `hTexRef`. The color value supports only float type and holds color components in the following sequence: pBorderColor[0] holds 'R' component pBorderColor[1] holds 'G' component pBorderColor[2] holds 'B' component pBorderColor[3] holds 'A' component

Note that the color values can be set only when the Address mode is set to CU_TR_ADDRESS_MODE_BORDER using [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."). Applications using integer border color values have to "reinterpret_cast" their values to float.

**See also:**

[cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetBorderColor](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g04303cad6225620089ad34ffb50caf48> "Gets the border color used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetFilterMode ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)Â fm )


Sets the filtering mode for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`fm`
    \- Filtering mode to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000028>)

###### Description

Specifies the filtering mode `fm` to be used when reading memory through the texture reference `hTexRef`. CUfilter_mode_enum is defined as:


    â   typedef enum CUfilter_mode_enum {
                [CU_TR_FILTER_MODE_POINT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2ae1e747d9e41685f6b6a5b85baf43e60d>) = 0,
                [CU_TR_FILTER_MODE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2a517adfbd8a0e09592378d77ba2d922d8>) = 1
             } [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>);

Note that this call has no effect if `hTexRef` is bound to linear memory.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetFlags ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, unsigned int Â Flags )


Sets the flags for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`Flags`
    \- Optional flags to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000034>)

###### Description

Specifies optional flags via `Flags` to specify the behavior of data returned through the texture reference `hTexRef`. The valid flags are:

  * [CU_TRSF_READ_AS_INTEGER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d2387c1b5dd5bc98f5b4c51cefdf41e>), which suppresses the default behavior of having the texture promote integer data to floating point data in the range [0, 1]. Note that texture with 32-bit integer format would not be promoted, regardless of whether or not this flag is specified;

  * [CU_TRSF_NORMALIZED_COORDINATES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7a19eb49fd506ecded6e8f314298d486>), which suppresses the default behavior of having the texture coordinates range from [0, Dim) where Dim is the width or height of the CUDA array. Instead, the texture coordinates [0, 1.0) reference the entire breadth of the array dimension;

  * [CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9512d7fa0ed8a2da30ef6f4ccc61fa4f>), which disables any trilinear filtering optimizations. Trilinear optimizations improve texture filtering performance by allowing bilinear filtering on textures in scenarios where it can closely approximate the expected results.


**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetFormat ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>)Â fmt, int Â NumPackedComponents )


Sets the format for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`fmt`
    \- Format to set
`NumPackedComponents`
    \- Number of components per array element

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000026>)

###### Description

Specifies the format of the data to be read by the texture reference `hTexRef`. `fmt` and `NumPackedComponents` are exactly analogous to the Format and NumChannels members of the CUDA_ARRAY_DESCRIPTOR structure: They specify the format of each component and the number of components per array element.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference."), [cudaCreateChannelDesc](<../cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g655725c27d8ffe75accb9b531ecf2d15>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetMaxAnisotropy ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, unsigned int Â maxAniso )


Sets the maximum anisotropy for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`maxAniso`
    \- Maximum anisotropy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000032>)

###### Description

Specifies the maximum anisotropy `maxAniso` to be used when reading memory through the texture reference `hTexRef`.

Note that this call has no effect if `hTexRef` is bound to linear memory.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetMipmapFilterMode ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>)Â fm )


Sets the mipmap filtering mode for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`fm`
    \- Filtering mode to set

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000029>)

###### Description

Specifies the mipmap filtering mode `fm` to be used when reading memory through the texture reference `hTexRef`. CUfilter_mode_enum is defined as:


    â   typedef enum CUfilter_mode_enum {
                [CU_TR_FILTER_MODE_POINT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2ae1e747d9e41685f6b6a5b85baf43e60d>) = 0,
                [CU_TR_FILTER_MODE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2a517adfbd8a0e09592378d77ba2d922d8>) = 1
             } [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>);

Note that this call has no effect if `hTexRef` is not bound to a mipmapped array.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetMipmapLevelBias ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, float Â bias )


Sets the mipmap level bias for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`bias`
    \- Mipmap level bias

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000030>)

###### Description

Specifies the mipmap level bias `bias` to be added to the specified mipmap level when reading memory through the texture reference `hTexRef`.

Note that this call has no effect if `hTexRef` is not bound to a mipmapped array.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetMipmapLevelClamp ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, float Â minMipmapLevelClamp, float Â maxMipmapLevelClamp )


Sets the mipmap min/max mipmap level clamps for a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference
`minMipmapLevelClamp`
    \- Mipmap min level clamp
`maxMipmapLevelClamp`
    \- Mipmap max level clamp

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000031>)

###### Description

Specifies the min/max mipmap level clamps, `minMipmapLevelClamp` and `maxMipmapLevelClamp` respectively, to be used when reading memory through the texture reference `hTexRef`.

Note that this call has no effect if `hTexRef` is not bound to a mipmapped array.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gac3a34b4b10983433865fdadb83b9118> "Binds an array as a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexRefSetMipmappedArray ( [CUtexref](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gec1e8eb9dc48ad748765d1fcc020d6b5>)Â hTexRef, [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>)Â hMipmappedArray, unsigned int Â Flags )


Binds a mipmapped array to a texture reference.

######  Parameters

`hTexRef`
    \- Texture reference to bind
`hMipmappedArray`
    \- Mipmapped array to bind
`Flags`
    \- Options (must be [CU_TRSA_OVERRIDE_FORMAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f0c76f9c215b3bdeca06456bec3e68>))

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### [Deprecated](<deprecated.html#deprecated__deprecated_1_deprecated000023>)

###### Description

Binds the CUDA mipmapped array `hMipmappedArray` to the texture reference `hTexRef`. Any previous address or CUDA array state associated with the texture reference is superseded by this function. `Flags` must be set to [CU_TRSA_OVERRIDE_FORMAT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gf9f0c76f9c215b3bdeca06456bec3e68>). Any CUDA array previously bound to `hTexRef` is unbound.

**See also:**

[cuTexRefSetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga6e288992f58e7a6e3350614bc9e813b> "Binds an address as a texture reference."), [cuTexRefSetAddress2D](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1gbdec8983628f68bcde5db4b4c3f90851> "Binds an address as a 2D texture reference."), [cuTexRefSetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga45732a5c4ec291c0682fffcbaa6d393> "Sets the addressing mode for a texture reference."), [cuTexRefSetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g93819286c48db39afc253c0f10358d2e> "Sets the filtering mode for a texture reference."), [cuTexRefSetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g9d4816a6561e1d09e0eef9f9c0cdbfa2> "Sets the flags for a texture reference."), [cuTexRefSetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g2d57eabbd5ef6780307c008b0f4ce83d> "Sets the format for a texture reference."), [cuTexRefGetAddress](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g56a175420c7fef8e547a66bc79671488> "Gets the address associated with a texture reference."), [cuTexRefGetAddressMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1ga41ceb7f8a452d59ae7e874b1c8e0c67> "Gets the addressing mode used by a texture reference."), [cuTexRefGetArray](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g7f74aff0d999af6613dfc9aff3a21087> "Gets the array bound to a texture reference."), [cuTexRefGetFilterMode](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g941748b72fe9c6f9767be38b8d02c95d> "Gets the filter-mode used by a texture reference."), [cuTexRefGetFlags](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0eb1b87656d661c4fbd4ddbbc0dd7b94> "Gets the flags used by a texture reference."), [cuTexRefGetFormat](<group__CUDA__TEXREF__DEPRECATED.html#group__CUDA__TEXREF__DEPRECATED_1g0453d286f81825e1d503d651f8b079d8> "Gets the format used by a texture reference.")

* * *
