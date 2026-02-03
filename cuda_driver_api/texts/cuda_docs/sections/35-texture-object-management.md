# Texture Object Management

## 6.28.Â Texture Object Management

This section describes the texture object management functions of the low-level CUDA driver application programming interface. The texture object API is only supported on devices of compute capability 3.0 or higher.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexObjectCreate](<#group__CUDA__TEXOBJECT_1g1f6dd0f9cbf56db725b1f45aa0a7218a>) ( [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)*Â pTexObject, const [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc, const [CUDA_TEXTURE_DESC](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1>)*Â pTexDesc, const [CUDA_RESOURCE_VIEW_DESC](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1>)*Â pResViewDesc )
     Creates a texture object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexObjectDestroy](<#group__CUDA__TEXOBJECT_1gcd522ba5e2d1852aff8c0388f66247fd>) ( [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )
     Destroys a texture object.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexObjectGetResourceDesc](<#group__CUDA__TEXOBJECT_1g0cc8eb2fa1e584d2b04d631586d0921f>) ( [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc, [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )
     Returns a texture object's resource descriptor.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexObjectGetResourceViewDesc](<#group__CUDA__TEXOBJECT_1g185fa4c933a1c3a7b6aebe3e4291a37b>) ( [CUDA_RESOURCE_VIEW_DESC](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1>)*Â pResViewDesc, [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )
     Returns a texture object's resource view descriptor.
[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>)Â [cuTexObjectGetTextureDesc](<#group__CUDA__TEXOBJECT_1g688de37b844df7313c8fce30fc912645>) ( [CUDA_TEXTURE_DESC](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1>)*Â pTexDesc, [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )
     Returns a texture object's texture descriptor.

### Functions

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexObjectCreate ( [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)*Â pTexObject, const [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc, const [CUDA_TEXTURE_DESC](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1>)*Â pTexDesc, const [CUDA_RESOURCE_VIEW_DESC](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1>)*Â pResViewDesc )


Creates a texture object.

######  Parameters

`pTexObject`
    \- Texture object to create
`pResDesc`
    \- Resource descriptor
`pTexDesc`
    \- Texture descriptor
`pResViewDesc`
    \- Resource view descriptor

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Creates a texture object and returns it in `pTexObject`. `pResDesc` describes the data to texture from. `pTexDesc` describes how the data should be sampled. `pResViewDesc` is an optional argument that specifies an alternate format for the data described by `pResDesc`, and also describes the subresource region to restrict access to when texturing. `pResViewDesc` can only be specified if the type of resource is a CUDA array or a CUDA mipmapped array not in a block compressed format.

Texture objects are only supported on devices of compute capability 3.0 or higher. Additionally, a texture object is an opaque value, and, as such, should only be accessed through CUDA API calls.

The CUDA_RESOURCE_DESC structure is defined as:


    â        typedef struct CUDA_RESOURCE_DESC_st
                  {
                      [CUresourcetype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9f0a76c9f6be437e75c8310aea5280f6>) resType;

                      union {
                          struct {
                              [CUarray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gd550651524a56766b60f10f0e7628042>) hArray;
                          } array;
                          struct {
                              [CUmipmappedArray](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g96db856ab3d2940fb694ce4501d9b583>) hMipmappedArray;
                          } mipmap;
                          struct {
                              [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) devPtr;
                              [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) format;
                              unsigned int numChannels;
                              size_t sizeInBytes;
                          } linear;
                          struct {
                              [CUdeviceptr](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g183f7b0d8ad008ea2a5fd552537ace4e>) devPtr;
                              [CUarray_format](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9b009d9a6aa4c5765c8a00289b6068f9>) format;
                              unsigned int numChannels;
                              size_t width;
                              size_t height;
                              size_t pitchInBytes;
                          } pitch2D;
                      } res;

                      unsigned int flags;
                  } [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>);

where:

  * [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) specifies the type of resource to texture from. CUresourceType is defined as:

        â        typedef enum CUresourcetype_enum {
                          [CU_RESOURCE_TYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f68171f299e8447a926051e13d613d77b1>)           = 0x00,
                          [CU_RESOURCE_TYPE_MIPMAPPED_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f642868e220af0309016ec733e37db7f24>) = 0x01,
                          [CU_RESOURCE_TYPE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f6ba58dadf78cb83742b2a0afe39256f87>)          = 0x02,
                          [CU_RESOURCE_TYPE_PITCH2D](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f62ba314f961b37dd487278b6894070dea>)         = 0x03
                      } [CUresourcetype](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9f0a76c9f6be437e75c8310aea5280f6>);


If [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) is set to [CU_RESOURCE_TYPE_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f68171f299e8447a926051e13d613d77b1>), CUDA_RESOURCE_DESC::res::array::hArray must be set to a valid CUDA array handle.

If [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) is set to [CU_RESOURCE_TYPE_MIPMAPPED_ARRAY](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f642868e220af0309016ec733e37db7f24>), CUDA_RESOURCE_DESC::res::mipmap::hMipmappedArray must be set to a valid CUDA mipmapped array handle.

If [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) is set to [CU_RESOURCE_TYPE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f6ba58dadf78cb83742b2a0afe39256f87>), CUDA_RESOURCE_DESC::res::linear::devPtr must be set to a valid device pointer, that is aligned to [CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a39531a2b5f533e749109e9e0189f38196>). CUDA_RESOURCE_DESC::res::linear::format and CUDA_RESOURCE_DESC::res::linear::numChannels describe the format of each component and the number of components per array element. CUDA_RESOURCE_DESC::res::linear::sizeInBytes specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3335584a4bc5128e2a5ae9a4417f5b758>). The number of elements is computed as (sizeInBytes / (sizeof(format) * numChannels)).

If [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) is set to [CU_RESOURCE_TYPE_PITCH2D](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f62ba314f961b37dd487278b6894070dea>), CUDA_RESOURCE_DESC::res::pitch2D::devPtr must be set to a valid device pointer, that is aligned to [CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a39531a2b5f533e749109e9e0189f38196>). CUDA_RESOURCE_DESC::res::pitch2D::format and CUDA_RESOURCE_DESC::res::pitch2D::numChannels describe the format of each component and the number of components per array element. CUDA_RESOURCE_DESC::res::pitch2D::width and CUDA_RESOURCE_DESC::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3afe638125896be2c465876a4955d699e>) and [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3ff1c959cba47edce2374f66f161489c4>) respectively. CUDA_RESOURCE_DESC::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to [CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3551f1067be9a6187d75da5fcda7960d0>). Pitch cannot exceed [CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gge12b8a782bebe21b1ac0091bf9f4e2a3ad66050d059c337dd9635bfb7574f3d7>).

  * flags must be set to zero.


The CUDA_TEXTURE_DESC struct is defined as


    â        typedef struct CUDA_TEXTURE_DESC_st {
                      [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>) addressMode[3];
                      [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>) filterMode;
                      unsigned int flags;
                      unsigned int maxAnisotropy;
                      [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>) mipmapFilterMode;
                      float mipmapLevelBias;
                      float minMipmapLevelClamp;
                      float maxMipmapLevelClamp;
                  } [CUDA_TEXTURE_DESC](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1>);

where

  * [CUDA_TEXTURE_DESC::addressMode](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1aeb3f6fa73835433a7700b80eea8d49b>) specifies the addressing mode for each dimension of the texture data. [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>) is defined as:

        â        typedef enum CUaddress_mode_enum {
                          [CU_TR_ADDRESS_MODE_WRAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff760888787d43274db1dfed07818895b04197fcb>) = 0,
                          [CU_TR_ADDRESS_MODE_CLAMP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088878ed20ebe21592443f61ecc06d61f32f4>) = 1,
                          [CU_TR_ADDRESS_MODE_MIRROR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088872490f71f92668604bd10d49d77d198b8>) = 2,
                          [CU_TR_ADDRESS_MODE_BORDER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff7608887ccee8c7882028f0865e8a2e542524fa4>) = 3
                      } [CUaddress_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc925457ee7128d6251071f6ff7608887>);

This is ignored if [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) is [CU_RESOURCE_TYPE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f6ba58dadf78cb83742b2a0afe39256f87>). Also, if the flag, [CU_TRSF_NORMALIZED_COORDINATES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7a19eb49fd506ecded6e8f314298d486>) is not set, the only supported address mode is [CU_TR_ADDRESS_MODE_CLAMP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088878ed20ebe21592443f61ecc06d61f32f4>).


  * [CUDA_TEXTURE_DESC::filterMode](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_182e92599f2f13422d8cc6cfe947e6b17>) specifies the filtering mode to be used when fetching from the texture. CUfilter_mode is defined as:

        â        typedef enum CUfilter_mode_enum {
                          [CU_TR_FILTER_MODE_POINT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2ae1e747d9e41685f6b6a5b85baf43e60d>) = 0,
                          [CU_TR_FILTER_MODE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2a517adfbd8a0e09592378d77ba2d922d8>) = 1
                      } [CUfilter_mode](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g4fb799d90872f1d6cd074b4349f37c2a>);

This is ignored if [CUDA_RESOURCE_DESC::resType](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1_1fe341889f4a57165e7acc0efcfc38b64>) is [CU_RESOURCE_TYPE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9f0a76c9f6be437e75c8310aea5280f6ba58dadf78cb83742b2a0afe39256f87>).


  * [CUDA_TEXTURE_DESC::flags](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1f50575c45ea8767561db54f4e785cabd>) can be any combination of the following:
    * [CU_TRSF_READ_AS_INTEGER](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g6d2387c1b5dd5bc98f5b4c51cefdf41e>), which suppresses the default behavior of having the texture promote integer data to floating point data in the range [0, 1]. Note that texture with 32-bit integer format would not be promoted, regardless of whether or not this flag is specified.

    * [CU_TRSF_NORMALIZED_COORDINATES](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g7a19eb49fd506ecded6e8f314298d486>), which suppresses the default behavior of having the texture coordinates range from [0, Dim) where Dim is the width or height of the CUDA array. Instead, the texture coordinates [0, 1.0) reference the entire breadth of the array dimension; Note that for CUDA mipmapped arrays, this flag has to be set.

    * [CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g9512d7fa0ed8a2da30ef6f4ccc61fa4f>), which disables any trilinear filtering optimizations. Trilinear optimizations improve texture filtering performance by allowing bilinear filtering on textures in scenarios where it can closely approximate the expected results.

    * [CU_TRSF_SEAMLESS_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g8517a99047de21e19986531ad3958e22>), which enables seamless cube map filtering. This flag can only be specified if the underlying resource is a CUDA array or a CUDA mipmapped array that was created with the flag [CUDA_ARRAY3D_CUBEMAP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gfce9ad9aa3df839571b84b47febfb7ae>). When seamless cube map filtering is enabled, texture address modes specified by [CUDA_TEXTURE_DESC::addressMode](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1aeb3f6fa73835433a7700b80eea8d49b>) are ignored. Instead, if the [CUDA_TEXTURE_DESC::filterMode](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_182e92599f2f13422d8cc6cfe947e6b17>) is set to [CU_TR_FILTER_MODE_POINT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2ae1e747d9e41685f6b6a5b85baf43e60d>) the address mode [CU_TR_ADDRESS_MODE_CLAMP](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc925457ee7128d6251071f6ff76088878ed20ebe21592443f61ecc06d61f32f4>) will be applied for all dimensions. If the [CUDA_TEXTURE_DESC::filterMode](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_182e92599f2f13422d8cc6cfe947e6b17>) is set to [CU_TR_FILTER_MODE_LINEAR](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg4fb799d90872f1d6cd074b4349f37c2a517adfbd8a0e09592378d77ba2d922d8>) seamless cube map filtering will be performed when sampling along the cube face borders.


  * [CUDA_TEXTURE_DESC::maxAnisotropy](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_139035372a07e031d233445673427f34a>) specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be clamped to the range [1,16].


  * [CUDA_TEXTURE_DESC::mipmapFilterMode](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_117333d832def1d97420ddd0bce8c73ce>) specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.


  * [CUDA_TEXTURE_DESC::mipmapLevelBias](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_10b1c10e7f1eedc4a8ab547da5741ce2d>) specifies the offset to be applied to the calculated mipmap level.


  * [CUDA_TEXTURE_DESC::minMipmapLevelClamp](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1bf7da8359ccf52ba8afcd28c31b48be8>) specifies the lower end of the mipmap level range to clamp access to.


  * [CUDA_TEXTURE_DESC::maxMipmapLevelClamp](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1149eca11ac6059979a1754bffcc6c210>) specifies the upper end of the mipmap level range to clamp access to.


The CUDA_RESOURCE_VIEW_DESC struct is defined as


    â        typedef struct CUDA_RESOURCE_VIEW_DESC_st
                  {
                      [CUresourceViewFormat](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ge7db5e5fe7c197287e55f2e97289dfcf>) format;
                      size_t width;
                      size_t height;
                      size_t depth;
                      unsigned int firstMipmapLevel;
                      unsigned int lastMipmapLevel;
                      unsigned int firstLayer;
                      unsigned int lastLayer;
                  } [CUDA_RESOURCE_VIEW_DESC](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1>);

where:

  * [CUDA_RESOURCE_VIEW_DESC::format](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_17ee307ae64bba468ea89bb502d3f8386>) specifies how the data contained in the CUDA array or CUDA mipmapped array should be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block compressed format, then the underlying CUDA array or CUDA mipmapped array has to have a base of format [CU_AD_FORMAT_UNSIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f952b891ad5d4080db0fb2e23fe71614a0>). with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying CUDA array to have a format of [CU_AD_FORMAT_UNSIGNED_INT32](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gg9b009d9a6aa4c5765c8a00289b6068f952b891ad5d4080db0fb2e23fe71614a0>) with 2 channels. The other BC formats require the underlying resource to have the same base format but with 4 channels.


  * [CUDA_RESOURCE_VIEW_DESC::width](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_17a74f483b2af8a73d2a1876688926e63>) specifies the new width of the texture data. If the resource view format is a block compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats, this value has to be equal to that of the original resource.


  * [CUDA_RESOURCE_VIEW_DESC::height](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_1a2737a7f88568199ab94d7c3f696bed3>) specifies the new height of the texture data. If the resource view format is a block compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats, this value has to be equal to that of the original resource.


  * [CUDA_RESOURCE_VIEW_DESC::depth](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_164ca74e2623b821d8dfbbabbc5c839f6>) specifies the new depth of the texture data. This value has to be equal to that of the original resource.


  * [CUDA_RESOURCE_VIEW_DESC::firstMipmapLevel](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_134105380b4498ef2a7b6f9898f983df3>) specifies the most detailed mipmap level. This will be the new mipmap level zero. For non-mipmapped resources, this value has to be zero.[CUDA_TEXTURE_DESC::minMipmapLevelClamp](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1bf7da8359ccf52ba8afcd28c31b48be8>) and [CUDA_TEXTURE_DESC::maxMipmapLevelClamp](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1_1149eca11ac6059979a1754bffcc6c210>) will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified, then the actual minimum mipmap level clamp will be 3.2.


  * [CUDA_RESOURCE_VIEW_DESC::lastMipmapLevel](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_16e0dc4da9bcf7518fc66aee8f0dd928e>) specifies the least detailed mipmap level. For non-mipmapped resources, this value has to be zero.


  * [CUDA_RESOURCE_VIEW_DESC::firstLayer](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_154262bb560a5c52aa73dfe97077334f9>) specifies the first layer index for layered textures. This will be the new layer zero. For non-layered resources, this value has to be zero.


  * [CUDA_RESOURCE_VIEW_DESC::lastLayer](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1_14f2845ad9438911320a33e3fb1017964>) specifies the last layer index for layered textures. For non-layered resources, this value has to be zero.


**See also:**

[cuTexObjectDestroy](<group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT_1gcd522ba5e2d1852aff8c0388f66247fd> "Destroys a texture object."), [cudaCreateTextureObject](<../cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g16ac75814780c3a16e4c63869feb9ad3>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexObjectDestroy ( [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )


Destroys a texture object.

######  Parameters

`texObject`
    \- Texture object to destroy

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Destroys the texture object specified by `texObject`.

**See also:**

[cuTexObjectCreate](<group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT_1g1f6dd0f9cbf56db725b1f45aa0a7218a> "Creates a texture object."), [cudaDestroyTextureObject](<../cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g27be12e215f162cc877be94390da75bb>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexObjectGetResourceDesc ( [CUDA_RESOURCE_DESC](<structCUDA__RESOURCE__DESC__v1.html#structCUDA__RESOURCE__DESC__v1>)*Â pResDesc, [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )


Returns a texture object's resource descriptor.

######  Parameters

`pResDesc`
    \- Resource descriptor
`texObject`
    \- Texture object

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the resource descriptor for the texture object specified by `texObject`.

**See also:**

[cuTexObjectCreate](<group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT_1g1f6dd0f9cbf56db725b1f45aa0a7218a> "Creates a texture object."), [cudaGetTextureObjectResourceDesc](<../cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g4ac6e3f033c356ecc4ab6fb85154f066>),

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexObjectGetResourceViewDesc ( [CUDA_RESOURCE_VIEW_DESC](<structCUDA__RESOURCE__VIEW__DESC__v1.html#structCUDA__RESOURCE__VIEW__DESC__v1>)*Â pResViewDesc, [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )


Returns a texture object's resource view descriptor.

######  Parameters

`pResViewDesc`
    \- Resource view descriptor
`texObject`
    \- Texture object

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the resource view descriptor for the texture object specified by `texObject`. If no resource view was set for `texObject`, the [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>) is returned.

**See also:**

[cuTexObjectCreate](<group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT_1g1f6dd0f9cbf56db725b1f45aa0a7218a> "Creates a texture object."), [cudaGetTextureObjectResourceViewDesc](<../cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g0332bef8105771003c64d7f09d6163fe>)

[CUresult](<group__CUDA__TYPES.html#group__CUDA__TYPES_1gc6c391505e117393cc2558fff6bfc2e9>) cuTexObjectGetTextureDesc ( [CUDA_TEXTURE_DESC](<structCUDA__TEXTURE__DESC__v1.html#structCUDA__TEXTURE__DESC__v1>)*Â pTexDesc, [CUtexObject](<group__CUDA__TYPES.html#group__CUDA__TYPES_1g65fb6720dea73d56db0b4d4974be052d>)Â texObject )


Returns a texture object's texture descriptor.

######  Parameters

`pTexDesc`
    \- Texture descriptor
`texObject`
    \- Texture object

###### Returns

[CUDA_SUCCESS](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a0eed720f8a87cd1c5fd1c453bc7a03d>), [CUDA_ERROR_DEINITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9acf52f132faf29b473cdda6061f0f44a>), [CUDA_ERROR_NOT_INITIALIZED](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e98feb999f0af99b4a25ab26b3866f4df8>), [CUDA_ERROR_INVALID_CONTEXT](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e9a484e9af32c1e9893ff21f0e0191a12d>), [CUDA_ERROR_INVALID_VALUE](<group__CUDA__TYPES.html#group__CUDA__TYPES_1ggc6c391505e117393cc2558fff6bfc2e990696c86fcee1f536a1ec7d25867feeb>)

###### Description

Returns the texture descriptor for the texture object specified by `texObject`.

**See also:**

[cuTexObjectCreate](<group__CUDA__TEXOBJECT.html#group__CUDA__TEXOBJECT_1g1f6dd0f9cbf56db725b1f45aa0a7218a> "Creates a texture object."), [cudaGetTextureObjectTextureDesc](<../cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html#group__CUDART__TEXTURE__OBJECT_1g152565714ff9dce6867b6099afc05e50>)

* * *
