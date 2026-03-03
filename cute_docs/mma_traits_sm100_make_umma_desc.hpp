// Source (exact function definition copied from):
// https://gitee.com/xurr2020/cutlass/blob/main/include/cute/atom/mma_traits_sm100.hpp
// Upstream path:
//   include/cute/atom/mma_traits_sm100.hpp

template <UMMA::Major MajorMode, class TEngine, class TLayout>
CUTE_HOST_DEVICE constexpr
SmemDescriptor
make_umma_desc(Tensor<TEngine,TLayout> const& tensor)
{
  static_assert(is_smem<TEngine>::value, "UMMA Descriptors can only be constructed on smem.");
  static_assert(TLayout::rank == 2, "UMMA Descriptors can only be constructed on rank-2 tensors.");
  using value_type = typename TEngine::value_type;


  Tensor u128_tensor = recast<uint128_t const>(tensor);


  // Result
  SmemDescriptor desc;
  desc.version_ = 1;     // Set the version for blackwell
  desc.lbo_mode_ = 0; // set to legacy mode by default


  // Layout type
  constexpr UMMA::LayoutType LAYOUT_TYPE = UMMA::layout_type(u128_tensor);
  desc.layout_type_ = uint8_t(LAYOUT_TYPE);


  // Start address (4LSB not included)
  uint32_t start_address = cast_smem_ptr_to_uint(raw_pointer_cast(u128_tensor.data()));
  desc.start_address_ = static_cast<uint16_t>(start_address >> 4);


  constexpr uint8_t base_offset = 0;
  desc.base_offset_ = base_offset;


  // LayoutType meta
  constexpr int SwizzleAtomMNSize = LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE         ? 1 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_32B          ? 2 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_64B          ? 4 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_128B         ? 8 :
                                    LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 8 : -1;


  if constexpr (MajorMode == UMMA::Major::MN)
  {
    /* In units of uint128_t, each UmmaDescriptor Major-MN describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE         : Swizzle<0,4,3> o smem_ptr o ((1,n),(8,k)):((X,SBO),(1,LBO))
     * LayoutType::B32                : Swizzle<1,4,3> o smem_ptr o ((2,n),(8,k)):((1,LBO),(2,SBO))
     * LayoutType::B64                : Swizzle<2,4,3> o smem_ptr o ((4,n),(8,k)):((1,LBO),(4,SBO))
     * LayoutType::B128               : Swizzle<3,4,3> o smem_ptr o ((8,n),(8,k)):((1,LBO),(8,SBO))
     * LayoutType::B128_BASE32B       : Swizzle<2,5,2> o smem_ptr o ((8,n),(4,k)):((1,LBO),(4,SBO))
     */


    constexpr int SwizzleAtomKSize = LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 4 : 8;


    // Construct the canonical UMMA T Layout with shape
    //    ((SwizzleAtomMNSize,n),(SwizzleAtomKSize,2))
    Layout canonical_layout =
        logical_divide(layout(u128_tensor),
                       make_tile(Layout<Int<SwizzleAtomMNSize>, _1>{},
                                 Layout<Int<SwizzleAtomKSize>,  _1>{}));


    // Check ranks of canonical
    CUTE_STATIC_ASSERT_V(rank<0>(canonical_layout) == Int<2>{}, "Not a canonical UMMA_MN Layout: No flat offset mode");
    CUTE_STATIC_ASSERT_V(rank<1>(canonical_layout) == Int<2>{}, "Not a canonical UMMA_MN Layout: No flat offset mode");
    // Check canonical mode strides
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE ? stride<0,0>(canonical_layout) : 1;
    static_assert(stride_00 == expected_stride_00, "Not a canonical UMMA_MN Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = SwizzleAtomMNSize;
    static_assert(stride_10 == expected_stride_10, "Not a canonical UMMA_MN Layout: Expected stride failure.");


    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);
    constexpr uint32_t stride_11 = stride<1,1>(canonical_layout);


    desc.stride_byte_offset_  = (LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE) ? stride_01 : stride_11;
    desc.leading_byte_offset_ = (LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE) ? stride_11 : stride_01;
  } else
  if constexpr (MajorMode == UMMA::Major::K)
  {
    /* In units of uint128_t, each UmmaDescriptor Major-K describes a canonical layout of the form
     *
     * LayoutType::INTERLEAVE    : Swizzle<0,4,3> o smem_ptr o ((8,n),2):((1,SBO),LBO)
     * LayoutType::B32           : Swizzle<1,4,3> o smem_ptr o ((8,n),2):((2,SBO),1)
     * LayoutType::B64           : Swizzle<2,4,3> o smem_ptr o ((8,n),2):((4,SBO),1)
     * LayoutType::B128          : Swizzle<3,4,3> o smem_ptr o ((8,n),2):((8,SBO),1)
     * LayoutType::B128_BASE32B  : Not applicable for Major-K
     */


    static_assert(LAYOUT_TYPE != UMMA::LayoutType::SWIZZLE_128B_BASE32B, "SWIZZLE_128B_BASE32B is invalid for Major-K");
    CUTE_STATIC_ASSERT_V(size<0>(u128_tensor) % Int<8>{} == Int<0>{},          // N|M size
                         "Not a canonical UMMA_K Layout: Expected MN-size multiple of 8.");


    // Construct the canonical UMMA N Layout with shape ((8,n),(2,1))
    Layout canonical_layout = logical_divide(layout(u128_tensor), make_tile(Layout<_8,_1>{}, Layout<_2,_1>{}));


    // Check ranks of canonical
    CUTE_STATIC_ASSERT_V(rank<0>(canonical_layout) == Int<2>{}, "Not a canonical UMMA_K Layout: No flat offset mode");
    CUTE_STATIC_ASSERT_V(rank<1>(canonical_layout) == Int<2>{}, "Not a canonical UMMA_K Layout: No flat offset mode");
    // Check canonical mode strides
    constexpr uint32_t stride_00 = stride<0,0>(canonical_layout);
    constexpr uint32_t expected_stride_00 = SwizzleAtomMNSize;
    static_assert(stride_00 == expected_stride_00, "Not a canonical UMMA_K Layout: Expected stride failure.");
    constexpr uint32_t stride_10 = stride<1,0>(canonical_layout);
    constexpr uint32_t expected_stride_10 = (LAYOUT_TYPE == UMMA::LayoutType::SWIZZLE_NONE) ? stride<1,0>(canonical_layout) : 1;
    static_assert(stride_10 == expected_stride_10, "Not a canonical UMMA_K Layout: Expected stride failure.");


    // stride dimension byte offset and leading dimension byte offset (4LSB not included == uint128_t units)
    constexpr uint32_t stride_01 = stride<0,1>(canonical_layout);


    desc.stride_byte_offset_  = stride_01;
    desc.leading_byte_offset_ = stride_10;
  } else {
    static_assert(MajorMode != UMMA::Major::MN && MajorMode != UMMA::Major::K, "Unrecognized MajorMode!");
  }
  return desc;
}
