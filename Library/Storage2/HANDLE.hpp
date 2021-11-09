#pragma once

#include "./SLICE_HOLDER.hpp"
#include "./TYPE_TRANSFORM.hpp"
#include "./UTILS.hpp"

namespace JGSL {
namespace storage {
  template <class AoSoA, typename... Types>
  struct LinearHandle {
    template <int Index>
    using TypeAt = typename ExtractTypeAt<Index, Types...>::Type;

    template <int Index>
    using TransformedTypeAt = typename TypeTransform<TypeAt<Index>>::To;

    const SliceHolder<AoSoA, Types...> &slice_holder;

    const std::size_t i;

    KOKKOS_FUNCTION LinearHandle(const SliceHolder<AoSoA, Types...> &slice_holder, const std::size_t i)
        : slice_holder(slice_holder), i(i) {}

    KOKKOS_INLINE_FUNCTION std::size_t index() const {
      return i;
    }

    template <int Index>
    KOKKOS_INLINE_FUNCTION TypeAt<Index> get() const {
      return TypeTransform<TypeAt<Index>>::get(slice_holder.template get<Index>(), i);
    }

    template <int Index>
    KOKKOS_INLINE_FUNCTION void set(const TypeAt<Index> &c) const {
      TypeTransform<TypeAt<Index>>::set(slice_holder.template get<Index>(), i, c);
    }
  };

  template <template <class> typename Extractor, typename... Storages>
  struct JoinedLinearHandle {
    using Base = JoinedStorageGroup<Storages...>;

    using SliceHolder = JoinedSliceHolder<Extractor, Base, Storages...>;

    using Offset = JoinedOffset<Base, Storages...>;

    template <int StorageIndex>
    using StorageAt = typename ExtractTypeAt<StorageIndex, Storages...>::Type;

    template <int StorageIndex, int FieldIndex>
    using TypeAt = typename StorageAt<StorageIndex>::template TypeAt<FieldIndex>;

    template <int StorageIndex, int FieldIndex>
    using SliceAt = typename SliceHolder::template SliceAt<StorageIndex, FieldIndex>;

    const SliceHolder &slice_holder;

    const Offset &offset;

    const std::size_t i;

    KOKKOS_FUNCTION
    JoinedLinearHandle(const SliceHolder &slice_holder, const Offset &offset, const std::size_t i)
        : slice_holder(slice_holder), offset(offset), i(i) {}

    template <int StorageIndex>
    KOKKOS_INLINE_FUNCTION std::size_t index() const {
      return offset.template local_offset<StorageIndex>() + i;
    }

    template <int StorageIndex, int FieldIndex>
    KOKKOS_INLINE_FUNCTION TypeAt<StorageIndex, FieldIndex> get() const {
      auto slice = slice_holder.template get<StorageIndex, FieldIndex>();
      auto off = offset.template local_offset<StorageIndex>();
      return TypeTransform<TypeAt<StorageIndex, FieldIndex>>::get(slice, off + i);
    }

    template <int StorageIndex, int FieldIndex>
    KOKKOS_INLINE_FUNCTION void set(const TypeAt<StorageIndex, FieldIndex> &c) const {
      auto slice = slice_holder.template get<StorageIndex, FieldIndex>();
      auto off = offset.template local_offset<StorageIndex>();
      TypeTransform<TypeAt<StorageIndex, FieldIndex>>::set(slice, off + i, c);
    }
  };
} // namespace storage
}