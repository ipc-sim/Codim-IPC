#pragma once

#include "./UTILS.hpp"
#include <Cabana_Core.hpp>

namespace JGSL {
namespace storage {
  template <int Index, typename AoSoA>
  struct SliceHolderBase {
    using Slice = decltype(Cabana::slice<Index>(std::declval<AoSoA>()));

    Slice slice;

    SliceHolderBase(const AoSoA &data) : slice(Cabana::slice<Index>(data)) {}
  };

  template <int Index, typename AoSoA, typename... Types>
  struct SliceHolderImpl {
    SliceHolderImpl(const AoSoA &data) {}
  };

  template <int Index, typename AoSoA, typename T, typename... Types>
  struct SliceHolderImpl<Index, AoSoA, T, Types...>
      : public SliceHolderBase<Index, AoSoA>, public SliceHolderImpl<Index + 1, AoSoA, Types...> {
    SliceHolderImpl(const AoSoA &data)
        : SliceHolderBase<Index, AoSoA>(data), SliceHolderImpl<Index + 1, AoSoA, Types...>(data) {}
  };

  template <typename AoSoA, typename... Types>
  struct SliceHolder : public SliceHolderImpl<0, AoSoA, Types...> {
    template <int Index>
    using SliceAt = decltype(Cabana::slice<Index>(std::declval<AoSoA>()));

    SliceHolder(const AoSoA &data) : SliceHolderImpl<0, AoSoA, Types...>(data) {}

    template <int Index>
    KOKKOS_INLINE_FUNCTION const SliceAt<Index> &get() const {
      return SliceHolderBase<Index, AoSoA>::slice;
    }
  };

  template <int StorageIndex, int FieldIndex, typename AoSoA>
  struct JoinedSliceHolderBaseImpl
      : public JoinedSliceHolderBaseImpl<StorageIndex, FieldIndex - 1, AoSoA> {
    using Slice = decltype(Cabana::slice<FieldIndex>(std::declval<AoSoA>()));

    Slice slice;

    JoinedSliceHolderBaseImpl(const AoSoA &s)
        : slice(Cabana::slice<FieldIndex>(s)),
          JoinedSliceHolderBaseImpl<StorageIndex, FieldIndex - 1, AoSoA>(s) {}
  };

  template <int StorageIndex, typename AoSoA>
  struct JoinedSliceHolderBaseImpl<StorageIndex, -1, AoSoA> {
    JoinedSliceHolderBaseImpl(const AoSoA &s) {}
  };

  template <int StorageIndex, template <class> typename Extractor, typename Joined, typename S>
  struct JoinedSliceHolderBase
      : public JoinedSliceHolderBaseImpl<StorageIndex, S::N - 1, typename Extractor<S>::AoSoA> {
    using AoSoA = typename Extractor<S>::AoSoA;

    JoinedSliceHolderBase(const Joined &j)
        : JoinedSliceHolderBaseImpl<StorageIndex, S::N - 1, AoSoA>(
              Extractor<S>::get(j.template get<StorageIndex>())) {}
  };

  template <int StorageIndex,
            template <class>
            typename Extractor,
            typename Joined,
            typename... Storages>
  struct JoinedSliceHolderImpl {
    JoinedSliceHolderImpl(const Joined &j) {}
  };

  template <int StorageIndex,
            template <class>
            typename Extractor,
            typename Joined,
            typename S,
            typename... Storages>
  struct JoinedSliceHolderImpl<StorageIndex, Extractor, Joined, S, Storages...>
      : public JoinedSliceHolderBase<StorageIndex, Extractor, Joined, S>,
        public JoinedSliceHolderImpl<StorageIndex + 1, Extractor, Joined, Storages...> {
    JoinedSliceHolderImpl(const Joined &j)
        : JoinedSliceHolderBase<StorageIndex, Extractor, Joined, S>(j),
          JoinedSliceHolderImpl<StorageIndex + 1, Extractor, Joined, Storages...>(j) {}
  };

  template <template <class> typename Extractor, typename Joined, typename... Storages>
  struct JoinedSliceHolder : public JoinedSliceHolderImpl<0, Extractor, Joined, Storages...> {
    JoinedSliceHolder(const Joined &j)
        : JoinedSliceHolderImpl<0, Extractor, Joined, Storages...>(j) {}

    template <int StorageIndex>
    using StorageAt = typename ExtractTypeAt<StorageIndex, Storages...>::Type;

    template <int StorageIndex>
    using AoSoAAt = typename Extractor<StorageAt<StorageIndex>>::AoSoA;

    template <int StorageIndex, int FieldIndex>
    using SliceAt =
        typename JoinedSliceHolderBaseImpl<StorageIndex, FieldIndex, AoSoAAt<StorageIndex>>::Slice;

    template <int StorageIndex, int FieldIndex>
    KOKKOS_INLINE_FUNCTION const SliceAt<StorageIndex, FieldIndex> &get() const {
      return JoinedSliceHolderBaseImpl<StorageIndex, FieldIndex, AoSoAAt<StorageIndex>>::slice;
    }
  };

  template <typename S>
  struct HostAoSoAExtractor {
    using AoSoA = typename S::HostAoSoA;

    static inline const AoSoA &get(const S &s) {
      return s.host_data;
    }
  };

  template <typename S>
  struct DeviceAoSoAExtractor {
    using AoSoA = typename S::DeviceAoSoA;

    static inline const AoSoA &get(const S &s) {
      return s.device_data;
    }
  };
} // namespace storage
}