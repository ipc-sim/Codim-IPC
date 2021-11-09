#pragma once

#include "./HANDLE.hpp"
#include "./UTILS.hpp"
#include "./RANGE.hpp"

namespace JGSL {
namespace storage {
  template <class AoSoA, class F, typename... Types>
  struct LinearKernel {
    using Handle = LinearHandle<AoSoA, Types...>;

    SliceHolder<AoSoA, Types...> slice_holder;

    F kernel;

    LinearKernel(const AoSoA &data, F kernel) : slice_holder(data), kernel(kernel) {}

    KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
      Handle handle(slice_holder, i);
      kernel(handle);
    }
  };

  template <template <class> typename Extractor, class F, typename... Storages>
  struct JoinedLinearKernel {
    using Base = JoinedStorageGroup<Storages...>;

    using Handle = JoinedLinearHandle<Extractor, Storages...>;

    using SliceHolder = JoinedSliceHolder<Extractor, Base, Storages...>;

    using Offset = JoinedOffset<Base, Storages...>;

    SliceHolder slice_holder;

    Offset offset;

    F kernel;

    JoinedLinearKernel(const Base &base, const Range &range, F kernel)
        : slice_holder(base), offset(range, base), kernel(kernel) {}

    KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
      Handle handle(slice_holder, offset, i);
      kernel(handle);
    }
  };
} // namespace storage
}