#pragma once

#include "./RANGE.hpp"
#include "./UTILS.hpp"
#include "./KERNEL_FUNCTOR.hpp"

namespace JGSL {
namespace storage {
  template <typename Config, typename... Storages>
  struct JoinedStorage {
    using DeviceExecutionSpace = typename Config::DeviceExecutionSpace;

    using Base = JoinedStorageGroup<Storages...>;

    using HostSliceHolder = JoinedSliceHolder<HostAoSoAExtractor, Base, Storages...>;

    using DeviceSliceHolder = JoinedSliceHolder<DeviceAoSoAExtractor, Base, Storages...>;

    using Offset = JoinedOffset<Base, Storages...>;

    using HostHandle = JoinedLinearHandle<HostAoSoAExtractor, Storages...>;

    using DeviceHandle = JoinedLinearHandle<DeviceAoSoAExtractor, Storages...>;

    static const std::size_t N = sizeof...(Storages);

    Base base;

    JoinedStorage(const Storages &... storages) : base(storages...) {}

    template <typename F>
    void each(F kernel) const {
      HostSliceHolder host_slice_holder(base);
      Ranges global_ranges = base.ranges();
      for (const Range &range : global_ranges) {
        Offset offset(range, base);
        for (std::size_t i = 0; i < range.amount; i++) {
          HostHandle handle(host_slice_holder, offset, i);
          kernel(handle);
        }
      }
    }

    template <typename F>
    void par_each(F kernel) const {
      using KernelFunctor = JoinedLinearKernel<DeviceAoSoAExtractor, F, Storages...>;
      Ranges global_ranges = base.ranges();
      for (const Range &range : global_ranges) {
        KernelFunctor kernel_functor(base, range, kernel);
        Kokkos::RangePolicy<DeviceExecutionSpace> linear_policy(0, range.amount);
        Kokkos::parallel_for(linear_policy, kernel_functor, "par_each");
      }
    }
  };
} // namespace storage
}