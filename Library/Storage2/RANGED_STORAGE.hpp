#pragma once

#include "./RANGE.hpp"
#include "./STORAGE.hpp"

namespace JGSL {
namespace storage {
  template <class Config, typename... Types>
  struct RangedStorage : public Storage<Config, Types...> {
    using Super = Storage<Config, Types...>;

    using HostExecutionSpace = typename Super::HostExecutionSpace;

    using HostAoSoA = typename Super::HostAoSoA;

    using DeviceAoSoA = typename Super::DeviceAoSoA;

    using HostHandle = typename Super::HostHandle;

    using DeviceHandle = typename Super::DeviceHandle;

    template <int Index>
    using TypeAt = typename Super::template TypeAt<Index>;

    RangedStorage() : Super() {}

    RangedStorage(std::size_t capacity) : Super(capacity) {}

    template <int Index>
    inline void fill(const TypeAt<Index> &c) {
      Super::template fill<Index>(c);
    }

    void fill(const Range &range, const Types &... cs) {
      std::size_t start = this->stored_length;
      this->ranges_map.add(start, range);

      this->stored_length += range.amount;
      if (this->stored_length > this->host_data.size()) {
        this->host_data.resize(this->stored_length);
      }

      auto tuple = ToCabanaTuple<Types...>::to_cabana(cs...);
      auto kernel = KOKKOS_LAMBDA(std::size_t i) {
        this->host_data.setTuple(i, tuple);
      };
      Kokkos::RangePolicy<HostExecutionSpace> linear_policy(start, this->stored_length);
      Kokkos::parallel_for(linear_policy, kernel, "fill");
    }
  };
} // namespace storage
}