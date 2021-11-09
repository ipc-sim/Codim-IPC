#pragma once

#include "./RANGE.hpp"
#include "./STORAGE.hpp"

namespace JGSL {
namespace storage {
  template <class Config, typename... Types>
  struct FullStorage : public Storage<Config, Types...> {
    using Super = Storage<Config, Types...>;

    using HostExecutionSpace = typename Super::HostExecutionSpace;

    using HostAoSoA = typename Super::HostAoSoA;

    using DeviceAoSoA = typename Super::DeviceAoSoA;

    using HostHandle = typename Super::HostHandle;

    using DeviceHandle = typename Super::DeviceHandle;

    template <int Index>
    using TypeAt = typename Super::template TypeAt<Index>;

    FullStorage() : Super() {
      this->ranges_map.add(0, 0, 0);
    }

    FullStorage(std::size_t capacity) : Super(capacity) {
      this->ranges_map.add(0, 0, 0);
    }

    template <int Index>
    inline void fill(const TypeAt<Index> &c) {
      Super::template fill<Index>(c);
    }

    Range fill(std::size_t amount, const Types &... cs) {
      std::size_t start = this->stored_length;

      this->stored_length += amount;
      if (this->stored_length > this->host_data.size()) {
        this->host_data.resize(this->stored_length);
      }

      auto tuple = ToCabanaTuple<Types...>::to_cabana(cs...);
      auto kernel = KOKKOS_LAMBDA(std::size_t i) {
        this->host_data.setTuple(i, tuple);
      };
      Kokkos::RangePolicy<HostExecutionSpace> linear_policy(start, this->stored_length);
      Kokkos::parallel_for(linear_policy, kernel, "fill");

      update_ranges_map();
      return Range(start, amount);
    }

    template <typename Iter, typename F>
    Range fill_iter(Iter iter, F callback) {
      std::size_t start = this->stored_length;

      std::size_t amount = iter.size();
      this->stored_length += amount;
      if (this->stored_length > this->host_data.size()) {
        this->host_data.resize(this->stored_length);
      }

      SliceHolder<HostAoSoA, Types...> slice_holder(this->host_data);
      for (auto data : iter) {
        LinearHandle<HostAoSoA, Types...> handle(slice_holder, start++);
        callback(data, handle);
      }

      update_ranges_map();
      return Range(start, amount);
    }

  private:
    void update_ranges_map() {
      this->ranges_map.globals[0].amount = this->stored_length;
    }
  };
} // namespace storage
}