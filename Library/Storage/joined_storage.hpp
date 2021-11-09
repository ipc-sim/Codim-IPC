#pragma once

#include "storage_utils.hpp"

namespace JGSL {
  template <std::size_t Index, typename S>
  class JoinedStorageBase {
  public:
    using RefTuple = typename S::RefTuple;

    using ConstRefTuple = typename S::ConstRefTuple;

    S &storage;

    JoinedStorageBase(S &storage) : storage(storage) {}
  };

  template <std::size_t Index, typename... Storages>
  class JoinedStorageGroup {
  public:
    using RefTuple = std::tuple<>;

    using ConstRefTuple = std::tuple<>;

    bool contains(std::size_t i) const {
      return true;
    }

    std::tuple<> Get_Unchecked(std::size_t i) {
      return std::tie();
    }

    std::tuple<> Get_Unchecked_Const(std::size_t i) const {
      return std::tie();
    }
  };

  template <std::size_t Index, typename S, typename... Storages>
  class JoinedStorageGroup<Index, S, Storages...>
      : public JoinedStorageBase<Index, S>,
        public JoinedStorageGroup<Index + 1, Storages...> {
  public:
    using RefTuple = decltype(std::tuple_cat(
        std::declval<typename JoinedStorageBase<Index, S>::RefTuple>(),
        std::declval<
            typename JoinedStorageGroup<Index + 1, Storages...>::RefTuple>()));

    using ConstRefTuple = decltype(std::tuple_cat(
        std::declval<typename JoinedStorageBase<Index, S>::ConstRefTuple>(),
        std::declval<typename JoinedStorageGroup<Index + 1, Storages...>::
                         ConstRefTuple>()));

    JoinedStorageGroup(S &s, Storages &... ss)
        : JoinedStorageBase<Index, S>(s),
          JoinedStorageGroup<Index + 1, Storages...>(ss...) {}

    bool contains(std::size_t i) const {
      if (JoinedStorageBase<Index, S>::storage.contains(i))
        return JoinedStorageGroup<Index + 1, Storages...>::contains(i);
      return false;
    }

    S &first_storage() const {
      return JoinedStorageBase<Index, S>::storage;
    }

    RefTuple Get_Unchecked(std::size_t i) {
      return std::tuple_cat(
          JoinedStorageBase<Index, S>::storage.Get_Unchecked(i),
          JoinedStorageGroup<Index + 1, Storages...>::Get_Unchecked(i));
    }

    ConstRefTuple Get_Unchecked_Const(std::size_t i) const {
      return std::tuple_cat(
          JoinedStorageBase<Index, S>::storage.Get_Unchecked_Const(i),
          JoinedStorageGroup<Index + 1, Storages...>::Get_Unchecked_Const(i));
    }
  };

  template <class ExSpace, class... Storages>
  class JoinedStorage {
  public:
    using StorageGroup = JoinedStorageGroup<0, Storages...>;

    using RefTuple = typename StorageGroup::RefTuple;

    using ConstRefTuple = typename StorageGroup::ConstRefTuple;

    JoinedStorage(Storages &... ss) : base(ss...) {}

    void Each(std::function<void(const int, RefTuple)> f) {
      for (int i = 0; i < base.first_storage().size; i++) {
        std::size_t global_index = base.first_storage().global_indices[i];
        if (base.contains(global_index)) {
          auto data = base.Get_Unchecked(global_index);
          f(global_index, data);
        }
      }
    }

    void Par_Each(std::function<void(const int, RefTuple)> f) {
      auto n = base.first_storage().size;
      auto kernel = KOKKOS_LAMBDA(const int i) {
        std::size_t global_index = base.first_storage().global_indices[i];
        if (base.contains(global_index)) {
          auto data = base.Get_Unchecked(global_index);
          f(global_index, data);
        }
      };
      Kokkos::RangePolicy<ExSpace> linear_policy(0, n);
      Kokkos::parallel_for(linear_policy, kernel, "joined_storage_each");
    }

  private:
    JoinedStorageGroup<0, Storages...> base;
  };

template <std::size_t OFFSET, typename EX_SPACE>
struct FIELDS_WITH_OFFSET<OFFSET, JoinedStorage<EX_SPACE>> {};

template <std::size_t OFFSET, typename EX_SPACE, typename S, typename... STORAGES>
struct FIELDS_WITH_OFFSET<OFFSET, JoinedStorage<EX_SPACE, S, STORAGES...>>
: public FIELDS_WITH_OFFSET<OFFSET, S>,
public FIELDS_WITH_OFFSET<OFFSET + S::N, JoinedStorage<EX_SPACE, STORAGES...>> {};

} // namespace storage