#pragma once

#include "./RANGE.hpp"
#include "./TYPE_TRANSFORM.hpp"

namespace JGSL {
namespace storage {
  template <int Index, typename T, typename... Types>
  struct ExtractTypeAt {
    using Type = typename ExtractTypeAt<Index - 1, Types...>::Type;
  };

  template <typename T, typename... Types>
  struct ExtractTypeAt<0, T, Types...> {
    using Type = T;
  };

  template <int Index, class SliceHolder, typename... Types>
  struct UpdatorImpl {
    static KOKKOS_INLINE_FUNCTION void
    update(const SliceHolder &slice_holder, const std::size_t i, const Types &... components) {}
  };

  template <int Index, class SliceHolder, typename T, typename... Types>
  struct UpdatorImpl<Index, SliceHolder, T, Types...> {
    static KOKKOS_INLINE_FUNCTION void
    update(const SliceHolder &slice_holder, const std::size_t i, const T &c, const Types &... components) {
      TypeTransform<T>::set(slice_holder.template get<Index>(), i, c);
      UpdatorImpl<Index + 1, SliceHolder, Types...>::update(slice_holder, i, components...);
    }
  };

  template <class SliceHolder, typename... Types>
  struct Updator : public UpdatorImpl<0, SliceHolder, Types...> {};

  // TODO: Change the name of these
  template <int Index, typename... Types>
  struct ToCabanaTupleImpl {
    template <typename Tuple>
    static KOKKOS_INLINE_FUNCTION void to_cabana_tuple(Tuple &target, const Types &... cs) {}
  };

  template <int Index, typename T, typename... Types>
  struct ToCabanaTupleImpl<Index, T, Types...> {
    template <typename Tuple>
    static KOKKOS_INLINE_FUNCTION void
    to_cabana_tuple(Tuple &target, const T &c, const Types &... cs) {
      TypeTransform<T>::template set<Index>(target, c);
      ToCabanaTupleImpl<Index + 1, Types...>::to_cabana_tuple(target, cs...);
    }
  };

  template <typename... Types>
  struct ToCabanaTuple {
    using MemberTypes = Cabana::MemberTypes<typename TypeTransform<Types>::To...>;

    using Tuple = Cabana::Tuple<MemberTypes>;

    static KOKKOS_INLINE_FUNCTION Tuple to_cabana(const Types &... cs) {
      Tuple target;
      ToCabanaTupleImpl<0, Types...>::to_cabana_tuple(target, cs...);
      return target;
    }
  };

  template <std::size_t Index, typename S>
  struct JoinedStorageBase {
    const S &storage;

    JoinedStorageBase(const S &storage) : storage(storage) {}
  };

  template <std::size_t Index, typename... Storages>
  struct JoinedStorageImpl {
    JoinedStorageImpl(const Storages &... ss) {}

    Ranges ranges() const {
      return Ranges(true); // Infinity
    }
  };

  template <std::size_t Index, typename S, typename... Storages>
  struct JoinedStorageImpl<Index, S, Storages...>
      : public JoinedStorageBase<Index, S>, public JoinedStorageImpl<Index + 1, Storages...> {
    JoinedStorageImpl(const S &s, const Storages &... ss)
        : JoinedStorageBase<Index, S>(s), JoinedStorageImpl<Index + 1, Storages...>(ss...) {}

    Ranges ranges() const {
      auto hd = JoinedStorageBase<Index, S>::storage.ranges().globals;
      auto rs = JoinedStorageImpl<Index + 1, Storages...>::ranges();
      return hd.intersect(rs);
    }
  };

  template <typename... Storages>
  struct JoinedStorageGroup : public JoinedStorageImpl<0, Storages...> {
    template <int Index>
    using StorageAt = typename ExtractTypeAt<Index, Storages...>::Type;

    static const std::size_t N = sizeof...(Storages);

    JoinedStorageGroup(const Storages &... ss) : JoinedStorageImpl<0, Storages...>(ss...) {}

    Ranges ranges() const {
      return JoinedStorageImpl<0, Storages...>::ranges();
    }

    template <int Index>
    const StorageAt<Index> &get() const {
      return JoinedStorageBase<Index, StorageAt<Index>>::storage;
    }
  };

  template <std::size_t Index, typename Joined, typename S>
  struct JoinedOffsetBase {
    std::size_t local_offset;

    JoinedOffsetBase(const Range &global, const Joined &j)
        : local_offset(j.template get<Index>().ranges().to_local(global.start)) {}
  };

  template <std::size_t Index, typename Joined, typename... Storages>
  struct JoinedOffsetImpl {
    JoinedOffsetImpl(const Range &global, const Joined &j) {}
  };

  template <std::size_t Index, typename Joined, typename S, typename... Storages>
  struct JoinedOffsetImpl<Index, Joined, S, Storages...>
      : public JoinedOffsetBase<Index, Joined, S>,
        public JoinedOffsetImpl<Index + 1, Joined, Storages...> {
    JoinedOffsetImpl(const Range &global, const Joined &j)
        : JoinedOffsetBase<Index, Joined, S>(global, j),
          JoinedOffsetImpl<Index + 1, Joined, Storages...>(global, j) {}
  };

  template <typename Joined, typename... Storages>
  struct JoinedOffset : public JoinedOffsetImpl<0, Joined, Storages...> {
    template <int Index>
    using StorageAt = typename ExtractTypeAt<Index, Storages...>::Type;

    JoinedOffset(const Range &global, const Joined &j)
        : JoinedOffsetImpl<0, Joined, Storages...>(global, j) {}

    template <int Index>
    KOKKOS_INLINE_FUNCTION std::size_t local_offset() const {
      return JoinedOffsetBase<Index, Joined, StorageAt<Index>>::local_offset;
    }
  };
} // namespace storage
}