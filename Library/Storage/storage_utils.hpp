#pragma once

#include <tuple>

#include <Cabana_Core.hpp>

#include "./type_transform.hpp"

namespace JGSL {

  template <typename... Types>
  using CabanaDataTypes = Cabana::MemberTypes<Types...>;

  template <typename... Types>
  using CabanaTuple = Cabana::Tuple<CabanaDataTypes<Types...>>;

  template <typename... Types>
  using Tuple = std::tuple<Types...>;

  template <std::size_t Index, typename... Types>
  struct RefTupleExtractor {
    template <typename AosoaType>
    static STORAGE_FORCE_INLINE Tuple<> get(AosoaType &data, std::size_t i) {
      return std::tie();
    }
  };

  template <std::size_t Index, typename T, typename... Types>
  struct RefTupleExtractor<Index, T, Types...> {
    template <typename AosoaType>
    static STORAGE_FORCE_INLINE Tuple<T &, Types &...> get(AosoaType &data,
                                                           std::size_t i) {
      T &hd = TypeTransform<T>::template get<Index, AosoaType>(data, i);
      auto rs = RefTupleExtractor<Index + 1, Types...>::get(data, i);
      return std::tuple_cat(std::tie(hd), rs);
    }
  };

  template <std::size_t Index, typename... Types>
  struct RefTupleUpdator {
    template <typename... FullTypes>
    static STORAGE_FORCE_INLINE void set(Tuple<FullTypes &...> &t,
                                         Types... ts) {}
  };

  template <std::size_t Index, typename T, typename... Types>
  struct RefTupleUpdator<Index, T, Types...> {
    template <typename... FullTypes>
    static STORAGE_FORCE_INLINE void
    set(Tuple<FullTypes &...> &t, T c, Types... ts) {
      std::get<Index>(t) = c;
      RefTupleUpdator<Index + 1, Types...>::set(t, ts...);
    }
  };

  template <std::size_t Index, typename... Types>
  struct DataUpdator {
    template <typename AosoaType>
    static STORAGE_FORCE_INLINE void
    set(AosoaType &data, std::size_t i, Types... ts) {}
  };

  template <std::size_t Index, typename T, typename... Types>
  struct DataUpdator<Index, T, Types...> {
    template <typename AosoaType>
    static STORAGE_FORCE_INLINE void
    set(AosoaType &data, std::size_t i, T t, Types... ts) {
      TypeTransform<T>::template set<Index, AosoaType>(data, i, t);
      DataUpdator<Index + 1, Types...>::set(data, i, ts...);
    }
  };

  template <std::size_t Index, typename T, typename... Types>
  struct ExtractTypeAt {
    using Type = typename ExtractTypeAt<Index - 1, Types...>::Type;
  };

  template <typename T, typename... Types>
  struct ExtractTypeAt<0, T, Types...> {
    using Type = T;
  };

  template <std::size_t OFFSET, typename STORAGE>
  struct FIELDS_WITH_OFFSET {
      enum INDICES {};
  };

  template <typename STORAGE>
  struct FIELDS : public FIELDS_WITH_OFFSET<0, STORAGE> {};

} // namespace storage