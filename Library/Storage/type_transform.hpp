#pragma once

#include <Math/VECTOR.h>
#include <Storage/prelude.hpp>

namespace JGSL {

  template <typename T>
  struct TypeTransform {
    using Type = T;

    template <std::size_t Index, typename AosoaType>
    static STORAGE_FORCE_INLINE T &get(const AosoaType &data, std::size_t i) {
      return Cabana::slice<Index, AosoaType>(data)(i);
    }

    template <std::size_t Index, typename AosoaType>
    static STORAGE_FORCE_INLINE void
    set(AosoaType &data, std::size_t i, T &value) {
      Cabana::slice<Index, AosoaType>(data)(i) = value;
    }
  };

  template <typename T, int dim>
  struct TypeTransform<VECTOR<T, dim>> {
    using Type = T[4];

    using From = VECTOR<T, dim>;

    template <std::size_t Index, typename AosoaType>
    static STORAGE_FORCE_INLINE From &get(const AosoaType &data,
                                          std::size_t i) {
      auto slice = Cabana::slice<Index, AosoaType>(data);
      auto offset = i & (slice.extent(1) - 1); // i % slice.extent(1)
      T *ptr = &slice(i - offset, offset);
      From *vptr = (From *)ptr;
      return *vptr;
    }

    template <std::size_t Index, typename AosoaType>
    static STORAGE_FORCE_INLINE void
    set(AosoaType &data, std::size_t i, const From &value) {
      auto slice = Cabana::slice<Index, AosoaType>(data);
      auto offset = i & (slice.extent(1) - 1);
      T *ptr = &slice(i - offset, offset);
      From *vptr = (From *)ptr;
      *vptr = value;
    }
  };

  template <typename T, int dim>
  struct TypeTransform<MATRIX<T, dim>> {
    using Type = T[dim * 4];

    using From = MATRIX<T, dim>;

    template <std::size_t Index, typename AosoaType>
    static STORAGE_FORCE_INLINE From &get(const AosoaType &data,
                                          std::size_t i) {
      auto slice = Cabana::slice<Index, AosoaType>(data);
      auto offset = i & (slice.extent(1) - 1);
      T *ptr = &slice(i - offset, offset * dim);
      From *vptr = (From *)ptr;
      return *vptr;
    }

    template <std::size_t Index, typename AosoaType>
    static STORAGE_FORCE_INLINE void
    set(AosoaType &data, std::size_t i, const From &value) {
      auto slice = Cabana::slice<Index, AosoaType>(data);
      auto offset = i & (slice.extent(1) - 1);
      T *ptr = &slice(i - offset, offset * dim);
      From *vptr = (From *)ptr;
      *vptr = value;
    }
  };
} // namespace storage