#pragma once

#include <iostream>

#include <Cabana_Core.hpp>
#include <Math/VECTOR.h>

namespace JGSL {
namespace storage {
  template <typename T>
  struct TypeTransform {
    using From = T;

    using To = T;

    template <typename Slice>
    static KOKKOS_INLINE_FUNCTION From get(const Slice &slice, const std::size_t i) {
      return slice(i);
    }

    template <int Index, typename Tuple>
    static KOKKOS_INLINE_FUNCTION From get(Tuple &tuple) {
      return Cabana::get<Index>(tuple);
    }

    template <typename Slice>
    static KOKKOS_INLINE_FUNCTION void set(const Slice &slice, const std::size_t i, const From &c) {
      slice(i) = c;
    }

    template <int Index, typename Tuple>
    static KOKKOS_INLINE_FUNCTION void set(Tuple &tuple, const From &c) {
      Cabana::get<Index>(tuple) = c;
    }
  };

  // template <class T, int D>
  // struct TypeTransform<math::Vector<T, D>> {
  //   using From = math::Vector<T, D>;

  //   using To = T[D];

  //   template <typename Slice>
  //   static KOKKOS_INLINE_FUNCTION From get(const Slice &slice, const std::size_t i) {
  //     From v;
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       v(j) = slice(i, j);
  //     }
  //     return v;
  //   }

  //   template <int Index, typename Tuple>
  //   static KOKKOS_INLINE_FUNCTION From get(Tuple &tuple) {
  //     From v;
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       v(j) = Cabana::get<Index>(tuple, j);
  //     }
  //     return v;
  //   }

  //   template <typename Slice>
  //   static KOKKOS_INLINE_FUNCTION void set(const Slice &slice, const std::size_t i, const From &c) {
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       slice(i, j) = c(j);
  //     }
  //   }

  //   template <int Index, typename Tuple>
  //   static KOKKOS_INLINE_FUNCTION void set(Tuple &tuple, const From &c) {
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       Cabana::get<Index>(tuple, j) = c(j);
  //     }
  //   }
  // };

  // template <class T, int D>
  // struct TypeTransform<math::Matrix<T, D>> {
  //   using From = math::Matrix<T, D>;

  //   using To = T[D][D];

  //   template <typename Slice>
  //   static KOKKOS_INLINE_FUNCTION From get(const Slice &slice, const std::size_t i) {
  //     From v;
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       #pragma unroll (D)
  //       for (int k = 0; k < D; k++) {
  //         v(j, k) = slice(i, j, k);
  //       }
  //     }
  //     return v;
  //   }

  //   template <int Index, typename Tuple>
  //   static KOKKOS_INLINE_FUNCTION From get(Tuple &tuple) {
  //     From v;
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       #pragma unroll (D)
  //       for (int k = 0; k < D; k++) {
  //         v(j, k) = Cabana::get<Index>(tuple, j, k);
  //       }
  //     }
  //   }

  //   template <typename Slice>
  //   static KOKKOS_INLINE_FUNCTION void set(const Slice &slice, const std::size_t i, const From &c) {
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       #pragma unroll (D)
  //       for (int k = 0; k < D; k++) {
  //         slice(i, j, k) = c(j, k);
  //       }
  //     }
  //   }

  //   template <int Index, typename Tuple>
  //   static KOKKOS_INLINE_FUNCTION void set(Tuple &tuple, const From &c) {
  //     #pragma unroll (D)
  //     for (int j = 0; j < D; j++) {
  //       #pragma unroll (D)
  //       for (int k = 0; k < D; k++) {
  //         Cabana::get<Index>(tuple, j, k) = c(j, k);
  //       }
  //     }
  //   }
  // };
} // namespace storage
}