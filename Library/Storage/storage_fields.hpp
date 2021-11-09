#pragma once

#include "./joined_storage.hpp"

namespace JGSL {
  template <std::size_t Offset, typename Storage>
  struct FieldsWithOffset {
    enum INDICES {};
  };

  template <typename Storage>
  struct Fields : public FieldsWithOffset<0, Storage> {};

  template <std::size_t Offset, typename ExSpace>
  struct FieldsWithOffset<Offset, JoinedStorage<ExSpace>> {};

  template <std::size_t Offset,
            typename ExSpace,
            typename S,
            typename... STORAGES>
  struct FieldsWithOffset<Offset, JoinedStorage<ExSpace, S, STORAGES...>>
      : public FieldsWithOffset<Offset, S>,
        public FieldsWithOffset<Offset + S::N,
                                JoinedStorage<ExSpace, STORAGES...>> {};
} // namespace storage