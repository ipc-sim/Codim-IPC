#pragma once

#ifdef _WIN64
#define STORAGE_FORCE_INLINE __forceinline
#else
#define STORAGE_FORCE_INLINE inline __attribute__((always_inline))
#endif

#include "storage.hpp"