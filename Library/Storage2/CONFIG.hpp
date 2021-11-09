#pragma once

#include <Kokkos_Core.hpp>

namespace JGSL {
namespace storage {
  struct DefaultConfig {
    using DeviceExecutionSpace = Kokkos::Serial;

    using HostExecutionSpace = Kokkos::Serial;

    using DeviceMemorySpace = Kokkos::HostSpace;
  };

  struct Config {

    // Device config
#ifdef STORAGE_ENABLED_CUDA
    using DeviceExecutionSpace = Kokkos::Cuda;
    using DeviceMemorySpace = Kokkos::CudaSpace;
#else
    using DeviceMemorySpace = Kokkos::HostSpace;
#ifdef STORAGE_ENABLED_OPENMP
    using DeviceExecutionSpace = Kokkos::OpenMP;
#else
    using DeviceExecutionSpace = Kokkos::Serial;
#endif // STORAGE_ENABLED_OPENMP
#endif // STORAGE_ENABLED_CUDA

    // Host config
#ifdef STORAGE_ENABLED_OPENMP
    using HostExecutionSpace = Kokkos::OpenMP;
#else
    using HostExecutionSpace = Kokkos::Serial;
#endif // STORAGE_ENABLED_OPENMP
  };
} // namespace storage
} // namespace JGSL