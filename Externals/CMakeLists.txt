find_package(Git)
# Enable DownloadProject CMake module
include(DownloadProject)
include(ExternalProject)

find_package(CUDA QUIET)
if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${GENCODE};-lineinfo;--expt-extended-lambda;--expt-relaxed-constexpr;--default-stream per-thread PARENT_SCOPE)
  set(CUDA_SEPARABLE_COMPILATION ON PARENT_SCOPE)
endif()

download_project(
        PROJ amgcl
        GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
        UPDATE_DISCONNECTED 1
)
add_library(amgcl INTERFACE)
target_include_directories(amgcl SYSTEM INTERFACE ${amgcl_SOURCE_DIR})
if (APPLE)
  find_package(OPENMP REQUIRED)
else()
  find_package(OpenMP REQUIRED)
endif()
if (OPENMP_FOUND)
  target_compile_options(amgcl INTERFACE ${OpenMP_CXX_FLAGS})
  target_link_libraries(amgcl INTERFACE
          $<$<CXX_COMPILER_ID:GNU>:${OpenMP_CXX_FLAGS}>
          $<$<CXX_COMPILER_ID:Clang>:${OpenMP_CXX_FLAGS}>
          $<$<CXX_COMPILER_ID:Intel>:${OpenMP_CXX_FLAGS}>
          )
endif ()

if (ENABLE_AMGCL_CUDA)
  download_project(
          PROJ VexCL
          GIT_REPOSITORY https://github.com/ddemidov/vexcl
          UPDATE_DISCONNECTED 1
  )
  add_subdirectory(${VexCL_SOURCE_DIR} ${VexCL_BINARY_DIR})
endif()

# Download and add Kokkos
download_project(
  PROJ Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG 3.1.01
  UPDATE_DISCONNECTED 1
)
set(Kokkos_ENABLE_SERIAL On CACHE INTERNAL "")
set(Kokkos_ENABLE_OPENMP On CACHE INTERNAL "")
add_subdirectory(${Kokkos_SOURCE_DIR} ${Kokkos_BINARY_DIR})

# Set the Kokkos_DEVICES variable for Cabana
set(Kokkos_DEVICES SERIAL OPENMP)

# Download and add Cabana
download_project(
  PROJ Cabana
  GIT_REPOSITORY https://github.com/Liby99/Cabana # Forked to get rid of find_package(Kokkos)
  UPDATE_DISCONNECTED 1
)
add_subdirectory(${Cabana_SOURCE_DIR} ${Cabana_BINARY_DIR})

# Download and add kokkos-kernels
download_project(
  PROJ Kokkos_kernels
  GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels
  GIT_TAG 3.1.01
  UPDATE_DISCONNECTED 1
)
add_subdirectory(${Kokkos_kernels_SOURCE_DIR} ${Kokkos_kernels_BINARY_DIR})
set(CUDA_FOUND ${CUDA_FOUND} PARENT_SCOPE)
