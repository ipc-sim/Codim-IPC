#---------------------------------------------------------------------------------------------------
# WARNING: By default, the GPL module Cholmod/Supernodal is enabled. This leads to a 2x speedup
# compared to simplicial mode. This is optional and can be disabled by setting WITH_GPL to OFF.
#---------------------------------------------------------------------------------------------------

if(TARGET SuiteSparse::SuiteSparse)
    return()
endif()

message(STATUS "Third-party: creating targets 'SuiteSparse::SuiteSparse'")

include(FetchContent)
FetchContent_Declare(
    suitesparse
    GIT_REPOSITORY https://github.com/sergiud/SuiteSparse.git
    GIT_TAG 3b92085cb5c7fe7917d12bcd5dd346502366d10f
)

FetchContent_GetProperties(suitesparse)
if(NOT suitesparse_POPULATED)
    FetchContent_Populate(suitesparse)
endif()

include(CMakeDependentOption)
option(BUILD_CXSPARSE "Build CXSparse" OFF)
option(WITH_FORTRAN "Enables Fortran support" OFF)
option(WITH_DEMOS "Build demos" OFF)
option(WITH_PRINT "Print diagnostic messages" ON)
option(WITH_TBB "Enables Intel Threading Building Blocks support" OFF)
option(WITH_LGPL "Enable GNU LGPL modules" ON)
option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_OPENMP "Enable OpenMP support" OFF)
cmake_dependent_option(WITH_GPL "Enable GNU GPL modules" ON "WITH_LGPL" OFF)
cmake_dependent_option(WITH_CAMD "Enable interfaces to CAMD, CCOLAMD, CSYMAMD in Partition module" OFF "WITH_LGPL" OFF)
cmake_dependent_option(WITH_PARTITION "Enable the Partition module" OFF "WITH_LGPL AND METIS_FOUND" OFF)
cmake_dependent_option(WITH_CHOLESKY "Enable the Cholesky module" ON "WITH_LGPL" OFF)
cmake_dependent_option(WITH_CHECK "Enable the Check module" OFF "WITH_LGPL" OFF)
cmake_dependent_option(WITH_MODIFY "Enable the Modify module" OFF "WITH_GPL" OFF)
cmake_dependent_option(WITH_MATRIXOPS "Enable the MatrixOps module" OFF "WITH_GPL" OFF)
cmake_dependent_option(WITH_SUPERNODAL "Enable the Supernodal module" ON "WITH_GPL" OFF)

if (NOT APPLE)
    include(mkl)
endif()
if(NOT TARGET blas)
    add_library(blas INTERFACE IMPORTED GLOBAL)
endif()
if(NOT TARGET lapack)
    add_library(lapack INTERFACE IMPORTED GLOBAL)
endif()

function(suitesparse_import_target)
    macro(ignore_package NAME)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}Config.cmake "")
        set(${NAME}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "" FORCE)
    endmacro()

    # Prefer Config mode before Module mode
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

    ignore_package(BLAS)
    ignore_package(CBLAS)
    ignore_package(LAPACK)

    add_subdirectory(${suitesparse_SOURCE_DIR} ${suitesparse_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

suitesparse_import_target()

cmake_policy(SET CMP0079 NEW)
if (NOT APPLE)
    target_link_libraries(cholmod PUBLIC mkl::mkl)
else()
    find_package(LAPACK)
    target_link_libraries(cholmod PUBLIC LAPACK::LAPACK)
endif()
add_library(SuiteSparse_SuiteSparse INTERFACE)
add_library(SuiteSparse::SuiteSparse ALIAS SuiteSparse_SuiteSparse)

foreach(name IN ITEMS cholmod)
    if(NOT TARGET ${name})
        message(FATAL_ERROR "${name} is not a valid CMake target. Please check your config!")
    endif()
    target_link_libraries(SuiteSparse_SuiteSparse INTERFACE ${name})
endforeach()
