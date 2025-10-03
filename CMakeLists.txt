cmake_minimum_required(VERSION 3.16)

# =========================
# Project
# =========================
project(GSI_sMINRES VERSION 1.0.0 LANGUAGES CXX)

# =========================
# C++ standard & flags
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

# 実行時にビルドツリーの lib/ を見に行く（インストールなしで examples を起動可）
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# =========================
# Install prefix
# =========================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/gsi-sminres_install" CACHE PATH "Install path" FORCE)
  message(STATUS "Install prefix not specified. Using default: ${CMAKE_INSTALL_PREFIX}")
else()
  message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
endif()

# =========================
# Options
# =========================
option(USE_OPENMP                     "Enable OpenMP parallelization" ON)
option(GSI_SMINRES_BUILD_SHARED       "Build shared library"           ON)
option(GSI_SMINRES_BUILD_STATIC       "Build static library"           ON)
option(GSI_SMINRES_BUILD_EXAMPLES     "Build example programs"         ON)

# =========================
# OpenMP
# =========================
set(OPENMP_CXX_TARGET "")
if(USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling parallelization.")
    set(OPENMP_CXX_TARGET OpenMP::OpenMP_CXX)
  else()
    message(WARNING "OpenMP not found, disabling parallelization.")
  endif()
else()
  message(STATUS "OpenMP disabled.")
endif()

# =========================
# BLAS / LAPACK
# =========================
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "BLAS  : ${BLAS_LIBRARIES}")
message(STATUS "LAPACK: ${LAPACK_LIBRARIES}")

# ---- Optional: OpenBLAS version check (ある場合のみ)
find_path(OPENBLAS_INCLUDE_DIR openblas_config.h)
if(OPENBLAS_INCLUDE_DIR)
  file(READ "${OPENBLAS_INCLUDE_DIR}/openblas_config.h" OPENBLAS_CONFIG_CONTENTS)
  string(REGEX MATCH "#define OPENBLAS_VERSION \"[^\"]+\"" OPENBLAS_VERSION_LINE "${OPENBLAS_CONFIG_CONTENTS}")
  string(REGEX REPLACE "#define OPENBLAS_VERSION \"([^\"]+)\"" "\\1" OPENBLAS_VERSION "${OPENBLAS_VERSION_LINE}")
  message(STATUS "Detected OpenBLAS: ${OPENBLAS_VERSION}")
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" OPENBLAS_VERSION_NUMBER "${OPENBLAS_VERSION}")
  set(OPENBLAS_REQUIRED_VERSION "0.3.27")
  if(OPENBLAS_VERSION_NUMBER VERSION_LESS OPENBLAS_REQUIRED_VERSION)
    message(FATAL_ERROR "OpenBLAS ${OPENBLAS_VERSION_NUMBER} is too old (< ${OPENBLAS_REQUIRED_VERSION}). Please upgrade.")
  else()
    message(STATUS "OpenBLAS ${OPENBLAS_VERSION_NUMBER} (OK)")
  endif()
endif()

# =========================
# Sources
# =========================
set(GSI_SMINRES_HEADERS
  ${PROJECT_SOURCE_DIR}/include
)

set(GSI_SMINRES_SOURCES
  # algorithms
  src/algorithms/standard_shifted_minres.cpp
  src/algorithms/shift_invert_shifted_minres.cpp
  src/algorithms/generalized_shifted_minres.cpp
  # extras/algorithms
  src/extras/algorithms/minres_pencil.cpp
  src/extras/algorithms/cg.cpp
  # extras/io
  src/extras/io/mm_csr.cpp
  src/extras/io/mm_zhp.cpp
)

# =========================
# Libraries
# =========================
# NOTE:
#   BLAS/LAPACK/OpenMP は PUBLIC として伝播させる
#   例: examples は gsi_sminres にリンクするだけでよい
if(GSI_SMINRES_BUILD_SHARED)
  add_library(gsi_sminres SHARED ${GSI_SMINRES_SOURCES})
  target_include_directories(gsi_sminres PUBLIC
    $<BUILD_INTERFACE:${GSI_SMINRES_HEADERS}>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(gsi_sminres PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OPENMP_CXX_TARGET})
  set_target_properties(gsi_sminres PROPERTIES
    OUTPUT_NAME gsi_sminres
    VERSION     ${PROJECT_VERSION}
    SOVERSION   1)
  add_library(gsi_sminres::gsi_sminres ALIAS gsi_sminres)
endif()

if(GSI_SMINRES_BUILD_STATIC)
  add_library(gsi_sminres_static STATIC ${GSI_SMINRES_SOURCES})
  target_include_directories(gsi_sminres_static PUBLIC
    $<BUILD_INTERFACE:${GSI_SMINRES_HEADERS}>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(gsi_sminres_static PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OPENMP_CXX_TARGET})
  # Static でも PIC を付けておくと便利（Linux での共有リンクに対応）
  set_target_properties(gsi_sminres_static PROPERTIES
    OUTPUT_NAME gsi_sminres
    POSITION_INDEPENDENT_CODE ON)
  add_library(gsi_sminres::gsi_sminres_static ALIAS gsi_sminres_static)
endif()

# =========================
# Examples
# =========================
if(GSI_SMINRES_BUILD_EXAMPLES)
  set(_examples
    example_std_zhp
    example_std_csr
    example_sis_zhp
    example_sis_csr
    example_gen_zhp
    example_gen_csr
  )

  foreach(x ${_examples})
    add_executable(${x} examples/${x}.cpp)
    target_include_directories(${x} PRIVATE ${GSI_SMINRES_HEADERS})
    if(TARGET gsi_sminres)
      target_link_libraries(${x} PRIVATE gsi_sminres)
    else()
      target_link_libraries(${x} PRIVATE gsi_sminres_static)
    endif()
  endforeach()
endif()

# =========================
# Install
# =========================
include(CMakePackageConfigHelpers)

# headers
install(DIRECTORY include/ DESTINATION include)

# libraries
if(TARGET gsi_sminres)
  install(TARGETS gsi_sminres
    EXPORT  gsi_sminresTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
endif()
if(TARGET gsi_sminres_static)
  install(TARGETS gsi_sminres_static
    EXPORT  gsi_sminresTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
endif()

# package config
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/gsi_sminresConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/gsi_sminresConfig.cmake
  INSTALL_DESTINATION lib/cmake/gsi_sminres)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/gsi_sminresConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/gsi_sminresConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/gsi_sminresConfigVersion.cmake
  DESTINATION lib/cmake/gsi_sminres)

install(EXPORT      gsi_sminresTargets
        FILE        gsi_sminresTargets.cmake
        NAMESPACE   gsi_sminres::
        DESTINATION lib/cmake/gsi_sminres)
