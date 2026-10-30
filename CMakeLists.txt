cmake_minimum_required(VERSION 3.20)
project(dqc_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(dqc_core
  cpp/src/rng.cpp
)
target_include_directories(dqc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_compile_features(dqc_core PUBLIC cxx_std_20)
target_compile_options(dqc_core PRIVATE -Wall -Wextra)
set_target_properties(dqc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- KaHyPar (GPL-3.0) -------------------------------------------------------
# The `HypergraphPartitioning` allocator is backed by KaHyPar, called through
# its C facade (`include/libkahypar.h`). Two human-provided layouts are
# supported, selected by one path input:
#
#   source tree : <root>/lib/libkahypar.cc (+ kahypar/, external_tools/)
#   install tree: <prefix>/lib/libkahypar.so + <prefix>/include/libkahypar.h
#
# Inputs:
#   KAHYPAR_ROOT             source tree or install prefix (env or -D)
#   DQC_KAHYPAR_SOURCE_DIR   legacy alias for a source tree
#   DQC_KAHYPAR_LIBRARY / DQC_KAHYPAR_INCLUDE_DIR  direct overrides
#   DQC_KAHYPAR_MODE         AUTO | SOURCE | INSTALLED | PKGCONFIG | OFF
set(DQC_KAHYPAR_MODE "AUTO" CACHE STRING
    "KaHyPar backend: AUTO | SOURCE | INSTALLED | PKGCONFIG | OFF")
set_property(CACHE DQC_KAHYPAR_MODE PROPERTY STRINGS AUTO SOURCE INSTALLED PKGCONFIG OFF)

option(DQC_REQUIRE_KAHYPAR "Fail if the KaHyPar backend cannot be configured" ON)

set(DQC_KAHYPAR_SOURCE_DIR "" CACHE PATH
    "KaHyPar source tree (legacy alias; prefer KAHYPAR_ROOT)")
set(KAHYPAR_ROOT "" CACHE PATH
    "KaHyPar source tree or install prefix (also read from the environment)")
if(NOT KAHYPAR_ROOT AND DEFINED ENV{KAHYPAR_ROOT})
  set(KAHYPAR_ROOT "$ENV{KAHYPAR_ROOT}" CACHE PATH
      "KaHyPar source tree or install prefix" FORCE)
endif()
set(DQC_KAHYPAR_LIBRARY "" CACHE FILEPATH "Explicit libkahypar location")
set(DQC_KAHYPAR_INCLUDE_DIR "" CACHE PATH "Explicit directory containing libkahypar.h")

if(DQC_KAHYPAR_MODE STREQUAL "AUTO")
  if(DQC_KAHYPAR_SOURCE_DIR AND EXISTS "${DQC_KAHYPAR_SOURCE_DIR}/lib/libkahypar.cc")
    set(DQC_KAHYPAR_MODE SOURCE)
  elseif(KAHYPAR_ROOT AND EXISTS "${KAHYPAR_ROOT}/lib/libkahypar.cc")
    set(DQC_KAHYPAR_MODE SOURCE)
  elseif(KAHYPAR_ROOT AND EXISTS "${KAHYPAR_ROOT}/include/libkahypar.h")
    set(DQC_KAHYPAR_MODE INSTALLED)
  elseif(DQC_KAHYPAR_LIBRARY AND DQC_KAHYPAR_INCLUDE_DIR)
    set(DQC_KAHYPAR_MODE INSTALLED)
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(_dqc_kahypar QUIET IMPORTED_TARGET libkahypar)
    endif()
    if(_dqc_kahypar_FOUND)
      set(DQC_KAHYPAR_MODE PKGCONFIG)
    else()
      set(DQC_KAHYPAR_MODE OFF)
    endif()
  endif()
endif()

set(DQC_HAS_KAHYPAR FALSE)

if(DQC_KAHYPAR_MODE STREQUAL "SOURCE")
  if(DQC_KAHYPAR_SOURCE_DIR)
    set(_dqc_kahypar_src "${DQC_KAHYPAR_SOURCE_DIR}")
  else()
    set(_dqc_kahypar_src "${KAHYPAR_ROOT}")
  endif()
  find_package(Boost 1.69 REQUIRED COMPONENTS program_options)
  find_package(Threads REQUIRED)
  set(KAHYPAR_VERSION_GIT_SHA1 dqc_build)
  configure_file(
    ${_dqc_kahypar_src}/kahypar/git_revision.h.in
    ${CMAKE_BINARY_DIR}/generated/kahypar/git_revision.h)
  add_library(dqc_kahypar STATIC
    ${_dqc_kahypar_src}/lib/libkahypar.cc)
  target_include_directories(dqc_kahypar PUBLIC
    ${_dqc_kahypar_src}
    ${_dqc_kahypar_src}/include
    ${_dqc_kahypar_src}/external_tools
    ${_dqc_kahypar_src}/external_tools/googletest/googletest/include
    ${_dqc_kahypar_src}/external_tools/googletest/googlemock/include
    ${CMAKE_BINARY_DIR}/generated)
  # KaHyPar targets C++17 (its code uses `requires` as an identifier, which is
  # a keyword in C++20), so compile the facade with C++17 even though the rest
  # of the core uses C++20. The directory-level CMAKE_CXX_STANDARD is 20, so
  # the target property must be set explicitly to override it.
  set_target_properties(dqc_kahypar PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)
  target_compile_options(dqc_kahypar PRIVATE -w)
  target_link_libraries(dqc_kahypar PUBLIC Boost::program_options Threads::Threads)
  set(DQC_HAS_KAHYPAR TRUE)

elseif(DQC_KAHYPAR_MODE STREQUAL "INSTALLED")
  if(NOT DQC_KAHYPAR_LIBRARY)
    find_library(DQC_KAHYPAR_LIBRARY NAMES kahypar
      HINTS ${KAHYPAR_ROOT}/lib ${KAHYPAR_ROOT}/lib64 REQUIRED)
  endif()
  if(NOT DQC_KAHYPAR_INCLUDE_DIR)
    find_path(DQC_KAHYPAR_INCLUDE_DIR NAMES libkahypar.h
      HINTS ${KAHYPAR_ROOT}/include REQUIRED)
  endif()
  add_library(dqc_kahypar SHARED IMPORTED)
  set_target_properties(dqc_kahypar PROPERTIES
    IMPORTED_LOCATION "${DQC_KAHYPAR_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${DQC_KAHYPAR_INCLUDE_DIR}")
  # The installed extension depends on libkahypar.so; keep its directory on
  # the installed RPATH for local installs.
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(DQC_HAS_KAHYPAR TRUE)

elseif(DQC_KAHYPAR_MODE STREQUAL "PKGCONFIG")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(KAHYPAR REQUIRED IMPORTED_TARGET libkahypar)
  add_library(dqc_kahypar INTERFACE)
  target_link_libraries(dqc_kahypar INTERFACE PkgConfig::KAHYPAR)
  set(DQC_HAS_KAHYPAR TRUE)
endif()

if(DQC_HAS_KAHYPAR)
  target_link_libraries(dqc_core PUBLIC dqc_kahypar)
  message(STATUS "KaHyPar: ${DQC_KAHYPAR_MODE}")
elseif(DQC_REQUIRE_KAHYPAR)
  message(FATAL_ERROR
    "KaHyPar not found. Set KAHYPAR_ROOT to a KaHyPar source tree or install "
    "prefix, or pass -DDQC_KAHYPAR_MODE=OFF -DDQC_REQUIRE_KAHYPAR=OFF to build "
    "without the HypergraphPartitioning backend.")
else()
  message(WARNING
    "KaHyPar disabled; the HypergraphPartitioning allocator is unavailable.")
endif()

option(DQC_BUILD_TESTS "Build unit tests" ON)
if(DQC_BUILD_TESTS)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(dqc_tests
    cpp/tests/test_rng.cpp
    cpp/tests/test_hypergraph.cpp
    cpp/tests/test_graph.cpp
    cpp/tests/test_pyset.cpp
    cpp/tests/test_circuit_analysis.cpp
    cpp/tests/test_placement.cpp
    cpp/tests/test_hypergraph_circuit.cpp
    cpp/tests/test_distribution.cpp
    cpp/tests/test_bipartite.cpp
    cpp/tests/test_gateset.cpp
    cpp/tests/test_pacman.cpp
    cpp/tests/test_allocators.cpp
    cpp/tests/test_refiners.cpp
    cpp/tests/test_to_pytket_circuit.cpp
    cpp/tests/test_network.cpp
    cpp/tests/test_composition.cpp
  )
  if(DQC_HAS_KAHYPAR)
    target_sources(dqc_tests PRIVATE cpp/tests/test_hypergraph_partitioning.cpp)
    target_compile_definitions(dqc_tests PRIVATE
      DQC_KAHYPAR_INI="${CMAKE_CURRENT_SOURCE_DIR}/python/dqc_cpp/km1_kKaHyPar_sea20.ini")
  endif()
  target_link_libraries(dqc_tests PRIVATE dqc_core Catch2::Catch2WithMain)
  target_compile_options(dqc_tests PRIVATE -Wall -Wextra)

  include(Catch)
  catch_discover_tests(dqc_tests)
endif()

option(DQC_BUILD_BINDINGS "Build pybind11 Python bindings" OFF)
if(DQC_BUILD_BINDINGS)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(dqc_cpp_core cpp/python/bindings.cpp)
  target_link_libraries(dqc_cpp_core PRIVATE dqc_core)
  if(DQC_HAS_KAHYPAR)
    target_compile_definitions(dqc_cpp_core PRIVATE DQC_HAS_KAHYPAR=1)
  endif()
  set_target_properties(dqc_cpp_core PROPERTIES
    OUTPUT_NAME "_core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dqc_cpp"
  )
  # Ship the KaHyPar INI next to the compiled module so the default ini_path
  # always resolves (the `_core` module dir fallback in `default_ini_path`).
  add_custom_command(TARGET dqc_cpp_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_SOURCE_DIR}/python/dqc_cpp/km1_kKaHyPar_sea20.ini"
      "${CMAKE_BINARY_DIR}/dqc_cpp/km1_kKaHyPar_sea20.ini")

  # Install layout consumed by both `cmake --install` and scikit-build-core:
  # the extension and the default INI live inside `pytket_dqc_cpp/`.
  install(TARGETS dqc_cpp_core
          LIBRARY DESTINATION pytket_dqc_cpp
          RUNTIME DESTINATION pytket_dqc_cpp)
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/dqc_cpp/km1_kKaHyPar_sea20.ini"
    DESTINATION pytket_dqc_cpp)
  if(NOT SKBUILD)
    # Plain `cmake --install` ships the pure-Python package too.
    # scikit-build-core already packs it from `wheel.packages`.
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/pytket_dqc_cpp/"
            DESTINATION pytket_dqc_cpp
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE
            PATTERN "*.so" EXCLUDE)
  endif()
endif()
