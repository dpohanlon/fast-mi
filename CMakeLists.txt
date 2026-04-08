cmake_minimum_required(VERSION 3.10...3.18)
project(FastMI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_BENCHMARK "Enable Google Benchmark" OFF)
option(ENABLE_BOOST "Enable Boost Libraries" ON)

set(PROJECT_VERSION 0.1)

if(DEFINED CLANG_PATH)
    set(CMAKE_CXX_COMPILER ${CLANG_PATH})
endif()

# Benchmark (optional)
if(ENABLE_BENCHMARK)
    find_package(benchmark REQUIRED)
endif()

# Python + pybind11 (mirror FastNB pattern)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT TARGET pybind11::module)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
endif()

# Threads
find_package(Threads REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# FastNB (in-tree)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/FastNB)

if(ENABLE_BOOST)
    # Specify the Boost components you need
    set(BOOST_COMPONENTS sort)

    # Optionally specify the Boost root if it's in a non-standard location
    # set(BOOST_ROOT "/path/to/boost")
    # set(BOOST_INCLUDEDIR "/path/to/boost/include")
    # set(BOOST_LIBRARYDIR "/path/to/boost/lib")

    set(Boost_USE_STATIC_LIBS OFF)
    find_package(Boost CONFIG REQUIRED headers)

    if(Boost_FOUND)
        message(STATUS "Boost version: ${Boost_VERSION}")
        message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
        message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
    else()
        message(FATAL_ERROR "Boost not found!")
    endif()
endif()

# Core library
add_library(fast_mi_lib STATIC fast_mutual_information/fast_mi.cpp)
add_library(FastMI::fast_mi_lib ALIAS fast_mi_lib)

# Executable
add_executable(fast_mi fast_mutual_information/fast_mi.cpp)

target_include_directories(fast_mi_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${Boost_INCLUDE_DIRS}
)

# Linking
target_link_libraries(fast_mi PRIVATE fast_mi_lib Threads::Threads)
target_link_libraries(fast_mi PUBLIC Eigen3::Eigen)
target_link_libraries(fast_mi_lib PUBLIC Eigen3::Eigen FastNB::fast_nb_lib)
target_link_libraries(fast_mi PRIVATE Boost::headers)

# Python module
pybind11_add_module(fast_mutual_information fast_mutual_information/bindings.cpp)
target_link_libraries(fast_mutual_information PRIVATE fast_mi_lib)

# Optional benchmark flags
if(ENABLE_BENCHMARK)
    target_link_libraries(fast_mi PRIVATE benchmark::benchmark)
    target_compile_definitions(fast_mi PRIVATE ENABLE_BENCHMARK=1)
    target_compile_definitions(fast_mi_lib PRIVATE ENABLE_BENCHMARK=1)
    target_compile_definitions(fast_mutual_information PRIVATE ENABLE_BENCHMARK=1)
endif()

# Compile flags
if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Building on GitHub Actions: Using no arch specific flags")
    target_compile_options(fast_mi_lib PRIVATE -O3 -ffast-math)
    target_compile_options(fast_mi PRIVATE -O3 -ffast-math)
else()
    message(STATUS "Building locally: Using -march=native")
    target_compile_options(fast_mi_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(fast_mi PRIVATE -O3 -march=native -ffast-math)
endif()

# OpenMP (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_mi_lib PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fast_mi PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fast_mutual_information PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install static lib for native consumers
install(TARGETS fast_mi_lib
    EXPORT FastMITargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(TARGETS fast_mutual_information
        LIBRARY DESTINATION fast_mutual_information
        RUNTIME DESTINATION fast_mutual_information
        ARCHIVE DESTINATION fast_mutual_information)

# Install C++ headers (not Python package files)
install(DIRECTORY fast_mutual_information/
    DESTINATION include/fast_mutual_information
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export CMake targets
install(EXPORT FastMITargets
    FILE FastMITargets.cmake
    NAMESPACE FastMI::
    DESTINATION lib/cmake/FastMI
)

# CMake package config
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FastMIConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/FastMIConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/FastMIConfig.cmake"
    INSTALL_DESTINATION lib/cmake/FastMI
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FastMIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FastMIConfigVersion.cmake"
    DESTINATION lib/cmake/FastMI
)
