cmake_minimum_required(VERSION 3.5...3.18)
project(FastMI LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow specifying a custom Clang compiler path
if(DEFINED CLANG_PATH)
    set(CMAKE_CXX_COMPILER ${CLANG_PATH})
endif()

# Option to enable Boost (ON by default)
option(ENABLE_BOOST "Enable Boost Libraries" ON)

# Option to enable Benchmark (OFF by default)
option(ENABLE_BENCHMARK "Enable Google Benchmark" OFF)

set(PROJECT_VERSION 0.1)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(extern/pybind11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

#find_package(FastNB REQUIRED)
add_subdirectory(extern/FastNB)

# Find required packages
if(ENABLE_BENCHMARK)
    find_package(benchmark REQUIRED)
endif()

if(ENABLE_BOOST)
    # Specify the Boost components you need
    set(BOOST_COMPONENTS filesystem system)

    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost 1.70.0 REQUIRED system)

    if(Boost_FOUND)
        message(STATUS "Boost version: ${Boost_VERSION}")
        message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
        message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
    else()
        message(FATAL_ERROR "Boost not found!")
    endif()
endif()

# Create a library for the core functionality
add_library(fast_mi_lib STATIC fast_mutual_information/fast_mi.cpp)
add_library(FastMI::fast_mi_lib ALIAS fast_mi_lib)

# Add executable target
add_executable(fast_mi fast_mutual_information/fast_mi.cpp)

target_include_directories(fast_mi_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/fast_mutual_information>
    $<INSTALL_INTERFACE:include/fast_mutual_information>
    $<$<BOOL:${ENABLE_BOOST}>:${Boost_INCLUDE_DIRS}>
)

find_package(Threads REQUIRED)

target_link_libraries(fast_mi PUBLIC
#    absl::flat_hash_map
    Threads::Threads
    Eigen3::Eigen
    Boost::headers
    Boost::system
    FastNB::fast_nb_lib
)

target_link_libraries(fast_mi_lib PUBLIC
#    absl::flat_hash_map
    Threads::Threads
    Eigen3::Eigen
    Boost::headers
    Boost::system
    FastNB::fast_nb_lib
)

# Create the Python module using Pybind11
pybind11_add_module(fast_mutual_information fast_mutual_information/bindings.cpp)

target_link_libraries(fast_mutual_information PRIVATE fast_mi_lib)

if(ENABLE_BENCHMARK)
    target_link_libraries(fast_mi PRIVATE benchmark::benchmark)
    target_compile_definitions(fast_mi PRIVATE ENABLE_BENCHMARK=1)
    target_compile_definitions(fast_mi_lib PRIVATE ENABLE_BENCHMARK=1)
    target_compile_definitions(fast_mutual_information PRIVATE ENABLE_BENCHMARK=1)
endif()

# Set compiler flags based on the environment - maybe we can be a bit less strict
# for Apple silicon?
if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Building on GitHub Actions: Using no arch specific flags")
    target_compile_options(fast_mi_lib PRIVATE -O3 -ffast-math)
    target_compile_options(fast_mi PRIVATE -O3 -ffast-math)
else()
    message(STATUS "Building locally: Using -march=native")
    target_compile_options(fast_mi_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(fast_mi PRIVATE -O3 -march=native -ffast-math)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_mi_lib PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fast_mi PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(fast_mutual_information PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS fast_mutual_information
    EXPORT FastMITargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(TARGETS fast_mi_lib
    EXPORT FastMITargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install the executable
install(TARGETS fast_mi
    RUNTIME DESTINATION bin
)

# Install the Python module
install(TARGETS fast_mutual_information
    LIBRARY DESTINATION lib/python_modules
)

# Install header files
install(DIRECTORY fast_mutual_information/
    DESTINATION include/fast_mutual_information
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Export the targets to a CMake script
install(EXPORT FastMITargets
    FILE FastMITargets.cmake
    NAMESPACE FastMI::
    DESTINATION lib/cmake/FastMI
)

# Create a package configuration file
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

# Install the package configuration files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/FastMIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/FastMIConfigVersion.cmake"
    DESTINATION lib/cmake/FastMI
)
