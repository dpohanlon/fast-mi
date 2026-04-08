cmake_minimum_required(VERSION 3.10...3.18)
project(mimiq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_BENCHMARK "Enable Google Benchmark" OFF)
option(ENABLE_BOOST "Enable Boost Libraries" ON)

set(PROJECT_VERSION 0.1)

if(DEFINED CLANG_PATH)
    set(CMAKE_CXX_COMPILER ${CLANG_PATH})
endif()

if(ENABLE_BENCHMARK)
    find_package(benchmark REQUIRED)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT TARGET pybind11::module)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)
endif()

find_package(Threads REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/FastNB)

if(ENABLE_BOOST)
    set(BOOST_COMPONENTS sort)

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

add_library(mimiq_lib STATIC mimiq/fast_mi.cpp)
add_library(Mimiq::mimiq_lib ALIAS mimiq_lib)

add_executable(mimiq_cli mimiq/fast_mi.cpp)

target_include_directories(mimiq_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(mimiq_cli PRIVATE mimiq_lib Threads::Threads)
target_link_libraries(mimiq_cli PUBLIC Eigen3::Eigen)
target_link_libraries(mimiq_lib PUBLIC Eigen3::Eigen FastNB::fast_nb_lib)

if(ENABLE_BOOST)
    target_link_libraries(mimiq_cli PRIVATE Boost::headers)
endif()

pybind11_add_module(mimiq mimiq/bindings.cpp)
target_link_libraries(mimiq PRIVATE mimiq_lib)

if(ENABLE_BENCHMARK)
    target_link_libraries(mimiq_cli PRIVATE benchmark::benchmark)
    target_compile_definitions(mimiq_cli PRIVATE ENABLE_BENCHMARK=1)
    target_compile_definitions(mimiq_lib PRIVATE ENABLE_BENCHMARK=1)
    target_compile_definitions(mimiq PRIVATE ENABLE_BENCHMARK=1)
endif()

if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Building on GitHub Actions: Using no arch specific flags")
    target_compile_options(mimiq_lib PRIVATE -O3 -ffast-math)
    target_compile_options(mimiq_cli PRIVATE -O3 -ffast-math)
else()
    message(STATUS "Building locally: Using -march=native")
    target_compile_options(mimiq_lib PRIVATE -O3 -march=native -ffast-math)
    target_compile_options(mimiq_cli PRIVATE -O3 -march=native -ffast-math)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mimiq_lib PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(mimiq_cli PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(mimiq PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS mimiq_lib
    EXPORT MimiqTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(TARGETS mimiq
    LIBRARY DESTINATION mimiq
    RUNTIME DESTINATION mimiq
    ARCHIVE DESTINATION mimiq
)

install(DIRECTORY mimiq/
    DESTINATION include/mimiq
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(EXPORT MimiqTargets
    FILE MimiqTargets.cmake
    NAMESPACE Mimiq::
    DESTINATION lib/cmake/MIMIQ
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MimiqConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MIMIQConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MIMIQConfig.cmake"
    INSTALL_DESTINATION lib/cmake/MIMIQ
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MIMIQConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MIMIQConfigVersion.cmake"
    DESTINATION lib/cmake/MIMIQ
)
