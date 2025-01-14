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

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
#find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(extern/pybind11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(FastNB REQUIRED)

# Include FetchContent module
include(FetchContent)

# Declare Google Abseil
FetchContent_Declare(
    abseil
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
)

# Optionally set Abseil CMake options before fetching
set(absl_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(absl_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(absl_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Make Abseil available
FetchContent_MakeAvailable(abseil)

if(ENABLE_BOOST)
    # Specify the Boost components you need
    set(BOOST_COMPONENTS filesystem system math)  # Added 'math'

    # Optionally specify the Boost root if it's in a non-standard location
    # set(BOOST_ROOT "/path/to/boost")
    # set(BOOST_INCLUDEDIR "/path/to/boost/include")
    # set(BOOST_LIBRARYDIR "/path/to/boost/lib")

    find_package(Boost 1.70.0 REQUIRED system math)

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

# Specify include directories for the library
target_include_directories(fast_mi_lib PUBLIC fast_mutual_information)

# Add executable target
add_executable(fast_mi fast_mutual_information/fast_mi.cpp)

target_include_directories(fast_mi_lib PUBLIC
    fast_mutual_information
    ${Boost_INCLUDE_DIRS}  # Ensure Boost includes are PUBLIC
)

find_package(Threads REQUIRED)

target_link_libraries(fast_mi PUBLIC
    absl::flat_hash_map
    Threads::Threads
    Eigen3::Eigen
    Boost::headers
    Boost::system
    FastNB::fast_nb_lib
)

target_link_libraries(fast_mi_lib PUBLIC
    absl::flat_hash_map
    Threads::Threads
    Eigen3::Eigen
    Boost::headers
    Boost::system
    FastNB::fast_nb_lib
)

# Create the Python module using Pybind11
pybind11_add_module(fast_mutual_information fast_mutual_information/bindings.cpp)

target_link_libraries(fast_mutual_information PRIVATE fast_mi_lib)

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
        LIBRARY DESTINATION fast_mutual_information)
