# CMakeLists.txt for building and linking Google Benchmark and Google Abseil
cmake_minimum_required(VERSION 3.14) # Updated to 3.14 for FetchContent
project(FastMI LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow specifying a custom Clang compiler path
if(DEFINED CLANG_PATH)
    set(CMAKE_CXX_COMPILER ${CLANG_PATH})
endif()

# Find Google Benchmark installed via Homebrew
find_package(benchmark REQUIRED)

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

# Add executable target
add_executable(fast_mi fast_mi.cpp)

# Link Google Benchmark, Abseil, and pthread
find_package(Threads REQUIRED)
target_link_libraries(fast_mi
    benchmark::benchmark
    absl::flat_hash_map # Replace with needed Abseil components
    Threads::Threads
)

# Set optimization flags for performance
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(fast_mi PRIVATE -O3 -march=native -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(fast_mi PRIVATE -O3 -march=native -ffast-math)
endif()

# Ensure OpenMP is not enforced on Apple Silicon
if(NOT APPLE)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fast_mi OpenMP::OpenMP_CXX)
    endif()
endif()
