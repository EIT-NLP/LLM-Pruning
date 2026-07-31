CMAKE_MINIMUM_REQUIRED(VERSION 3.26)

set(PRJ dynamic_pruning_kernels)
if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT "$ENV{CUDA_HOME}")
elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT "$ENV{CUDA_PATH}")
elseif(EXISTS "$ENV{HOME}/.local/opt/cuda/bin/nvcc")
    set(CUDA_TOOLKIT_ROOT "$ENV{HOME}/.local/opt/cuda")
else()
    set(CUDA_TOOLKIT_ROOT "/usr/local/cuda")
endif()

message(STATUS "CUDA_TOOLKIT_ROOT: ${CUDA_TOOLKIT_ROOT}")

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT}/bin/nvcc")
set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT}")
project(${PRJ} CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_ARCHITECTURES "120")

message("CMAKE_CURRENT_PATH: ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMAKE_PREFIX: ${CMAKE_PREFIX_PATH}")

file(GLOB_RECURSE srcs CONFIGURE_DEPENDS "csrc/*.cpp" "csrc/*.cu")

add_library(
    ${PRJ} MODULE
    ${srcs}
)

target_include_directories(
    ${PRJ} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm-ffi/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/sglang/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode enabled")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-G")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g")
endif()
