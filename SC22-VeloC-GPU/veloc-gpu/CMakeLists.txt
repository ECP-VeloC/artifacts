cmake_minimum_required (VERSION 3.10)
project (VELOC)

##### Global settings
include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
add_definitions(-O2 -g -Wall -D__BENCHMARK -D__ASSERT -D__INFO)
set(CMAKE_CXX_STANDARD 14)

##### Configuration setting
set(VELOC_RESOURCE_MANAGER "LSF" CACHE STRING "Resource Manager for CLI (SLURM LSF ALPS NONE)")
set_property(CACHE VELOC_RESOURCE_MANAGER PROPERTY STRINGS SLURM LSF ALPS NONE)
set(COMM_QUEUE "ipc_queue" CACHE STRING "Communication protocol between client library and active backend")
set_property(CACHE COMM_QUEUE PROPERTY STRINGS ipc_queue socket_queue thallium_queue)

##### Find dependencies

# PThreads
find_package(Threads REQUIRED)
set(COMM_QUEUE_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

# Communication protocol
if (${COMM_QUEUE} STREQUAL "ipc_queue")
    # Boost is required for ipc_queue
    set(Boost_FIND_REQUIRED True)
    find_package(Boost 1.40)
    include_directories(${Boost_INCLUDE_DIR})
    list(APPEND COMM_QUEUE_LIBRARIES "rt")
elseif (${COMM_QUEUE} STREQUAL "thallium_queue")
    # Thallium is required for thallium_queue
    find_package(thallium REQUIRED)
    list(APPEND COMM_QUEUE_LIBRARIES "thallium")
endif ()
configure_file(${PROJECT_SOURCE_DIR}/src/common/comm_queue.in
               ${PROJECT_SOURCE_DIR}/src/common/comm_queue.hpp
               @ONLY)

# OpenSSL
find_package(OpenSSL REQUIRED)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# internal dependencies
find_package(AXL REQUIRED)
include_directories(${AXL_INCLUDE_DIRS})

find_package(ER REQUIRED)
include_directories(${ER_INCLUDE_DIRS})

# CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# CUDA Profiler tools
# find_library(CUDA_NVTX_LIBRARY REQUIRED
#   NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
#   PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" "/usr/local/cuda-11.0/targets/x86_64-linux/include/" ENV LD_LIBRARY_PATH
#   PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
#   DOC "Location of the CUDA Toolkit Extension (NVTX) library"
#   NO_DEFAULT_PATH
# )

# -----------------------------------------------------------------------------------
include_directories(${VELOC_SOURCE_DIR})
add_subdirectory (src)

# -----------------------------------------------------------------------------------
# enable_testing()
# add_subdirectory (test)
