cmake_minimum_required (VERSION 3.5)
project (libOpenCL_loader)

add_library(libOpenCL_loader
        src/opencl_loader.cpp
)

target_include_directories(libOpenCL_loader PUBLIC "${PROJECT_SOURCE_DIR}/include/")

add_definitions(-DENABLE_OPENCL=1)
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-deprecated-declarations")
endif()

