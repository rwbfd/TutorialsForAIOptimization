
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(cmake_map_reduce LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(CUDAMapReduce src/main.cpp src/cuda_functions.cu
  )


# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(CUDAMapReduce
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )



