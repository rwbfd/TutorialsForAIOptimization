cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(cuda_first LANGUAGES CXX CUDA)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CXX_FLAGS "-O2 -Wall -Wextra")

add_executable(main main.cu)
#set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "8.6")
target_compile_features(main PUBLIC cxx_std_11)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)