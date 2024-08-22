cmake_minimum_required(VERSION 3.17.0)
set(CMAKE_GENERATOR "Ninja")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 86)

project(playground VERSION 0.1.0 LANGUAGES CUDA CXX)

add_subdirectory(nvbench)

find_package(TBB REQUIRED)

add_executable(playground main.cu)
add_executable(playgroundc main.cpp)

add_executable(all_of all_of.cpp)

target_compile_options(playground PRIVATE "--extended-lambda") 

target_include_directories(playground PRIVATE "/home/ggonidelis/cccl/cub" ${TBB_INCLUDE_DIRS})

target_link_libraries(playground PRIVATE TBB::tbb nvbench::main)

target_include_directories(playground PRIVATE "/home/ggonidelis/cccl/thrust")
target_include_directories(playground PRIVATE "/home/ggonidelis/cccl/libcudacxx/include")

message(STATUS "TBB include directories: ${TBB_INCLUDE_DIRS}")

target_include_directories(all_of PRIVATE "/home/ggonidelis/cccl/thrust")
target_include_directories(all_of PRIVATE "/home/ggonidelis/cccl/libcudacxx/include")
target_include_directories(all_of PRIVATE "/home/ggonidelis/cccl/cub" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_source_files_properties(all_of.cpp PROPERTIES LANGUAGE CUDA)



