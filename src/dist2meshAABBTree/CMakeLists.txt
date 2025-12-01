cmake_minimum_required(VERSION 3.12)
project(dist2meshAABBTree LANGUAGES CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(dist2meshAABBTree STATIC dist2meshAABBTree.cu)

# add common headers
target_include_directories(dist2meshAABBTree PUBLIC 
    ${EXTERNAL_DIR}/libmat/include
)

set_target_properties(dist2meshAABBTree PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)



