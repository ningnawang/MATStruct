# /////////////////////////////////////////////////////////////
# Make library
cmake_minimum_required(VERSION 3.16)
add_library(init 
    init_sampling.h
    init_sampling.cpp
)

target_link_libraries(init
    optimized geometry-central
    optimized inputs
)


# add common headers
target_include_directories(init PUBLIC
    ${EXTERNAL_DIR}/libmat/include
    ${EXTERNAL_DIR}/libmat/src/inputs
    ${EXTERNAL_DIR}/geometry-central/include
)
