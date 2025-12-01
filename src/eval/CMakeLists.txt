# /////////////////////////////////////////////////////////////
# Make library
cmake_minimum_required(VERSION 3.16)
add_library(eval 
    mesh_eval.h
    mesh_eval.cpp 
)

target_link_libraries(eval
    optimized geogram
    optimized topKdist2mesh
)

# add common headers
target_include_directories(eval PUBLIC
    ${EXTERNAL_DIR}/libmat/include
    ${EXTERNAL_DIR}/libmat/src/inputs
    ${EXTERNAL_DIR}/libmat/src/matbase
    ${CMAKE_SOURCE_DIR}/src/topKdist2mesh
)
