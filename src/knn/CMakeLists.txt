# /////////////////////////////////////////////////////////////
# Make library
cmake_minimum_required(VERSION 3.16)
add_library(knn 
    nanoflann.hpp
    knn.h
    knn.cpp 
)

target_link_libraries(knn
    optimized geogram
)

# add common headers
target_include_directories(knn PUBLIC
    ${EXTERNAL_DIR}/libmat/include
)
