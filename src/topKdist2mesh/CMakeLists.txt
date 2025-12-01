cmake_minimum_required(VERSION 3.16)
add_library(topKdist2mesh STATIC topKdist2mesh.h topKdist2mesh.cu)

# add common headers
target_include_directories(topKdist2mesh PUBLIC 
    ${EXTERNAL_DIR}/libmat/include
)

