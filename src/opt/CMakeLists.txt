cmake_minimum_required(VERSION 3.16)
add_library(opt STATIC  
    opt_rpd.h
    opt_rpd.cxx
    rpd3d_wrapper.h
    rpd3d_wrapper.cxx
    particle.h
    particle.cxx
)

# add lib
target_link_libraries(opt
    optimized inputs
    optimized matbase
    optimized rpd3d
    optimized rpd3d_api
    optimized IO_CXX
    optimized matfun
    optimized knn
    optimized dist2meshAABBTree
    # BGAL lib
    optimized LBFGS
    optimized Integral
)

# add common headers
target_include_directories(opt PUBLIC 
    ${EXTERNAL_DIR}/BGAL/LBFGS
    ${EXTERNAL_DIR}/BGAL/Integral
    ${EXTERNAL_DIR}/libmat/include
    ${EXTERNAL_DIR}/libmat/src/inputs
    ${EXTERNAL_DIR}/libmat/src/matbase
    ${EXTERNAL_DIR}/libmat/src/matfun
    ${EXTERNAL_DIR}/libmat/src/rpd_3d
    ${EXTERNAL_DIR}/libmat/src/rpd3d_api
    ${EXTERNAL_DIR}/libmat/src/IO/IO_CXX
    ${EXTERNAL_DIR}/libmat/src/IO/IO_CUDA
    ${CMAKE_SOURCE_DIR}/src/knn
    ${CMAKE_SOURCE_DIR}/src/dist2meshAABBTree
)
