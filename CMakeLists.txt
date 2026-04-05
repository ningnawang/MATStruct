# ###############################################################################
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_ARCHITECTURES 61 75)
project(VolumeVoronoiGPU LANGUAGES CXX CUDA)

# ###############################################################################
set(EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# ###############################################################################
find_package(CUDA REQUIRED)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# find_package(OpenMP)
# if(OPENMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

find_package(ZLIB REQUIRED)

find_package(CGAL REQUIRED)

# Get Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if (Eigen3_FOUND)
    # message(STATUS "${EIGEN3_VERSION_STRING}")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# add dependencies
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
option(AUTO_DOWNLOAD "Auto download dependencies" ON)
# enable GPU for libmat
option(LIBMAT_IS_USE_GPU    "Enable GPU for RPD" ON)
option(LIBMAT_IS_BUILD_TEST "Build tests, either CXX or CUDA" OFF)
option(MATSTRUCT_WITH_VISUALIZATION "Enable polyscope visualization (requires display)" OFF)
include(matstruct_dependencies)
add_subdirectory(extern/BGAL)
add_subdirectory(src/knn)
add_subdirectory(src/opt)
add_subdirectory(src/eval)
add_subdirectory(src/init)
add_subdirectory(src/topKdist2mesh)
add_subdirectory(src/dist2meshAABBTree)

# ###############################################################################
# For VolumeVoronoiGPU binary
# ###############################################################################
set(HEADER_LIST)
set(SOURCE_LIST
    "src/main_voronoi.cpp" # yes, not .cu (for compiling CGAL)
    "src/main_gui.cxx"
    "src/medial_struct.cpp"
    "src/fix_intf_wrapper.cpp"
    "src/thinning_wrapper.cpp"
    "src/io_wrapper.cpp"
)

# add_executable(${PROJECT_NAME} ${SOURCE_LIST} ${HEADER_LIST} ${CPU_SOURCES} ${CPU_HEADERS}
add_executable(${PROJECT_NAME} ${SOURCE_LIST} ${HEADER_LIST})
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    >)
# target_compile_features(${PROJECT_NAME} PUBLIC cuda_std_17)
target_compile_definitions(${PROJECT_NAME} PUBLIC -Dgeogram_EXPORTS)
if(MATSTRUCT_WITH_VISUALIZATION)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DMATSTRUCT_WITH_VISUALIZATION)
endif()

# add common_headers include
target_include_directories(${PROJECT_NAME} PUBLIC
	# $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${EXTERNAL_DIR}/libmat/include>
	$<INSTALL_INTERFACE:include>)

if(${MSVC})
    if(${CMAKE_VERSION} VERSION_GREATER 3.6.0)
        set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT "${PROJECT_NAME}")
    endif()
endif()

target_link_directories(${PROJECT_NAME} PUBLIC "${CMAKE_BINARY_DIR}/lib")
target_link_libraries(${PROJECT_NAME}
    stdc++fs
    #### 
    optimized libmat
    ####
    optimized opt
    optimized eval
    optimized init
    optimized topKdist2mesh
    optimized dist2meshAABBTree
    #### extern lib
    general cuda
    general cublas
    $<$<BOOL:${MATSTRUCT_WITH_VISUALIZATION}>:polyscope>
    optimized geogram
    optimized ZLIB::ZLIB
    optimized CGAL::CGAL
    optimized Eigen3::Eigen
    optimized CLI11::CLI11
)
target_link_libraries(${PROJECT_NAME} general ${CUDA_LIBRARIES})
# add common headers
target_include_directories(${PROJECT_NAME} PUBLIC 
    ${EXTERNAL_DIR}/libmat/include
    ${EXTERNAL_DIR}/libmat/src/matfun
    ${EXTERNAL_DIR}/libmat/src/matfun_fix
    ${EXTERNAL_DIR}/libmat/src/IO/IO_CXX
    ${EXTERNAL_DIR}/libmat/src/IO/IO_CUDA
    ${CMAKE_SOURCE_DIR}/src/opt
    ${CMAKE_SOURCE_DIR}/src/eval
    ${CMAKE_SOURCE_DIR}/src/init
    ${CMAKE_SOURCE_DIR}/src/topKdist2mesh
    ${CMAKE_SOURCE_DIR}/src/dist2meshAABBTree
)