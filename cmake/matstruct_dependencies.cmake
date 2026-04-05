# ###############################################################################
# CMake download helpers
# ###############################################################################

# download external dependencies
include(matstruct_downloadExternal)

# ###############################################################################
# Required dependencies
# ###############################################################################

# geogram
if(NOT TARGET geogram)
    rpd_download_geogram()
    include(geogram) # check geogram.cmake
endif()

# # Boost
# if(TETWILD_WITH_HUNTER)
# hunter_add_package(Boost COMPONENTS thread system)
# endif()

# # fmt
# if(NOT TARGET fmt::fmt)
# tetwild_download_fmt()
# add_subdirectory(${TETWILD_EXTERNAL}/fmt)
# endif()

# # spdlog
# if(NOT TARGET spdlog::spdlog)
# tetwild_download_spdlog()
# add_library(spdlog INTERFACE)
# add_library(spdlog::spdlog ALIAS spdlog)
# target_include_directories(spdlog INTERFACE ${TETWILD_EXTERNAL}/spdlog/include)
# target_compile_definitions(spdlog INTERFACE -DSPDLOG_FMT_EXTERNAL)
# target_link_libraries(spdlog INTERFACE fmt::fmt)
# endif()

# libigl
if(NOT TARGET igl::core)
    rpd_download_libigl()
endif()

# # pymesh loaders
# add_subdirectory(${TETWILD_EXTERNAL}/pymesh)

# CL11
if(NOT TARGET CLI11::CLI11)
    rpd_download_cli11()
    add_subdirectory(${EXTERNAL_DIR}/cli11)
endif()

# polyscope
#if(MATSTRUCT_WITH_VISUALIZATION)
#    if(NOT TARGET polyscope)
#        rpd_download_polyscope()
#        add_subdirectory(${EXTERNAL_DIR}/polyscope)
#    endif()
#endif()

# json
if(NOT TARGET nlohmann_json::nlohmann_json)
    rpd_download_json()
endif()

# # cub cuda
# if (NOT TARGET CUB)
#     add_subdirectory(${EXTERNAL_DIR}/cub)
# endif()


if(NOT TARGET libmat)
    rpd_download_libmat()
    add_subdirectory(${EXTERNAL_DIR}/libmat)
endif()


# if(NOT TARGET nanoflann)
#     rpd_download_nanoflann()
#     add_subdirectory(${EXTERNAL_DIR}/nanoflann)
# endif()

if (NOT TARGET geometry-central)
    rpd_download_geometrycentral()
endif()