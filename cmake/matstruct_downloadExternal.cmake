# ###############################################################################
include(DownloadProject)
include(FetchContent)

# Shortcut function
function(rpd_download_project name)
	if(AUTO_DOWNLOAD)
		download_project(
			PROJ ${name}
			SOURCE_DIR ${EXTERNAL_DIR}/${name}
			DOWNLOAD_DIR ${EXTERNAL_DIR}/.cache/${name}
			${ARGN}
		)
	endif()
endfunction()

# ###############################################################################

# libigl
function(rpd_download_libigl)
	# rpd_download_project(libigl
	# GIT_REPOSITORY https://github.com/libigl/libigl.git
	# GIT_TAG v2.4.0
	# )
	FetchContent_Declare(libigl
		GIT_REPOSITORY https://github.com/libigl/libigl.git
		GIT_TAG v2.5.0
	)
	FetchContent_MakeAvailable(libigl)
endfunction()

# geogram
function(rpd_download_geogram)
	rpd_download_project(geogram
		GIT_REPOSITORY https://github.com/alicevision/geogram.git
		GIT_TAG v1.7.5
	)
endfunction()

# ## fmt
# function(tetwild_download_fmt)
# rpd_download_project(fmt
# GIT_REPOSITORY https://github.com/fmtlib/fmt.git
# GIT_TAG        5.2.0
# )
# endfunction()

# ## spdlog
# function(tetwild_download_spdlog)
# rpd_download_project(spdlog
# GIT_REPOSITORY https://github.com/gabime/spdlog.git
# GIT_TAG        v1.1.0
# )
# endfunction()

## CLI11
function(rpd_download_cli11)
	rpd_download_project(cli11
		GIT_REPOSITORY     https://github.com/CLIUtils/CLI11
		GIT_TAG            v2.5.0
	)
endfunction()

# polyscope
function(rpd_download_polyscope)
	rpd_download_project(polyscope
		GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
		GIT_TAG eb07f8acaf5c8dba30de9587f84b14ba0c411ca1
	)
endfunction()

# json
function(rpd_download_json)
FetchContent_Declare(
	json 
	URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)
endfunction()

# nanoflann
function(rpd_download_nanoflann)
	rpd_download_project(nanoflann
		GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
		GIT_TAG v1.5.5
	)
endfunction()

# libmat 
function(rpd_download_libmat)
	rpd_download_project(libmat
		# GIT_REPOSITORY https://github.com/ningnawang/libmat.git
		GIT_REPOSITORY git@github.com:ningnawang/libmat.git
		GIT_TAG main
	)
endfunction()

# geometry central
# commit 98c4fc17f363ee45af151e41f393c196a984c505 contains CMAKE CMP0169 policy change
# which will make our project fail to build, use any commit before that
function(rpd_download_geometrycentral)
	FetchContent_Declare(geometry-central
		GIT_REPOSITORY https://github.com/nmwsharp/geometry-central.git
		GIT_TAG 43866a16ad7f77f51e1db284a3bd8cb6d38f9839
	)
	FetchContent_MakeAvailable(geometry-central)
endfunction()