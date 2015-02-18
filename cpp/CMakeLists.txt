cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("recurrentjs")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(MKL_MULTI_THREADED "1")

# REQUIRED PACKAGES
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Gflags REQUIRED)

# OPTIONAL PACAKGES
find_package(BLAS)
find_package(MKL)

# CLEANING UP AFTER FINDERS
# some library list variables are FALSE by default if libs are not found,
# but it is much more useful to keep it empty.
set(LIBVARS_TO_FIX MKL_LIBRARIES BLAS_LIBRARIES)
foreach(libvar ${LIBVARS_TO_FIX})
    if (NOT ${libvar})
        set(${libvar} "")
    endif (NOT ${libvar})
endforeach(libvar)

# COMPILER SETUP
SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# INCLUDES
include_directories(${MKL_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})

# ADDING DIRECTORIES WITH THEIR OWN CMAKELISTS FILES
add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf)
add_subdirectory(${PROJECT_SOURCE_DIR}/core)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)

