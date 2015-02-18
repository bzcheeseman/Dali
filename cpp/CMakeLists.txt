cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("recurrentjs")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# PACKAGES
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)

# COMPILER SETUP
SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS                "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

# INCLUDES
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

# ADDING DIRECTORIES WITH THEIR OWN CMAKELISTS FILES
add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf)
add_subdirectory(${PROJECT_SOURCE_DIR}/core)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)

