cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("RecurrentJs")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# PACKAGES
find_package(Protobuf REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)

# COMPILER SETUP
SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS                "-Wall -std=c++11")

# COMPILING PROTOBUFFS
file(GLOB ProtoFiles "${CMAKE_CURRENT_SOURCE_DIR}/examples/protobuff/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

# INCLUDES
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR}/Eigen)
include_directories(${ProtoHeaders})

# BINARIES
file(GLOB CppFiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
ADD_EXECUTABLE(character_prediction "examples/character_prediction.cpp" ${CppFiles} "OptionParser/OptionParser.cpp" ${ProtoSources})
target_link_libraries(character_prediction ${PROTOBUF_LIBRARY} ${ZLIB_LIBRARIES})
