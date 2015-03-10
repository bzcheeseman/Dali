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
find_package(GooglePerfTools)
find_package(BLAS)
find_package(MKL)
find_package(GTest)

if (NOT GPERFTOOLS_FOUND)
    message("WARNING: consider installing gperftools for better multithreaded performance.\n    > brew install gperftools")
endif(NOT GPERFTOOLS_FOUND)

# This monstrosity came to live to alleviate two conditions:
#      1) ctest ignores colored output
#      2) make test does not build tests (wtf)
function(add_testsuite TEST_TARGET TEST_FILE  TEST_DEPENDENCIES)
    if(GTEST_FOUND)
        add_executable(${TEST_TARGET} ${TEST_FILE})
        add_dependencies(${TEST_TARGET} ${TEST_DEPENDENCIES})
        target_link_libraries(${TEST_TARGET} ${TEST_DEPENDENCIES} ${GTEST_BOTH_LIBRARIES})
        add_test(${TEST_TARGET} ${TEST_TARGET})
        add_dependencies(build_tests ${TEST_TARGET})

        add_custom_target(${TEST_TARGET}_run ${TEST_TARGET})
        add_dependencies(run_tests ${TEST_TARGET}_run)
    endif(GTEST_FOUND)
endfunction(add_testsuite)

if(GTEST_FOUND)
    enable_testing()
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_custom_target(run_tests true)
    add_custom_target(build_tests)
    add_dependencies(run_tests build_tests)
else(GTEST_FOUND)
    message("WARNING: GTESTS not found - install gtests to be able to run tests")
endif(GTEST_FOUND)

# CLEANING UP AFTER FINDERS
# some library list variables are FALSE by default if libs are not found,
# but it is much more useful to keep it empty.
set(LIBVARS_TO_FIX MKL_LIBRARIES BLAS_LIBRARIES GPERFTOOLS_TCMALLOC_AND_PROFILER)
foreach(libvar ${LIBVARS_TO_FIX})
    if (NOT ${libvar})
        set(${libvar} "")
    endif (NOT ${libvar})
endforeach(libvar)

# GFLAGS STUFF
add_definitions(-DGFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})

# COMPILER SETUP
add_definitions(-DBUILD_DIR=${CMAKE_BUILD_DIR})

message("=> BUILD_TYPE " ${CMAKE_BUILD_TYPE})

set (CMAKE_CXX_COMPILER             "clang++")
if(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
else(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w")
endif(CMAKE_BUILD_TYPE MATCHES "debug")



# INCLUDES
include_directories(${MKL_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${GFLAGS_INCLUDE_DIRS})
if (GPERFTOOLS_FOUND)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
endif (GPERFTOOLS_FOUND)

# ADDING DIRECTORIES WITH THEIR OWN CMAKELISTS FILES
add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf)
add_subdirectory(${PROJECT_SOURCE_DIR}/core)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)

