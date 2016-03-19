cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("Dali")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# REQUIRED PACKAGES
find_package(ZLIB REQUIRED)
find_package(Gflags REQUIRED)
find_package(BLAS REQUIRED)

# OPTIONAL PACKAGES
find_package(GooglePerfTools)
find_package(MKL)
find_package(GTest)
find_package(CUDA)

# CUDA BEGINS HERE
if (DEFINED WITH_CUDA)
    message("WITH_CUDA ${WITH_CUDA}")
else()
    if (CUDA_FOUND STREQUAL TRUE)
        SET(WITH_CUDA TRUE)
    else()
        SET(WITH_CUDA FALSE)
        message("CUDA not found - did you know that Dali has GPU support?")
    endif()
endif(DEFINED WITH_CUDA)


if (WITH_CUDA)
    SET(DALI_USE_CUDA 1)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    LIST(APPEND CUDA_NVCC_FLAGS --compiler-options ${CUDA_EXTRA_FLAGS} -lineinfo -Xptxas -dlcm=cg  -use_fast_math -std=c++11)
    SET(CUDA_EXTRA_FLAGS "-fno-strict-aliasing")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda/lib")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_EXTRA_FLAGS}")
    SET(MSHADOW_USE_CUDA 1)
else (WITH_CUDA)
    SET(MSHADOW_USE_CUDA 0)
endif (WITH_CUDA)

# CUDA ENDS HERE
if (NOT GPERFTOOLS_FOUND)
    message("WARNING: consider installing gperftools for better multithreaded performance.\n    > brew install gperftools")
endif(NOT GPERFTOOLS_FOUND)

# This monstrosity came to live to alleviate two conditions:
#      1) ctest ignores colored output
#      2) make test does not build tests (wtf)
set(TESTS_MAIN "${PROJECT_SOURCE_DIR}/dali/tests_main.cpp")
function(add_testsuite TEST_TARGET TEST_DEPENDENCIES TEST_FILES)
    list(APPEND TEST_FILES ${TESTS_MAIN})
    if(GTEST_FOUND)
        add_executable(${TEST_TARGET} ${TEST_FILES})
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
    ADD_CUSTOM_TARGET(run_tests_heapcheck
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/test_runner.sh ${CMAKE_BUILD_DIR}
        WORKING_DIRECTORY ${CMAKE_BUILD_DIR})
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

# add submodule libraries


# Relative path in C++ are hard. Adding a macro:
set(DALI_DATA_DIR "${PROJECT_SOURCE_DIR}/data")


message("-- Build type: " ${CMAKE_BUILD_TYPE})
if (APPLE)
    set (CMAKE_CXX_COMPILER "clang++")
    set (MAYBE_SHARED "")
else(APPLE)
    # land of Linux or Windows
    set (MAYBE_SHARED "SHARED")
endif(APPLE)


if(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC")
else(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w -fPIC")
endif(CMAKE_BUILD_TYPE MATCHES "debug")

# mshadow
set(MSHADOW_USE_CBLAS 1)
set(MSHADOW_USE_MKL 0)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/mshadow)

# INCLUDES
include_directories(${MKL_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/third_party/libcuckoo/src)

if (GPERFTOOLS_FOUND)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
endif (GPERFTOOLS_FOUND)


# ADDING DIRECTORIES WITH THEIR OWN CMAKELISTS FILES
add_subdirectory(${PROJECT_SOURCE_DIR}/dali)

# THOSE HEADERS ARE REQUIRED FOR DALI HEADERS TO WORK, SO WE AUTOMATICALLY INSTALL THEM.
# install mshadow
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/mshadow/mshadow DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    PATTERN "*.cuh")
# # install cuckoo
# file(GLOB_RECURSE CUCKOO_HEADERS       "${PROJECT_SOURCE_DIR}/third_party/libcuckoo/src/*.hh")
# install(FILES ${CUCKOO_HEADERS} DESTINATION include)
