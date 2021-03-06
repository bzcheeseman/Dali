cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

project("dali")
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake
                      ${PROJECT_SOURCE_DIR}/cmake/FindCUDA
                      ${CMAKE_MODULE_PATH})

if( UNIX AND NOT APPLE )
    execute_process(COMMAND /usr/bin/lsb_release -i -s
                            OUTPUT_VARIABLE LINUX_DISTRO
                            OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(UNIX AND NOT APPLE)


if (NOT DEFINED DALI_STACKTRACES)
    if (DEFINED ENV{DALI_STACKTRACES})
        set(DALI_STACKTRACES $ENV{DALI_STACKTRACES})
    else()
        set(DALI_STACKTRACES "none")
    endif()
endif()

set (DALI_STACKTRACES_VALUES "none;backward-cpp;apple")

list (FIND DALI_STACKTRACES_VALUES "${DALI_STACKTRACES}" DALI_STACKTRACES_VALUE_FOUND)
if (DALI_STACKTRACES_VALUE_FOUND EQUAL -1)
    message(FATAL_ERROR "DALI_STACKTRACES must be one of ${DALI_STACKTRACES_VALUES} (was: ${DALI_STACKTRACES})")
endif()


if (DALI_STACKTRACES STREQUAL "apple")
    set(DALI_APPLE_STACKTRACES 1)
else()
    set(DALI_APPLE_STACKTRACES 0)
endif()

################################################################################
##                            COMPILER SETUP                                  ##
################################################################################

# ensure correct GCC version
set(GCC_MINIMUM_REQUIRED "4.9.2")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${GCC_MINIMUM_REQUIRED})
    message(FATAL_ERROR "Mimimum required version of gcc is ${GCC_MINIMUM_REQUIRED}")
  endif()
endif()

if (APPLE)
    set (CMAKE_CXX_COMPILER "clang++")
endif(APPLE)

if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    SET(DALI_CXX_COMPILER_ID "clang")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
    SET(DALI_CXX_COMPILER_ID "gnu")
else()
    message(FATAL "Compiler ${DALI_CXX_COMPILER_ID} is not supported.")
endif()

set(ENABLED_WARNINGS "-Wno-return-type -Wno-undefined-bool-conversion -Wno-potentially-evaluated-expression")

if(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC ${ENABLED_WARNINGS}")
elseif(CMAKE_BUILD_TYPE MATCHES "nooptimize")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${ENABLED_WARNINGS}")
else(CMAKE_BUILD_TYPE MATCHES "debug")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -w ${ENABLED_WARNINGS}")
endif(CMAKE_BUILD_TYPE MATCHES "debug")

if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "default")
endif (CMAKE_BUILD_TYPE STREQUAL "")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (NOT DALI_STACKTRACES STREQUAL "none")
    message(STATUS "compiling with pretty stacktraces")
endif()

################################################################################
##                            PACKAGES                                        ##
################################################################################
if( UNIX AND NOT APPLE AND LINUX_DISTRO STREQUAL Fedora)
    set(WITH_BLAS "open")
endif()

# REQUIRED PACKAGES
find_package(ZLIB REQUIRED)
find_package(BLAS REQUIRED)

# OPTIONAL PACKAGES
find_package(OpenBlas QUIET)
find_package(GooglePerfTools QUIET)
find_package(MKL QUIET)
find_package(GTest QUIET)
find_package(CUDA 5.5)
if (NOT OpenBLAS_FOUND)
    set(OpenBLAS_LIB "")
endif(NOT OpenBLAS_FOUND)

IF ((NOT OpenBLAS_FOUND) AND (NOT MKL_FOUND) AND APPLE AND (BLAS_LIBRARIES MATCHES "Accelerate.framework"))
    message(STATUS "Using Apple's Accelerate Framework BLAS")
    list(APPEND BLAS_LIBRARIES "blas")
    include_directories("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
    set(DALI_BLAS_INCLUDE_DIR "/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/")
ENDIF()

if (NOT GPERFTOOLS_FOUND)
    if (APPLE)
        message(WARNING "consider installing gperftools for better multithreaded performance.\n    > brew install gperftools")
    else(APPLE)
        message(WARNING "consider installing gperftools for better multithreaded performance.")
    endif(APPLE)
endif(NOT GPERFTOOLS_FOUND)


# CLEANING UP AFTER FINDERS
# some library list variables are FALSE by default if libs are not found,
# but it is much more useful to keep it empty.
message(STATUS "GPERFTOOLS_TCMALLOC_AND_PROFILER ${GPERFTOOLS_TCMALLOC_AND_PROFILER}")
set(LIBVARS_TO_FIX MKL_LIBRARIES BLAS_LIBRARIES GPERFTOOLS_TCMALLOC_AND_PROFILER OpenBLAS_LIB)
foreach(libvar ${LIBVARS_TO_FIX})
    if (NOT ${libvar})
        set(${libvar} "")
    endif (NOT ${libvar})
endforeach(libvar)

# INCLUDES
include_directories(${MKL_INCLUDE_DIR})
IF (OpenBLAS_FOUND)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(DALI_BLAS_INCLUDE_DIR ${OpenBLAS_INCLUDE_DIR})
ENDIF(OpenBLAS_FOUND)
include_directories(${PROJECT_SOURCE_DIR}/third_party/libcuckoo/src)

if (GPERFTOOLS_FOUND)
    include_directories(${GPERFTOOLS_INCLUDE_DIR})
endif (GPERFTOOLS_FOUND)

if (DALI_STACKTRACES STREQUAL "backward-cpp")
    # add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/backward-cpp)
    include(${PROJECT_SOURCE_DIR}/third_party/backward-cpp/BackwardMacros.cmake)
endif()


################################################################################
##                            CUDA SETUP                                      ##
################################################################################

if (CUDA_FOUND STREQUAL TRUE)
    find_library(CUDA_DRIVER_LIB cuda PATHS "/usr/lib64/"
                                            "/usr/lib"
                                            "/usr/lib/nvidia-current/"
                                            "${CUDA_TOOLKIT_ROOT_DIR}/lib/"
                                            NO_DEFAULT_PATH)
    message(STATUS "CUDA_DRIVER_LIB ${CUDA_DRIVER_LIB}")
    list(APPEND CUDA_LIBRARIES ${CUDA_DRIVER_LIB})
endif (CUDA_FOUND STREQUAL TRUE)

# CUDA BEGINS HERE
if (DEFINED WITH_CUDA)
    message(STATUS "Cuda support: ${WITH_CUDA}")
    if (WITH_CUDA)
        message(STATUS "Using CUDA version ${CUDA_VERSION_STRING}")
    else(WITH_CUDA)
        message(STATUS "Found (but not using) CUDA version ${CUDA_VERSION_STRING}")
    endif(WITH_CUDA)
else()
    if (CUDA_FOUND STREQUAL TRUE)
        SET(WITH_CUDA TRUE)
        message(STATUS "Using CUDA version ${CUDA_VERSION_STRING}")
    else()
        SET(WITH_CUDA FALSE)
        message(WARNING "CUDA not found - did you know that Dali has GPU support?")
    endif()
endif(DEFINED WITH_CUDA)

if (WITH_CUDA)
    SET(DALI_USE_CUDA 1)
    # Note: As of Jan 28 2018, using clang 3.9 is broken due to lack of support for tuples
    # https://devtalk.nvidia.com/default/topic/879726/nvcc-fails-if-c-11-activated-and-certain-headers-included/
    # if (${CUDA_VERSION} VERSION_GREATER "9.0" OR ${CUDA_VERSION} VERSION_EQUAL "9.0")
    #     if (${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    #         EXECUTE_PROCESS( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
    #         string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION_STRING ${clang_full_version_string})
    #         if (CLANG_VERSION_STRING VERSION_EQUAL "3.9")
    #             message(STATUS "setting nvcc C++ compiler to clang 3.9 ${CMAKE_C_COMPILER}")
    #             set(CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
    #         endif()
    #     endif()
    # endif()

    SET(DALI_CUDA_CXX_COMPILER ${CUDA_HOST_COMPILER})
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
    LIST(APPEND CUDA_NVCC_FLAGS
        --compiler-options ${CUDA_EXTRA_FLAGS}
        -lineinfo
        -use_fast_math
        -std=c++11
    )
    # precompile JIT optimizations for specific architectures
    set(SUPPORTED_JIT_PRECOMPILATIONS "")
    if (DEFINED ENV{DALI_GPU_ARCH})
        string(REPLACE ":" ";" DALI_GPU_ARCH_LIST $ENV{DALI_GPU_ARCH})
        list(APPEND SUPPORTED_JIT_PRECOMPILATIONS ${DALI_GPU_ARCH_LIST})
        message(STATUS "detected custom GPU architecture choices: ${SUPPORTED_JIT_PRECOMPILATIONS}")
    else()
        # have some future proofing (arch=compute_xx,code=compute_xx ensures forward compatibility)
        LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_20,code=compute_20)
        # also generate these architectures:
        list(APPEND SUPPORTED_JIT_PRECOMPILATIONS "30" "52" "53")
        if (${CUDA_VERSION} VERSION_GREATER "8.0" OR ${CUDA_VERSION} VERSION_EQUAL "8.0")
            list(APPEND SUPPORTED_JIT_PRECOMPILATIONS "61")
        endif()
    endif()

    foreach(gpu_arch ${SUPPORTED_JIT_PRECOMPILATIONS})
        # for PASCAL architectures we require cuda 8.0
        LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${gpu_arch},code=sm_${gpu_arch})
    endforeach()
    SET(CUDA_EXTRA_FLAGS "-fno-strict-aliasing")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_EXTRA_FLAGS}")
else (WITH_CUDA)
    SET(DALI_CUDA_CXX_COMPILER "compiled_without_cuda")
endif (WITH_CUDA)

function(cuda_compile_on)
    set(OLD_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "-O3 -w -fPIC ${CUDA_EXTRA_FLAGS}" PARENT_SCOPE)
endfunction()

function(cuda_compile_off)
    set(CMAKE_CXX_FLAGS ${OLD_CXX_FLAGS} PARENT_SCOPE)
    #SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} PARENT_SCOPE)
endfunction()

if (WITH_CUDA)
    find_package(CuDNN)
    if (CUDNN_FOUND)
        set(DALI_USE_CUDNN 1)
    else()
        set(DALI_USE_CUDNN 0)
    endif()
endif()

################################################################################
##                               DALI SETUP                                   ##
##                                   ---                                      ##
##                       COMPLIE TIME CONSTANTS                               ##
################################################################################

# Relative path in C++ are hard. Adding a macro:
set(DALI_DATA_DIR "${PROJECT_SOURCE_DIR}/data")

if (NOT DEFINED DALI_MAX_GPU_DEVICES)
    set(DALI_MAX_GPU_DEVICES 16)
endif()

if (NOT DALI_USE_CUDA)
    set(DALI_MAX_GPU_DEVICES 0)
endif()

################################################################################
##                               DALI SETUP                                   ##
##                                   ---                                      ##
##                       GENERATING VIRTUAL SOURCE TREE                       ##
################################################################################

if (NOT DEFINED DALI_SOURCE_DIRS)
    set(DALI_SOURCE_DIRS "dali")
endif()

set(ARRAY_ONLY 1)
if (ARRAY_ONLY)
    list(APPEND DALI_SOURCE_DIRS_EXCLUDE "dali/layers;dali/execution")
    list(APPEND DALI_SOURCE_DIRS_EXCLUDE "")
endif()


set(GENERATED_FILES_DIR ${CMAKE_BINARY_DIR}/dali_generated)
file(MAKE_DIRECTORY ${GENERATED_FILES_DIR})

function(dali_generate_file OldPath NewPath)
    add_custom_command(OUTPUT ${NewPath}
            COMMAND ${CMAKE_COMMAND} -E copy ${OldPath} ${NewPath}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            DEPENDS ${OldPath})
endfunction()

foreach(SourceDir ${DALI_SOURCE_DIRS})
    message(STATUS "Scanning sources under ${SourceDir}...")
    file(GLOB_RECURSE FilesInDir "${PROJECT_SOURCE_DIR}/${SourceDir}/*")
    foreach(FilePath ${FilesInDir})
        get_filename_component(FileNameWe ${FilePath} NAME_WE)
        get_filename_component(FileExt    ${FilePath} EXT)
        get_filename_component(FileDir  ${FilePath} DIRECTORY)
        file(RELATIVE_PATH FileRelDir  "${PROJECT_SOURCE_DIR}" "${FileDir}")
        file(RELATIVE_PATH FileRelPath "${PROJECT_SOURCE_DIR}" "${FilePath}")
        SET(IGNORE_FILE FALSE)
        foreach(ExcludePattern ${DALI_SOURCE_DIRS_EXCLUDE})
            if (FileRelPath MATCHES ${ExcludePattern})
                message(STATUS "Ignoring: ${FileRelPath}")
                SET(IGNORE_FILE TRUE)
            endif()
        endforeach()
        if (NOT IGNORE_FILE)
            if (FileExt STREQUAL ".cpp" OR (FileExt STREQUAL ".cu.cpp" AND NOT WITH_CUDA))
                set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.cpp")
                dali_generate_file("${FilePath}" "${NewFilePath}")
                if ((FileNameWe MATCHES "tests$") OR (FileNameWe STREQUAL "tests_main"))
                    list(APPEND DaliTests "${NewFilePath}")
                    list(APPEND DaliCudaTests "${NewFilePath}")
                else()
                    list(APPEND DaliSources "${NewFilePath}")
                endif()
            elseif(FileExt STREQUAL ".cu" OR (FileExt STREQUAL ".cu.cpp" AND WITH_CUDA))
                set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.cu")
                dali_generate_file("${FilePath}" "${NewFilePath}")
                if (FileNameWe MATCHES "tests$")
                    list(APPEND DaliCudaTests "${NewFilePath}")
                else()
                    list(APPEND DaliCudaSources "${NewFilePath}")
                endif()
            elseif(FileExt STREQUAL ".h.in")
                set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.h")
                configure_file("${FilePath}" "${NewFilePath}")
                list(APPEND DaliHeaders ${NewFilePath})
            elseif(FileExt STREQUAL ".h")
                set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.h")
                dali_generate_file("${FilePath}" "${NewFilePath}")
                list(APPEND DaliHeaders ${NewFilePath})
            elseif(FileExt STREQUAL ".DS_Store")
            else()
                message(Warning ": File ignored by compiler \"${FileRelPath}\"")
            endif()
        endif()
    endforeach()
endforeach()

message(STATUS "Copying nervana kernels")
file(GLOB_RECURSE FilesInDir "${PROJECT_SOURCE_DIR}/third_party/nervana_kernels/*")
foreach(FilePath ${FilesInDir})
    get_filename_component(FileNameWe ${FilePath} NAME_WE)
    get_filename_component(FileExt    ${FilePath} EXT)
    get_filename_component(FileDir  ${FilePath} DIRECTORY)
    file(RELATIVE_PATH FileRelDir  "${PROJECT_SOURCE_DIR}/third_party" "${FileDir}")
    if (FileExt STREQUAL ".sass")
        set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.sass")
        dali_generate_file("${FilePath}" "${NewFilePath}")
        list(APPEND NervanaKernels ${NewFilePath})
    elseif(FileExt STREQUAL ".pl")
        set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.pl")
        dali_generate_file("${FilePath}" "${NewFilePath}")
        list(APPEND NervanaKernels ${NewFilePath})
    elseif(FileExt STREQUAL ".pm")
        set(NewFilePath "${GENERATED_FILES_DIR}/${FileRelDir}/${FileNameWe}.pm")
        dali_generate_file("${FilePath}" "${NewFilePath}")
        list(APPEND NervanaKernels ${NewFilePath})
    endif()
endforeach()

if (DALI_STACKTRACES STREQUAL "backward-cpp")
    list(APPEND DaliSources "${PROJECT_SOURCE_DIR}/third_party/backward-cpp/backward.cpp")
    include_directories("${PROJECT_SOURCE_DIR}/third_party/backward-cpp")
    foreach(def ${BACKWARD_DEFINITIONS})
        add_definitions("-D${def}")
    endforeach()
endif()

################################################################################
##                               DALI SETUP                                   ##
##                                   ---                                      ##
##                      CONFIGURATION AND COMPILATION                         ##
################################################################################

include_directories(${GENERATED_FILES_DIR})

## CUDA FILES
if (WITH_CUDA)
    if (CUDNN_FOUND)
        include_directories(${CUDNN_INCLUDE})
    endif(CUDNN_FOUND)
    cuda_include_directories(${GENERATED_FILES_DIR})

    cuda_compile_on()
    cuda_include_directories(${DaliHeaders})
    cuda_compile(maybe_cuda_compiled_files ${DaliCudaSources} ${DaliHeaders})
    cuda_compile_off()
else()
    set(maybe_cuda_compiled_files "")
endif()

## CPP TARGET
add_library(dali SHARED ${DaliSources} ${maybe_cuda_compiled_files} ${DaliHeaders})

# ensure that headers are copied over before compilation
add_custom_target(dali_cuda_headers DEPENDS ${DaliHeaders})
add_dependencies(dali dali_cuda_headers)

# ensure that nervana kernels are copied over before compilation
add_custom_target(nervana_kernels DEPENDS ${NervanaKernels})
add_dependencies(dali nervana_kernels)

target_link_libraries(dali ${ZLIB_LIBRARIES}
                           ${BLAS_LIBRARIES}
                           ${MKL_LIBRARIES}
                           ${CMAKE_THREAD_LIBS_INIT}
                           ${OpenBLAS_LIB}
                           ${BACKWARD_LIBRARIES}
                           dl)

if (GPERFTOOLS_FOUND)
    target_link_libraries(dali ${GPERFTOOLS_LIBRARIES})
endif (GPERFTOOLS_FOUND)

if (WITH_CUDA)
    target_link_libraries(dali ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    if (CUDNN_FOUND)
        target_link_libraries(dali ${CUDNN_LIBRARY})
    endif (CUDNN_FOUND)
endif(WITH_CUDA)


# JIT FOLDER:
if (DALI_CACHE_DIR)
    add_definitions(-DDALI_CACHE_DIR=${DALI_CACHE_DIR})
endif(DALI_CACHE_DIR)


################################################################################
##                               DALI SETUP                                   ##
##                                   ---                                      ##
##                                TESTING                                     ##
################################################################################

# This monstrosity came to live to alleviate two conditions:
#      1) ctest ignores colored output
#      2) make test does not build tests (wtf)
function(add_testsuite TEST_TARGET RUN_TARGET TEST_DEPENDENCIES TEST_FILES DEVICE)
        if (DEVICE STREQUAL "GPU")
            cuda_compile_on()
            cuda_add_executable(${TEST_TARGET} ${TEST_FILES})
            cuda_compile_off()
        else (DEVICE STREQUAL "GPU")
            add_executable(${TEST_TARGET} ${TEST_FILES})
        endif(DEVICE STREQUAL "GPU")

        add_dependencies(${TEST_TARGET} ${TEST_DEPENDENCIES})
        target_link_libraries(${TEST_TARGET} ${TEST_DEPENDENCIES} ${GTEST_BOTH_LIBRARIES})
        add_test(${TEST_TARGET} ${TEST_TARGET})
        add_custom_target(${TEST_TARGET}_hidden_run_target ${TEST_TARGET})
        add_custom_target(${RUN_TARGET} true)
        add_dependencies(${RUN_TARGET} ${TEST_TARGET}_hidden_run_target)
endfunction(add_testsuite)

if(GTEST_FOUND)
    enable_testing()
    set(CTEST_OUTPUT_ON_FAILURE TRUE)
    include_directories(${GTEST_INCLUDE_DIRS})
    ADD_CUSTOM_TARGET(run_tests_heapcheck
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/test_runner.sh ${CMAKE_BINARY_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    if (WITH_CUDA)
        add_testsuite("test_dali" "run_tests" "dali" "${DaliTests};${DaliCudaTests}" "GPU")
        add_testsuite("test_dali_cpu_only" "run_cpu_tests" "dali" "${DaliTests}" "CPU")
    else (WITH_CUDA)
        add_testsuite("test_dali" "run_tests" "dali" "${DaliTests}" "CPU")
    endif (WITH_CUDA)
else(GTEST_FOUND)
    message(WARNING "GTESTS not found - install gtests to be able to run tests")
endif(GTEST_FOUND)

################################################################################
##                               DALI SETUP                                   ##
##                                   ---                                      ##
##                             INSTALLATION                                   ##
################################################################################

# THOSE HEADERS ARE REQUIRED FOR DALI HEADERS TO WORK, SO WE AUTOMATICALLY INSTALL THEM.
install(TARGETS dali DESTINATION lib)
install(DIRECTORY ${DaliDir} ${GENERATED_FILES_DIR}/dali DESTINATION include
        FILES_MATCHING PATTERN "*.h")

install(FILES ${PROJECT_SOURCE_DIR}/cmake/DaliConfig.cmake   DESTINATION lib/dali/cmake)
install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindOpenBlas.cmake DESTINATION lib/openblas/cmake RENAME OpenBlasConfig.cmake)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/nervana_kernels DESTINATION include
        FILES_MATCHING PATTERN "*.sass")

################################################################################
##                            PACKAGE CREATION                                ##
################################################################################
# Generic CPACK setttings

set (CPACK_PACKAGE_NAME ${PROJECT_NAME})
set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set (CPACK_PACKAGE_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
set (CPACK_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
set (CPACK_PACKAGE_DESCRIPTION "Buttery Smooth Automatic Differentiation in C++")

# name package according to device support
if (DALI_USE_CUDA)
    set(CPACK_XPU "gpu")
    set(CPACK_FORMULA_CLASSNAME "DaliGpu")
    set(CPACK_FORMULA_FNAME "dali-gpu")
else (DALI_USE_CUDA)
    set(CPACK_XPU "cpu")
    set(CPACK_FORMULA_CLASSNAME "DaliCpu")
    set(CPACK_FORMULA_FNAME "dali-cpu")
endif (DALI_USE_CUDA)


set (CPACK_SOURCE_GENERATOR "TGZ")
set (CPACK_SOURCE_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION})
set (CPACK_SOURCE_IGNORE_FILES "/data/;/docs/;/build/;/build2/;/build_2/;/build_cpu/;/.git/")

if ( UNIX AND NOT APPLE AND NOT WIN32 )
    message(STATUS "Package will be generated for Linux (DEB,RPM)")
    if (CMAKE_SIZEOF_VOID_P MATCHES "8")
        set(CPACK_ARCHITECTURE "x86_64")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    else()
        set(CPACK_ARCHITECTURE "i386")
        set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
    endif()
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}-${CPACK_ARCHITECTURE}-${CPACK_XPU}")

    set(CPACK_PACKAGE_CONTACT "Szymon Sidor szymon.sidor@gmail.com")
    # DEBIAN
    list(APPEND CPACK_GENERATOR "DEB")
	set(CPACK_DEBIAN_PACKAGE_DEPENDS "zlib1g-dev,build-essential,libblas-dev,gcc-4.9,g++-4.9,cpp-4.9")

    # FEDORA
    list(APPEND CPACK_GENERATOR "RPM")
    set(CPACK_RPM_PACKAGE_NAME ${PROJECT_NAME})
    set(CPACK_RPM_PACKAGE_SUMMARY ${CPACK_PACKAGE_DESCRIPTION})
    SET(CPACK_RPM_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}")
    SET(CPACK_RPM_PACKAGE_RELEASE "1")
    SET(CPACK_RPM_PACKAGE_LICENSE "MIT")
    SET(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")  # FROM: https://fedoraproject.org/wiki/RPMGroups
    SET(CPACK_RPM_PACKAGE_VENDOR "Dali Inc.")
    # TODO(szymon): add fedora dependencies
    #set(CPACK_RPM_PACKAGE_REQUIRES "python >= 2.5.0, cmake >= 2.8")

    include(CPack)
endif (UNIX AND NOT APPLE AND NOT WIN32 )

if ( APPLE )
    message(STATUS "Package will be generated for Mac OSX")
    set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}-${CPACK_XPU}")
    set(CPACK_PACKAGE_CONTACT "Jonathan Raiman jonathanraiman@gmail.com")
    SET(CPACK_GENERATOR "ZIP")

    add_custom_target(package_apple
        COMMAND "${CMAKE_CPACK_COMMAND}"
    "-C" "$<CONFIGURATION>" "--config" "${CMAKE_BINARY_DIR}/CpackConfig.cmake")

    add_custom_command(
        TARGET package_apple
        POST_BUILD
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/homebrew_dali_template.sh
                ${PROJECT_SOURCE_DIR}/misc/homebrew_dali_template.rb.in
                ${CMAKE_BINARY_DIR}/${CPACK_FORMULA_FNAME}.rb
                ${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_FILE_NAME}.zip
                "CPACK_PACKAGE_FILE_NAME:${CPACK_PACKAGE_FILE_NAME}"
                "CPACK_PACKAGE_DESCRIPTION:${CPACK_PACKAGE_DESCRIPTION}"
                "CPACK_PACKAGE_VERSION:${CPACK_PACKAGE_VERSION}"
                "CPACK_FORMULA_CLASSNAME:${CPACK_FORMULA_CLASSNAME}"

    )
    include(CPack)
endif (APPLE)
