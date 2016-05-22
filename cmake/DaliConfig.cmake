# Find the Dali library.
#
# This module defines
#  DALI_FOUND                   - True if Dali was found.
#  DALI_INCLUDE_DIRS            - Include directories for Dali headers.
#  DALI_AND_DEPS_INCLUDE_DIRS   - Include directories for Dali and its dependencies.
#  DALI_LIBRARIES               - Libraries for Dali.
#  DALI_AND_DEPS_LIBRARIES      - Libraries for Dali and its dependencies.
#

set(DALI_FOUND FALSE)
set(DALI_INCLUDE_DIRS)
set(DALI_AND_DEPS_INCLUDE_DIRS)
set(DALI_LIBRARIES)
set(DALI_AND_DEPS_LIBRARIES)


if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(Red         "${Esc}[31m")
endif(NOT WIN32)

if (Dali_FIND_REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(ZLIB REQUIRED)
else()
    find_package(BLAS)
    find_package(ZLIB)
    if (NOT BLAS_FOUND OR NOT ZLIB_FOUND)
        message(WARNING "Dali's dependencies are missing.")
    endif()
endif()
find_package(OpenBlas)

if (BLAS_FOUND)
    list(APPEND DALI_AND_DEPS_LIBRARIES ${BLAS_LIBRARIES})
endif (BLAS_FOUND)
if (ZLIB_FOUND)
    list(APPEND DALI_AND_DEPS_LIBRARIES ${ZLIB_LIBRARIES})
endif(ZLIB_FOUND)
if (OpenBLAS_FOUND)
    list(APPEND DALI_AND_DEPS_LIBRARIES ${OpenBLAS_LIB})
endif (OpenBLAS_FOUND)

# find cuda
find_package(CUDA)
if(CUDA_FOUND STREQUAL TRUE)
    list(APPEND DALI_AND_DEPS_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND STREQUAL TRUE)


if (DEFINED ENV{DALI_HOME} AND NOT "$ENV{DALI_HOME}" STREQUAL "")
    message(STATUS "Looking for dali at custom path $ENV{DALI_HOME}")
    set(DALI_CUSTOM_PATH TRUE)
    # where to look for Dali libraries
    LIST(APPEND DALI_LIBRARY_CUSTOM_PATHS "$ENV{DALI_HOME}/build")
    LIST(APPEND DALI_LIBRARY_CUSTOM_PATHS "$ENV{DALI_HOME}/build_cpu")
    LIST(APPEND DALI_LIBRARY_CUSTOM_PATHS "$ENV{DALI_HOME}/")

    # where to look for Dali headers
    LIST(APPEND DALI_HEADERS_CUSTOM_PATHS "$ENV{DALI_HOME}/build/dali_generated/")
    LIST(APPEND DALI_HEADERS_CUSTOM_PATHS "$ENV{DALI_HOME}/build_cpu/dali_generated/")
    LIST(APPEND DALI_HEADERS_CUSTOM_PATHS "$ENV{DALI_HOME}/dali_generated/")
else()
    set(DALI_CUSTOM_PATH FALSE)
endif ()


if (DALI_CUSTOM_PATH)
    find_library(DALI_LIBRARIES dali PATHS ${DALI_LIBRARY_CUSTOM_PATHS} NO_DEFAULT_PATH)
else()
    find_library(DALI_LIBRARIES dali HINTS)
endif()

if(DALI_LIBRARIES)
    set(DALI_FOUND TRUE)
    IF (APPLE)
        # Apple has trouble static linking, and this is the remedy:
        if (DALI_CUSTOM_PATH)
            find_library(DALI_LIBRARIES dali_cuda PATHS ${DALI_LIBRARY_CUSTOM_PATHS} NO_DEFAULT_PATH)
        else()
            find_library(DALI_LIBRARIES dali_cuda HINTS)
        endif()


        IF (DALI_CUDA_LIBRARIES)
            # Cuda is missing?
            list(APPEND DALI_LIBRARIES ${DALI_CUDA_LIBRARIES})
            IF (CUDA_FOUND STREQUAL TRUE)
                list(APPEND DALI_AND_DEPS_LIBRARIES ${CUDA_curand_LIBRARIES})
                list(APPEND DALI_AND_DEPS_LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
                list(APPEND DALI_AND_DEPS_LIBRARIES ${CUDA_LIBRARIES})
            ENDIF (CUDA_FOUND STREQUAL TRUE)
        ENDIF (DALI_CUDA_LIBRARIES)
        # BLAS not found:
    ENDIF (APPLE)

    message(STATUS "Found Dali: " ${DALI_LIBRARIES})
    list(APPEND DALI_AND_DEPS_LIBRARIES ${DALI_LIBRARIES})
else()
    if(Dali_FIND_QUIETLY)
        message(STATUS "Failed to find Dali   " ${REASON_MSG} ${ARGN})
    elseif(Dali_FIND_REQUIRED)
        message(FATAL_ERROR "${Red} Failed to find Dali   ${ColorReset}" ${REASON_MSG} ${ARGN})
    else()
        # Neither QUIETLY nor REQUIRED, use no priority which emits a message
        # but continues configuration and allows generation.
        message(WARNING "Failed to find Dali   " ${REASON_MSG} ${ARGN})
    endif()
endif()

if (DALI_CUSTOM_PATH)
    find_path(DALI_CONFIG_PATH "dali/config.h" PATHS ${DALI_HEADERS_CUSTOM_PATHS} NO_DEFAULT_PATH)
else()
    find_path(DALI_CONFIG_PATH "dali/config.h")
endif()

if (NOT DALI_CONFIG_PATH)
    if (Dali_FIND_REQUIRED)
        message(FATAL_ERROR "${Red} Failed to find Dali headers  ${ColorReset}" ${REASON_MSG} ${ARGN})
    else()
        message(WARNING "Failed to find Dali headers" ${REASON_MSG} ${ARGN})
    endif(Dali_FIND_REQUIRED)

    set(DALI_FOUND FALSE)
else()
    list(APPEND DALI_INCLUDE_DIRS ${DALI_CONFIG_PATH})

    list(APPEND DALI_AND_DEPS_INCLUDE_DIRS ${DALI_INCLUDE_DIRS})
endif(NOT DALI_CONFIG_PATH)

if(DALI_FOUND)
    MARK_AS_ADVANCED(DALI_INCLUDE_DIRS
                     DALI_LIBRARIES
                     DALI_AND_DEPS_INCLUDE_DIRS
                     DALI_AND_DEPS_LIBRARIES)
endif(DALI_FOUND)
