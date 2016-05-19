# Script inspired by Caffe's version

function(detect_cudnn)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")
    set(CUDNN_FOUND FALSE PARENT_SCOPE)

    find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory." )

    get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
    find_library(CUDNN_LIBRARY NAMES libcudnn.so # libcudnn_static.a
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist}
            DOC "Path to cuDNN library.")

    if (CUDNN_INCLUDE AND CUDNN_LIBRARY)
        file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

        # cuDNN v3 and beyond
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

        if(NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "???")
        else()
            set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        endif()


        string(COMPARE EQUAL "${CUDNN_VERSION_MAJOR}" 5 cuDNNVersionCompatible)

        if(cuDNNVersionCompatible)
            message(STATUS "Found cuDNN v${CUDNN_VERSION} (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY}).")
            set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
            set(CUDNN_FOUND   TRUE               PARENT_SCOPE)
            mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)
        else()
            message(WARNING "cuDNN v5 is required. Found cuDNN v${CUDNN_VERSION_MAJOR} (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY}).")
        endif()

    else()
        message(WARNING "cudnn not found")
    endif()
endfunction()
