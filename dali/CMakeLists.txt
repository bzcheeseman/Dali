set(DaliDir "${PROJECT_SOURCE_DIR}/dali")

set(GENERATED_FILES_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_FILES_DIR})
include_directories(${GENERATED_FILES_DIR})

configure_file("${PROJECT_SOURCE_DIR}/dali/config.h.in" "${GENERATED_FILES_DIR}/dali/config.h")

file(GLOB_RECURSE DaliHeaders           "${DaliDir}/*.h")
file(GLOB_RECURSE DaliSources           "${DaliDir}/*.cpp")
file(GLOB_RECURSE DaliTemplates         "${DaliDir}/*.cpp.template")
file(GLOB_RECURSE DaliMaybeCudaSources  "${DaliDir}/*.cu.cpp")
file(GLOB_RECURSE DaliCudaSources       "${DaliDir}/*.cu")

if (WITH_CUDA)
    foreach(DaliCudaFile ${DaliMaybeCudaSources})
        file(RELATIVE_PATH DaliCudaFilePath "${DaliDir}" "${DaliCudaFile}")
        set(NEW_FILENAME "${CMAKE_BINARY_DIR}/cuda_files/${DaliCudaFilePath}.cu")
        ADD_CUSTOM_COMMAND(OUTPUT ${NEW_FILENAME}
                COMMAND ${CMAKE_COMMAND} -E copy ${DaliCudaFile} ${NEW_FILENAME}
                WORKING_DIRECTORY ${DaliDir}
                DEPENDS ${DaliCudaFile})
        list(REMOVE_ITEM DaliSources ${DaliCudaFile})
        list(APPEND DaliCudaSources ${NEW_FILENAME})
    endforeach(DaliCudaFile)
    # also remove sources from C compilation

    SET(OLD_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    SET(CMAKE_CXX_FLAGS "-O3 -w -fPIC ${CUDA_EXTRA_FLAGS}")

    cuda_add_library(dali_cuda STATIC ${DaliCudaSources})
    target_link_libraries(dali_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} openblas)

    # Apple has trouble performing static linking between dali and dali_cuda
    # to make the task easier we do the linking, while apple reaps the profits
    # thanks.
    IF (APPLE)
        INSTALL(TARGETS dali_cuda DESTINATION lib)
    ENDIF (APPLE)

    # message("${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY}")

    SET(CMAKE_CXX_FLAGS ${OLD_CXX_FLAGS})
    SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif(WITH_CUDA)

# todo: move file with add_custom_command

# Remove tests from compilation
file(GLOB_RECURSE DaliTests "${DaliDir}/tests.cpp")
list(REMOVE_ITEM DaliSources ${DaliTests} ${TESTS_MAIN})
foreach(header ${DaliTemplates})
    message(STATUS "Template instantiator will run for ${header}")
    ADD_CUSTOM_TARGET(dali_templates
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/python_template_instantiator.sh ${header}
        WORKING_DIRECTORY ${DaliDir}
        DEPENDS ${header})
endforeach(header)

add_library(dali ${MAYBE_SHARED}
        ${DaliSources}
        ${DaliHeaders}
)

add_custom_target(fix_libiomp5
	COMMAND sudo sh ${PROJECT_SOURCE_DIR}/scripts/fix_libiomp5.sh
)

if(dali_templates)
    add_dependencies(dali dali_templates)
endif(dali_templates)

if (BLAS_FOUND AND MKL_FOUND)
    add_dependencies(dali fix_libiomp5)
endif (BLAS_FOUND AND MKL_FOUND)

target_link_libraries(dali ${ZLIB_LIBRARIES}
                           ${BLAS_LIBRARIES}
                           ${MKL_LIBRARIES}
                           ${GFLAGS_LIBRARIES}
                           ${CMAKE_THREAD_LIBS_INIT})

if (GPERFTOOLS_FOUND)
    target_link_libraries(dali ${GPERFTOOLS_LIBRARIES})
endif (GPERFTOOLS_FOUND)

if (WITH_CUDA)
    IF (APPLE)
        target_link_libraries(dali -force_load dali_cuda)
    ELSE (APPLE)
        target_link_libraries(dali "-Wl,--whole-archive")
        target_link_libraries(dali dali_cuda)
        target_link_libraries(dali "-Wl,--no-whole-archive")
    ENDIF(APPLE)
endif (WITH_CUDA)

INSTALL(TARGETS dali DESTINATION lib)
install(DIRECTORY ${DaliDir} ${GENERATED_FILES_DIR}/dali  DESTINATION include
        FILES_MATCHING PATTERN "*.h")

add_testsuite("test_dali" "dali" "${DaliTests}")
