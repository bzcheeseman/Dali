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
file(GLOB_RECURSE DaliTests             "${DaliDir}/tests.cpp")



set(excluded_folders "${DaliDir}/tensor;${DaliDir}/data_processing;${DaliDir}/execution;${DaliDir}/layers;${DaliDir}/math")

foreach(excluded_folder ${excluded_folders})
    file(GLOB_RECURSE ExcludedDaliHeaders           "${excluded_folder}/*.h")
    file(GLOB_RECURSE ExcludedDaliSources           "${excluded_folder}/*.cpp")
    file(GLOB_RECURSE ExcludedDaliTemplates         "${excluded_folder}/*.cpp.template")
    file(GLOB_RECURSE ExcludedDaliMaybeCudaSources  "${excluded_folder}/*.cu.cpp")
    file(GLOB_RECURSE ExcludedDaliCudaSources       "${excluded_folder}/*.cu")
    file(GLOB_RECURSE ExcludedDaliTests             "${excluded_folder}/tests.cpp")

    foreach(file ${ExcludedDaliHeaders})
        list(REMOVE_ITEM DaliHeaders ${file})
    endforeach()
    foreach(file ${ExcludedDaliSources})
        list(REMOVE_ITEM DaliSources ${file})
    endforeach()
    foreach(file ${ExcludedDaliTemplates})
        list(REMOVE_ITEM DaliTemplates ${file})
    endforeach()
    foreach(file ${ExcludedDaliMaybeCudaSources})
        list(REMOVE_ITEM DaliMaybeCudaSources ${file})
    endforeach()
    foreach(file ${ExcludedDaliCudaSources})
        list(REMOVE_ITEM DaliCudaSources ${file})
    endforeach()
    foreach(file ${ExcludedDaliTests})
        list(REMOVE_ITEM DaliTests ${file})
    endforeach()
endforeach()


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
    target_link_libraries(dali_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    if (OpenBLAS_FOUND)
        target_link_libraries(dali_cuda ${OpenBLAS_LIB})
    endif(OpenBLAS_FOUND)

    # Apple has trouble performing static linking between dali and dali_cuda
    # to make the task easier we do the linking, while apple reaps the profits
    # thanks.
    IF (APPLE)
        INSTALL(TARGETS dali_cuda DESTINATION lib)
    ENDIF (APPLE)

    SET(CMAKE_CXX_FLAGS ${OLD_CXX_FLAGS})
    SET(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif(WITH_CUDA)

# todo: move file with add_custom_command

# Remove tests from compilation
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
                           ${CMAKE_THREAD_LIBS_INIT}
                           ${OpenBLAS_LIB})

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
install(DIRECTORY ${DaliDir} ${GENERATED_FILES_DIR}/dali DESTINATION include
        FILES_MATCHING PATTERN "*.h")
# This potentially does not work on Apple, check out:
# http://stackoverflow.com/questions/10765885/how-to-install-your-custom-cmake-find-module
install(FILES ${PROJECT_SOURCE_DIR}/cmake/DaliConfig.cmake   DESTINATION lib/dali/cmake)
install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindOpenBlas.cmake DESTINATION lib/openblas/cmake RENAME OpenBlasConfig.cmake)

add_testsuite("test_dali" "dali" "${DaliTests}")
