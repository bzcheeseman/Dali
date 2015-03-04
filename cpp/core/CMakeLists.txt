set(CoreDir "${PROJECT_SOURCE_DIR}/core")

file(GLOB_RECURSE CoreHeaders "${CoreDir}/*.h")
file(GLOB_RECURSE CoreSources "${CoreDir}/*.cpp")
file(GLOB_RECURSE CoreTemplates "${CoreDir}/*.cpp.template")

# Remove tests from compilation
file(GLOB CoreTestFile "${CoreDir}/tests.cpp")
list(REMOVE_ITEM CoreSources ${CoreTestFile})

foreach(header ${CoreTemplates})
    message(STATUS "Template instantiator will run for ${header}")
    ADD_CUSTOM_TARGET(core_templates
        COMMAND ${PROJECT_SOURCE_DIR}/scripts/python_template_instantiator.sh ${header}
        WORKING_DIRECTORY ${CoreDir}
        DEPENDS ${header})
endforeach(header)

add_library(core
        ${CoreSources}
        ${CoreHeaders}
)

add_custom_target(fix_libiomp5
	COMMAND sudo sh ${PROJECT_SOURCE_DIR}/scripts/fix_libiomp5.sh
)

add_dependencies(core core_templates proto)

if (BLAS_FOUND AND MKL_FOUND)
    add_definitions(-DEIGEN_USE_MKL_VML
                    -DEIGEN_USE_BLAS)
    add_dependencies(core fix_libiomp5)
endif (BLAS_FOUND AND MKL_FOUND)

target_link_libraries(core ${ZLIB_LIBRARIES}
                           proto
                           ${BLAS_LIBRARIES}
                           ${MKL_LIBRARIES}
                           ${GFLAGS_LIBRARIES}
                           ${GPERFTOOLS_LIBRARIES})



add_testsuite("test_core" ${CoreTestFile} "core")
