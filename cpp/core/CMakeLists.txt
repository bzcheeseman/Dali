set(CoreDir "${PROJECT_SOURCE_DIR}/core")

file(GLOB CoreHeaders "${CoreDir}/*.h")
file(GLOB CoreSources "${CoreDir}/*.cpp")
file(GLOB CoreTemplates "${CoreDir}/*.cpp.template")


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

ADD_CUSTOM_TARGET(fix
    COMMAND sudo sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build
)

message("here are the libraries you were looking for => " ${MKL_LIBRARIES})
message("here are the BLAS libraries you were looking for => " ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})

add_dependencies(core core_templates)

target_link_libraries(core ${ZLIB_LIBRARIES} proto ${BLAS_LIBRARIES} ${MKL_LIBRARIES})
