set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
set(EXAMPLE_PROGRAMS character_prediction
                     stacked_lstm
                     sparkfun_prediction
                     language_model_from_senti
                     language_model
                     test_lattice
                     lattice_prediction
                     lattice_prediction_from_protobuff
                     )

foreach(program ${EXAMPLE_PROGRAMS})
    add_executable(${program} ${EXAMPLES_DIR}/${program}.cpp)
    target_link_libraries(${program} core)
    if (MKL_FOUND AND BLAS_FOUND)
        add_custom_command(TARGET ${program}
            COMMAND /bin/sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh ${program}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
    endif (MKL_FOUND AND BLAS_FOUND)
endforeach(program)

# If custom libraries are needed for some targets add them below
# like this:
#     target_link_libraries(character_prediction SPECIAL_DEPENDENCY)
