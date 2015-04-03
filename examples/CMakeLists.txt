
set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
set(EXAMPLES_BUILD_DIR ${PROJECT_SOURCE_DIR}/build/examples)
set(EXAMPLE_PROGRAMS api_showcase
                     babi_solvers
                     bidirectional_sentiment
                     character_prediction
                     grid_search_simple
                     language_model
                     language_model_from_senti
                     lattice_prediction
                     lattice_prediction_from_protobuff
                     loading_from_sqlite
                     lstm_sentiment
                     mlbasics_learn_to_add
                     mlbasics_rnn_binary_addition
                     sparkfun_prediction
                     stacked_lstm
                     test_lattice
                     )
foreach(program ${EXAMPLE_PROGRAMS})
    add_executable(${program} ${EXAMPLES_DIR}/${program}.cpp)
    target_link_libraries(${program} dali)
    if (MKL_FOUND AND BLAS_FOUND)
        add_custom_command(TARGET ${program}
            COMMAND sh ${PROJECT_SOURCE_DIR}/scripts/fix_dylib.sh ${EXAMPLES_BUILD_DIR}/${program}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
    endif (MKL_FOUND AND BLAS_FOUND)
endforeach(program)


# If custom libraries are needed for some targets add them below
# like this:
#     target_link_libraries(character_prediction SPECIAL_DEPENDENCY)
