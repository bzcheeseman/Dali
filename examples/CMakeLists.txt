
set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)
set(EXAMPLES_BUILD_DIR ${PROJECT_SOURCE_DIR}/build/examples)
set(EXAMPLE_PROGRAMS api_showcase
                     arithmetic
                     babi_solvers
                     beam_tree_training
                     bidirectional_sentiment
                     character_prediction
                     grid_search_simple
                     language_model
                     language_model_from_senti
                     lattice_prediction
                     lattice_prediction_from_protobuff
                     loading_from_sqlite
                     lstm_sentiment
                     machine_comprehension
                     mlbasics_learn_to_add
                     mlbasics_rnn_binary_addition
                     sparkfun_prediction
                     sparse_lstm_sentiment
                     sparse_ner
                     sparse_paraphrase
                     visualizer
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

# cuda_compile(mshadow_cuda)
# cuda_add_library(zeugma ${mshadow_cuda})
# target_link_libraries(zeugma ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cuda cudart)

# add_executable(hello_word ${UniSources})
# target_link_libraries(hello_word zeugma)

SET(OLD_CXX_FLAGS ${CMAKE_CXX_FLAGS})
SET(CMAKE_CXX_FLAGS "-O3 -w -fPIC ${CUDA_EXTRA_FLAGS}")
set(CMAKE_LEGACY_CYGWIN_WIN32 0)

cuda_compile(mycuda ${EXAMPLES_DIR}/mshadow.cu)
cuda_add_executable(mycuda ${EXAMPLES_DIR}/mshadow.cu)
target_link_libraries(mycuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cuda cudart)

SET(CMAKE_CXX_FLAGS ${OLD_CXX_FLAGS})




# If custom libraries are needed for some targets add them below
# like this:
#     target_link_libraries(character_prediction SPECIAL_DEPENDENCY)
