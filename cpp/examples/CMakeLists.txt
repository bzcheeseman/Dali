set(EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples)

ADD_EXECUTABLE(character_prediction "${EXAMPLES_DIR}/character_prediction.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(character_prediction core)

ADD_EXECUTABLE(stacked_lstm "${EXAMPLES_DIR}/stacked_lstm.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(stacked_lstm core)

ADD_EXECUTABLE(sparkfun_prediction "${EXAMPLES_DIR}/sparkfun_prediction.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(sparkfun_prediction core)

ADD_EXECUTABLE(language_sentiment "${EXAMPLES_DIR}/language_model_from_senti.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(language_sentiment core)

ADD_EXECUTABLE(language_model "${EXAMPLES_DIR}/language_model.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(language_model core)

ADD_EXECUTABLE(test_lattice "${EXAMPLES_DIR}/test_lattice.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(test_lattice core)

ADD_EXECUTABLE(lattice_prediction "${EXAMPLES_DIR}/lattice_prediction.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(lattice_prediction core)

ADD_EXECUTABLE(lattice_prediction_protobuff "${EXAMPLES_DIR}/lattice_prediction.cpp" "${PROJECT_SOURCE_DIR}/OptionParser/OptionParser.cpp")
target_link_libraries(lattice_prediction_protobuff core)
