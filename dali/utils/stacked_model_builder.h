#include <gflags/gflags.h>

#include <string>

#include "dali/models/StackedModel.h"
#include "dali/models/StackedGatedModel.h"


DEFINE_int32(stack_size,        4,    "How many LSTMs should I stack ?");
DEFINE_int32(input_size,        100,  "Size of the word vectors");
DEFINE_int32(hidden,            100,  "How many Cells and Hidden Units should each LSTM have ?");
DEFINE_double(decay_rate,       0.95, "What decay rate should RMSProp use ?");
DEFINE_double(rho,              0.95, "What rho / learning rate should the Solver use ?");
DEFINE_bool(shortcut,           true, "Use a Stacked LSTM with shortcuts");
DEFINE_bool(memory_feeds_gates, true, "LSTM's memory cell also control gate outputs");
DEFINE_double(memory_penalty, 0.3, "L1 Penalty on Input Gate activation.");


template<typename Z>
StackedModel<Z> stacked_model_from_CLI(
        std::string load_location,
        int vocab_size,
        int output_size,
        bool verbose) {
    if (verbose)
        std::cout << "Load location         = " << ((load_location == "") ? "N/A" : load_location) << std::endl;
    // Load or Construct the model
    auto model = (load_location != "") ?
        StackedModel<Z>::load(load_location) :
        StackedModel<Z>(
                vocab_size,
                FLAGS_input_size,
                FLAGS_hidden,
                FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
                output_size,
                FLAGS_shortcut,
                FLAGS_memory_feeds_gates);
    if (verbose) {
        std::cout << (
                    (load_location == "") ?
                        "Constructed Stacked LSTMs" :
                        "Loaded Model"
                    )
                  << std::endl
                  << "Vocabulary size       = "
                  << model.embedding.dims(0)
                  << std::endl
                  << "Input size            = "
                  << model.embedding.dims(1)
                  << std::endl
                  << "Output size           = "
                  << model.output_size
                  << std::endl
                  << "Stack size            = "
                  << model.hidden_sizes.size()
                  << std::endl
                  << "Shortcut connections  = "
                  << (model.use_shortcut ? "true" : "false")
                  << std::endl
                  << "Memory feeds gates    = "
                  << (model.memory_feeds_gates ? "true" : "false")
                  << std::endl;
    }
    return model;
}


template<typename Z>
StackedGatedModel<Z> stacked_gated_model_from_CLI(
        std::string load_location,
        int vocab_size,
        int output_size,
        bool verbose) {
    if (verbose)
        std::cout << "Load location         = "
                  << ((load_location == "") ? "N/A" : load_location)
                  << std::endl;
    // Load or Construct the model
    auto model = (load_location != "") ?
        StackedGatedModel<Z>::load(load_location) :
        StackedGatedModel<Z>(
            vocab_size,
            FLAGS_input_size,
            FLAGS_hidden,
            FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
            output_size,
            FLAGS_shortcut,
            FLAGS_memory_feeds_gates,
            FLAGS_memory_penalty);
    if (verbose) {
        std::cout << (
                    (load_location == "") ?
                        "Constructed Stacked LSTMs" :
                        "Loaded Model"
                    )
                  << std::endl
                  << "Vocabulary size       = "
                  << model.embedding.dims(0)
                  << std::endl
                  << "Input size            = "
                  << model.embedding.dims(1)
                  << std::endl
                  << "Output size           = "
                  << model.output_size
                  << std::endl
                  << "Stack size            = "
                  << model.hidden_sizes.size()
                  << std::endl
                  << "Shortcut connections  = "
                  << (model.use_shortcut ? "true" : "false")
                  << std::endl
                  << "Memory feeds gates    = "
                  << (model.memory_feeds_gates ? "true" : "false")
                  << std::endl;
    }
    return model;
}
