#include "StackedGatedModel.h"

DEFINE_double(memory_penalty, 0.3, "L1 Penalty on Input Gate activation.");

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;

template<typename Z>
vector<Mat<Z>> StackedGatedModel<Z>::parameters() const {
    auto parameters = StackedModel<Z>::parameters();
    auto gate_params = gate.parameters();
    parameters.insert(parameters.end(), gate_params.begin(), gate_params.end());
    return parameters;
}

/**
Configuration
-------------

Return a map with keys corresponding to hyperparameters for
the model and where values are vectors of strings containing
the assignments to each hyperparameter for the loaded model.

Useful for saving the model to file and reloading it later.

Outputs
-------

std::map<std::string, std::vector< std::string >> config : configuration map

**/

template<typename Z>
typename StackedGatedModel<Z>::config_t StackedGatedModel<Z>::configuration() const  {
    auto config = StackedModel<Z>::configuration();
    config["memory_penalty"].emplace_back(to_string(memory_penalty));
    return config;
}

template<typename Z>
StackedGatedModel<Z> StackedGatedModel<Z>::build_from_CLI(
        string load_location,
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
                  << model.input_size
                  << std::endl
                  << "Output size           = "
                  << model.output_size
                  << std::endl
                  << "Stack size            = "
                  << model.stack_size
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
/**
Load
----

Load a saved copy of this model from a directory containing the
configuration file named "config.md", and from ".npy" saves of
the model parameters in the same directory.

Inputs
------

std::string dirname : directory where the model is currently saved

Outputs
-------

StackedGatedModel<Z> model : the saved model

**/
template<typename Z>
StackedGatedModel<Z> StackedGatedModel<Z>::load(std::string dirname) {
    // fname should be a directory:
    utils::ensure_directory(dirname);
    // load the configuration file
    auto config_name = dirname + "config.md";

    auto config = utils::text_to_map(config_name);

    utils::assert_map_has_key(config, "input_size");
    utils::assert_map_has_key(config, "hidden_sizes");
    utils::assert_map_has_key(config, "vocabulary_size");
    utils::assert_map_has_key(config, "memory_penalty");
    utils::assert_map_has_key(config, "output_size");

    // construct the model using the map
    auto model =  StackedGatedModel<Z>(config);

    // get the current parameters of the model.
    auto params = model.parameters();

    // get the new parameters from the saved numpy files
    utils::load_matrices(params, dirname);

    return model;
}

template<typename Z>
std::tuple<Z, Z> StackedGatedModel<Z>::masked_predict_cost(
    shared_index_mat data,
    shared_index_mat target_data,
    shared_eigen_index_vector start_loss,
    shared_eigen_index_vector codelens,
    uint offset,
    Z drop_prob) {

    auto initial_state    = this->initial_states();

    mat input_vector;
    mat memory;
    mat logprobs;
    std::tuple<Z, Z> cost(0.0, 0.0);

    auto n = data->cols();
    for (uint i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector = this->embedding.rows_pluck(data->col(i));
        memory = gate.activate(input_vector, initial_state.back().hidden );
        input_vector = input_vector.eltmul_broadcast_rowwise(memory);
        // pass this letter to the LSTM for processing
        initial_state = this->stacked_lstm->activate(initial_state, input_vector, drop_prob);
        // classifier takes as input the final hidden layer's activation:
        vector<Mat<Z>> hiddens = LSTM<Z>::State::hiddens(initial_state);
        logprobs      = this->decoder->activate(input_vector, hiddens);

        if (graph::backprop_enabled) {
            std::get<0>(cost) += masked_cross_entropy(
                logprobs,
                i,
                start_loss,
                codelens,
                (target_data->col(i+1).array() - offset).matrix()
            );
            std::get<1>(cost) += masked_sum(
                memory,
                i,
                0,
                start_loss,
                memory_penalty
            );
        } else {
            std::get<0>(cost) += masked_cross_entropy_no_grad(
                logprobs,
                i,
                start_loss,
                codelens,
                (target_data->col(i+1).array() - offset).matrix()
            );
            std::get<1>(cost) += masked_sum_no_grad(
                memory,
                i,
                0,
                start_loss,
                memory_penalty
            );
        }
    }
    return cost;
}

template<typename Z>
std::tuple<Z, Z> StackedGatedModel<Z>::masked_predict_cost(
    shared_index_mat data,
    shared_index_mat target_data,
    uint start_loss,
    shared_eigen_index_vector codelens,
    uint offset,
    Z drop_prob) {

    auto initial_state    = this->initial_states();
    mat input_vector;
    mat memory;
    mat logprobs;
    std::tuple<Z, Z> cost(0.0, 0.0);

    auto n = data->cols();
    for (uint i = 0; i < n-1; ++i) {
            // pick this letter from the embedding
            input_vector = this->embedding.rows_pluck(data->col(i));
            memory = gate.activate(
                input_vector,
                initial_state.back().hidden
            );
            input_vector = input_vector.eltmul_broadcast_rowwise(memory);
            // pass this letter to the LSTM for processing
            initial_state = this->stacked_lstm->activate(
                initial_state,
                input_vector,
                drop_prob
            );
            // classifier takes as input the final hidden layer's activation:
            vector<Mat<Z>> hiddens = LSTM<Z>::State::hiddens(initial_state);
            logprobs      = this->decoder->activate(input_vector, hiddens);
            if (graph::backprop_enabled) {
                std::get<0>(cost) += masked_cross_entropy(
                    logprobs,
                    i,
                    start_loss,
                    codelens,
                    (target_data->col(i+1).array() - offset).matrix()
                );
                std::get<1>(cost) += masked_sum(
                    memory,
                    i,
                    0,
                    start_loss,
                    memory_penalty
                );
            } else {
                std::get<0>(cost) += masked_cross_entropy_no_grad(
                    logprobs,
                    i,
                    start_loss,
                    codelens,
                    (target_data->col(i+1).array() - offset).matrix()
                );
                std::get<1>(cost) += masked_sum_no_grad(
                    memory,
                    i,
                    0,
                    start_loss,
                    memory_penalty
                );
            }
    }
    return cost;
}

template<typename Z>
StackedGatedModel<Z>::StackedGatedModel (
    int vocabulary_size,
    int input_size,
    int hidden_size,
    int stack_size,
    int output_size,
    bool use_shortcut,
    bool memory_feeds_gates,
    Z _memory_penalty)
      : StackedModel<Z>(
        vocabulary_size,
        input_size,
        hidden_size,
        stack_size,
        output_size,
        use_shortcut,
        memory_feeds_gates),
      memory_penalty(_memory_penalty),
      gate(input_size, hidden_size) {
}

using utils::from_string;

/**
StackedGatedModel Constructor from configuration map
----------------------------------------------------

Construct a model from a map of configuration parameters.
Useful for reinitializing a model that was saved to a file
using the `utils::file_to_map` function to obtain a map of
configurations.

Inputs
------

std::map<std::string, std::vector<std::string>& config : model hyperparameters

**/
template<typename Z>
StackedGatedModel<Z>::StackedGatedModel (
      const typename StackedGatedModel<Z>::config_t& config) :
    memory_penalty(from_string<Z>(config.at("memory_penalty")[0])),
    StackedModel<Z>(config),
    gate(
        from_string<int>(
          config.at("input_size")[0]),
        from_string<int>(
          config.at("hidden_sizes")[config.at("hidden_sizes").size()-1]
        )
    ) {
}

template<typename Z>
StackedGatedModel<Z>::StackedGatedModel (
  int vocabulary_size,
  int input_size,
  int output_size,
  std::vector<int>& hidden_sizes,
  bool use_shortcut,
  bool memory_feeds_gates,
  Z _memory_penalty)
        :
        StackedModel<Z>(
            vocabulary_size,
            input_size,
            output_size,
            hidden_sizes,
            use_shortcut,
            memory_feeds_gates),
        memory_penalty(_memory_penalty),
        gate(input_size, hidden_sizes[0]) {
}

/**
StackedGatedModel<Z>::StackedGatedModel
-------------

Copy constructor with option to make a shallow
or deep copy of the underlying parameters.

If the copy is shallow then the parameters are shared
but separate gradients `dw` are used for each of
thread StackedGatedModel<Z>.

Shallow copies are useful for Hogwild and multithreaded
training

See `Mat<Z>::shallow_copy`, `examples/character_prediction.cpp`,
`StackedGatedModel<Z>::shallow_copy`

Inputs
------

      StackedGatedModel<Z> l : StackedGatedModel from which to source parameters and dw
    bool copy_w : whether parameters for new StackedGatedModel should be copies
                  or shared
   bool copy_dw : whether gradients for new StackedGatedModel should be copies
                  shared (Note: sharing `dw` should be used with
                  caution and can lead to unpredictable behavior
                  during optimization).

Outputs
-------

StackedGatedModel<Z> out : the copied StackedGatedModel with deep or shallow copy of parameters

**/
template<typename Z>
StackedGatedModel<Z>::StackedGatedModel (const StackedGatedModel<Z>& model, bool copy_w, bool copy_dw) :
    StackedModel<Z>(model, copy_w, copy_dw),
        memory_penalty(model.memory_penalty),
        gate(model.gate, copy_w, copy_dw) {}

/**
Shallow Copy
------------

Perform a shallow copy of a StackedGatedModel<Z> that has
the same parameters but separate gradients `dw`
for each of its parameters.

Shallow copies are useful for Hogwild and multithreaded
training

See `StackedGatedModel<Z>::shallow_copy`, `examples/character_prediction.cpp`.

Outputs
-------

StackedGatedModel<Z> out : the copied layer with sharing parameters,
                           but with separate gradients `dw`

**/
template<typename Z>
StackedGatedModel<Z> StackedGatedModel<Z>::shallow_copy() const {
    return StackedGatedModel<Z>(*this, false, true);
}

template<typename Z>
typename StackedGatedModel<Z>::state_type StackedGatedModel<Z>::get_final_activation(
        Indexing::Index example,
        Z drop_prob) const {
    mat input_vector;
    mat memory;
    auto initial_state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = this->embedding.row_pluck(example[i]);
        memory        = gate.activate(
            input_vector,
            initial_state.back().hidden
        );
        input_vector  = input_vector.eltmul_broadcast_rowwise(memory);
        // pass this letter to the LSTM for processing
        initial_state = this->stacked_lstm->activate(initial_state, input_vector);
        // decoder takes as input the final hidden layer's activation:
    }
    return initial_state;
}

/**
Activate
--------

Run Stacked Gated Model by 1 timestep by observing
the element from embedding with index `index`
and report the activation, cell, and hidden
states

Inputs
------

std::pair<std::vector<std::shared_ptr<Mat<Z>>>, std::vector<std::shared_ptr<Mat<Z>>>>& : previous state
uint index : embedding observation

Outputs
-------

std::tuple<std::pair<vector<shared_ptr<Mat<Z>>>, vector<shared_ptr<Mat<Z>>>>, shared_ptr<Mat<Z>>, Z> out :
    pair of LSTM hidden and cell states, and probabilities from the decoder, and memory usage.

**/
template<typename Z>
typename StackedGatedModel<Z>::activation_t StackedGatedModel<Z>::activate(
        state_type& previous_state,
        const uint& index) const {
    activation_t out;

    auto input_vector = this->embedding.row_pluck(index);
    auto memory       = gate.activate(
        input_vector,
        previous_state.back().hidden
    );
    input_vector      = input_vector.eltmul_broadcast_rowwise(memory);

    std::get<0>(out) = this->stacked_lstm->activate(
        previous_state,
        input_vector
    );
    std::get<1>(out) = MatOps<Z>::softmax_no_grad(
        this->decoder->activate(
            input_vector,
            LSTM<Z>::State::hiddens(std::get<0>(out))
        )
    );
    std::get<2>(out) = memory;
    return out;
}

template<typename Z>
typename StackedGatedModel<Z>::activation_t StackedGatedModel<Z>::activate(
        state_type& previous_state,
        const eigen_index_block indices) const {
    activation_t out;
    auto input_vector = this->embedding.rows_pluck(indices);
    auto memory       = gate.activate(
        input_vector,
        previous_state.back().hidden
    );
    input_vector      = input_vector.eltmul_broadcast_rowwise(memory);

    std::get<0>(out) = this->stacked_lstm->activate(
        previous_state,
        input_vector);
    std::get<1>(out) = MatOps<Z>::softmax_no_grad(
        this->decoder->activate(
            input_vector,
            LSTM<Z>::State::hiddens(std::get<0>(out))
        )
    );
    std::get<2>(out) = memory;

    return out;
}

template<typename Z>
std::vector<int> StackedGatedModel<Z>::reconstruct(
        Indexing::Index example,
        int eval_steps,
        int symbol_offset) const {

    graph::NoBackprop nb;
    auto initial_state = get_final_activation(example);

    mat input_vector;
    mat memory;
    vector<int> outputs;
    auto last_symbol = this->decoder->activate(
      input_vector,
      LSTM<Z>::State::hiddens(initial_state)
    ).argmax();
    outputs.emplace_back(last_symbol);
    last_symbol += symbol_offset;
    for (uint j = 0; j < eval_steps - 1; j++) {
            input_vector  = this->embedding.row_pluck(last_symbol);
            memory        = gate.activate(
                input_vector,
                initial_state.back().hidden
            );
            input_vector  = input_vector.eltmul_broadcast_rowwise(memory);
            initial_state = this->stacked_lstm->activate(
                initial_state,
                input_vector);
            last_symbol   = this->decoder->activate(
                input_vector,
                LSTM<Z>::State::hiddens(initial_state)
            ).argmax();
            outputs.emplace_back(last_symbol);
            last_symbol += symbol_offset;
    }
    return outputs;
}

template<typename Z>
std::vector<utils::OntologyBranch::shared_branch> StackedGatedModel<Z>::reconstruct_lattice(
        Indexing::Index example,
        utils::OntologyBranch::shared_branch root,
        int eval_steps) const {
    graph::NoBackprop nb;
    mat input_vector;
    mat memory;
    auto pos = root;
    auto initial_state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = this->embedding.row_pluck(example[i]);
        memory        = gate.activate(input_vector, initial_state.back().hidden);
        input_vector  = input_vector.eltmul_broadcast_rowwise(memory);
        // pass this letter to the LSTM for processing
        initial_state = this->stacked_lstm->activate(initial_state, input_vector);
        // decoder takes as input the final hidden layer's activation:
    }
    vector<utils::OntologyBranch::shared_branch> outputs;
    // Take the argmax over the available options (0 for go back to
    // root, and 1..n for the different children of the current position)
    auto last_turn = this->decoder->activate(
        input_vector,
        LSTM<Z>::State::hiddens(initial_state)
    ).argmax_slice(
        0,
        pos->children.size() + 1
    );
    // if the turn is 0 go back to root, else go to one of the children using
    // the lattice pointers:
    pos = (last_turn == 0) ? root : pos->children[last_turn-1];
    // add this decision to the output :
    outputs.emplace_back(pos);
    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = this->embedding.row_pluck(pos->id);
        memory        = gate.activate(input_vector, initial_state.back().hidden);
        input_vector  = input_vector.eltmul_broadcast_rowwise(memory);
        initial_state = this->stacked_lstm->activate(initial_state, input_vector);
        last_turn     = this->decoder->activate(
            input_vector,
            LSTM<Z>::State::hiddens(initial_state)
        ).argmax_slice(
            0,
            pos->children.size() + 1
        );
        pos = (last_turn == 0) ? root : pos->children[last_turn-1];
        outputs.emplace_back(pos);
    }
    return outputs;
}

template class StackedGatedModel<float>;
template class StackedGatedModel<double>;
