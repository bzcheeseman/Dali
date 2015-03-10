#include "StackedGatedModel.h"

DEFINE_double(memory_penalty, 0.3, "L1 Penalty on Input Gate activation.");

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;

template<typename R>
vector<SHARED_MAT> StackedGatedModel<R>::parameters() const {
  auto parameters = StackedModel<R>::parameters();
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

template<typename R>
typename StackedGatedModel<R>::config_t StackedGatedModel<R>::configuration() const  {
    auto config = StackedModel<R>::configuration();
    config["memory_penalty"].emplace_back(to_string(memory_penalty));
    return config;
}

template<typename R>
StackedGatedModel<R> StackedGatedModel<R>::build_from_CLI(string load_location,
                                                          int vocab_size,
                                                          int output_size,
                                                          bool verbose) {
        if (verbose)
          std::cout << "Load location         = " << ((load_location == "") ? "N/A" : load_location)  << std::endl;
        // Load or Construct the model
        auto model = (load_location != "") ?
                StackedGatedModel<R>::load(load_location) :
                StackedGatedModel<R>(
                        vocab_size,
                        FLAGS_input_size,
                        FLAGS_hidden,
                        FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
                        output_size,
                        FLAGS_memory_penalty);
        if (verbose) {
                std::cout << ((load_location == "") ? "Constructed Stacked LSTMs" : "Loaded Model") << std::endl;
                std::cout << "Vocabulary size       = " << model.embedding->dims[0] << std::endl;
                std::cout << "Input size            = " << model.input_size         << std::endl;
                std::cout << "Output size           = " << model.output_size        << std::endl;
                std::cout << "Stack size            = " << model.stack_size         << std::endl;
                std::cout << "Memory Penalty        = " << model.memory_penalty     << std::endl;
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

StackedGatedModel<R> model : the saved model

**/
template<typename R>
StackedGatedModel<R> StackedGatedModel<R>::load(std::string dirname) {
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
        auto model =  StackedGatedModel<R>(config);

        // get the current parameters of the model.
        auto params = model.parameters();

        // get the new parameters from the saved numpy files
        utils::load_matrices(params, dirname);

        return model;
}

template<typename R>
std::tuple<R, R> StackedGatedModel<R>::masked_predict_cost(
        shared_index_mat data,
        shared_index_mat target_data,
        shared_eigen_index_vector start_loss,
        shared_eigen_index_vector codelens,
        uint offset,
        R drop_prob) {

        auto initial_state    = this->initial_states();

        SHARED_MAT input_vector;
        SHARED_MAT memory;
        SHARED_MAT logprobs;
        // SHARED_MAT probs;
        std::tuple<R, R> cost(0.0, 0.0);

        auto n = data->cols();
        for (uint i = 0; i < n-1; ++i) {
                // pick this letter from the embedding
                input_vector = this->embedding->rows_pluck(data->col(i));
                memory = gate.activate(input_vector, initial_state.second[0]);
                input_vector = input_vector->eltmul_broadcast_rowwise(memory);
                // pass this letter to the LSTM for processing
                initial_state = this->stacked_lstm->activate(initial_state, input_vector, drop_prob);
                // classifier takes as input the final hidden layer's activation:
                logprobs      = this->decoder->activate(input_vector, initial_state.second);

                std::get<0>(cost) += graph::backprop_enabled ? masked_cross_entropy(
                                                                      logprobs,
                                                                      i,
                                                                      start_loss,
                                                                      codelens,
                                                                      (target_data->col(i+1).array() - offset).matrix()) :
                                                        masked_cross_entropy_no_grad(
                                                                      logprobs,
                                                                      i,
                                                                      start_loss,
                                                                      codelens,
                                                                      (target_data->col(i+1).array() - offset).matrix());
                std::get<1>(cost) += graph::backprop_enabled ? masked_sum(
                                                                      memory,
                                                                      i,
                                                                      0,
                                                                      start_loss,
                                                                      memory_penalty) :
                                                        masked_sum_no_grad(
                                                                      memory,
                                                                      i,
                                                                      0,
                                                                      start_loss,
                                                                      memory_penalty);
        }
        return cost;
}

template<typename R>
std::tuple<R, R> StackedGatedModel<R>::masked_predict_cost(
        shared_index_mat data,
        shared_index_mat target_data,
        uint start_loss,
        shared_eigen_index_vector codelens,
        uint offset,
        R drop_prob) {

        auto initial_state    = this->initial_states();

        SHARED_MAT input_vector;
        SHARED_MAT memory;
        SHARED_MAT logprobs;
        // SHARED_MAT probs;
        std::tuple<R, R> cost(0.0, 0.0);

        auto n = data->cols();
        for (uint i = 0; i < n-1; ++i) {
                // pick this letter from the embedding
                input_vector = this->embedding->rows_pluck(data->col(i));
                memory = gate.activate(input_vector, initial_state.second[this->stack_size-1]);
                input_vector = input_vector->eltmul_broadcast_rowwise(memory);
                // pass this letter to the LSTM for processing
                initial_state = this->stacked_lstm->activate(initial_state, input_vector, drop_prob);
                // classifier takes as input the final hidden layer's activation:
                logprobs      = this->decoder->activate(input_vector, initial_state.second);
                std::get<0>(cost) += graph::backprop_enabled ? masked_cross_entropy(
                                                                      logprobs,
                                                                      i,
                                                                      start_loss,
                                                                      codelens,
                                                                      (target_data->col(i+1).array() - offset).matrix()) :
                                                        masked_cross_entropy_no_grad(
                                                                      logprobs,
                                                                      i,
                                                                      start_loss,
                                                                      codelens,
                                                                      (target_data->col(i+1).array() - offset).matrix());
                std::get<1>(cost) += graph::backprop_enabled ? masked_sum(
                                                                      memory,
                                                                      i,
                                                                      0,
                                                                      start_loss,
                                                                      memory_penalty) :
                                                        masked_sum_no_grad(
                                                                      memory,
                                                                      i,
                                                                      0,
                                                                      start_loss,
                                                                      memory_penalty);
        }
        return cost;
}

template<typename R>
StackedGatedModel<R>::StackedGatedModel (int vocabulary_size, int input_size, int hidden_size, int stack_size, int output_size, bool use_shortcut, R _memory_penalty)
        : StackedModel<R>(vocabulary_size, input_size, hidden_size, stack_size, output_size, use_shortcut),
        memory_penalty(_memory_penalty),
        gate(input_size, hidden_size) {}

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
template<typename R>
StackedGatedModel<R>::StackedGatedModel (
        const typename StackedGatedModel<R>::config_t& config)
        :
        memory_penalty(from_string<R>(config.at("memory_penalty")[0])),
        StackedModel<R>(config),
        gate(
            from_string<int>(config.at("input_size")[0]),
            from_string<int>(config.at("hidden_sizes")[config.at("hidden_sizes").size()-1])) {}

template<typename R>
StackedGatedModel<R>::StackedGatedModel (int vocabulary_size, int input_size, int output_size, std::vector<int>& hidden_sizes, bool use_shortcut, R _memory_penalty)
        :
        StackedModel<R>(vocabulary_size, input_size, output_size, hidden_sizes, use_shortcut),
        memory_penalty(_memory_penalty),
        gate(input_size, hidden_sizes[0]) {}

/**
StackedGatedModel<R>::StackedGatedModel
-------------

Copy constructor with option to make a shallow
or deep copy of the underlying parameters.

If the copy is shallow then the parameters are shared
but separate gradients `dw` are used for each of
thread StackedGatedModel<R>.

Shallow copies are useful for Hogwild and multithreaded
training

See `Mat<R>::shallow_copy`, `examples/character_prediction.cpp`,
`StackedGatedModel<R>::shallow_copy`

Inputs
------

      StackedGatedModel<R> l : StackedGatedModel from which to source parameters and dw
    bool copy_w : whether parameters for new StackedGatedModel should be copies
                  or shared
   bool copy_dw : whether gradients for new StackedGatedModel should be copies
                  shared (Note: sharing `dw` should be used with
                  caution and can lead to unpredictable behavior
                  during optimization).

Outputs
-------

StackedGatedModel<R> out : the copied StackedGatedModel with deep or shallow copy of parameters

**/
template<typename R>
StackedGatedModel<R>::StackedGatedModel (const StackedGatedModel<R>& model, bool copy_w, bool copy_dw) :
    StackedModel<R>(model, copy_w, copy_dw),
        memory_penalty(model.memory_penalty),
        gate(model.gate, copy_w, copy_dw) {}

/**
Shallow Copy
------------

Perform a shallow copy of a StackedGatedModel<R> that has
the same parameters but separate gradients `dw`
for each of its parameters.

Shallow copies are useful for Hogwild and multithreaded
training

See `StackedGatedModel<R>::shallow_copy`, `examples/character_prediction.cpp`.

Outputs
-------

StackedGatedModel<R> out : the copied layer with sharing parameters,
                           but with separate gradients `dw`

**/
template<typename R>
StackedGatedModel<R> StackedGatedModel<R>::shallow_copy() const {
    return StackedGatedModel<R>(*this, false, true);
}

template<typename R>
typename StackedGatedModel<R>::state_type StackedGatedModel<R>::get_final_activation(
        Indexing::Index example,
        R drop_prob) const {
        SHARED_MAT input_vector;
        SHARED_MAT memory;
        auto initial_state = this->initial_states();
        auto n = example.size();
        for (uint i = 0; i < n; ++i) {
                // pick this letter from the embedding
                input_vector  = this->embedding->row_pluck(example[i]);
                memory        = gate.activate(input_vector, initial_state.second[0]);
                input_vector  = input_vector->eltmul_broadcast_rowwise(memory);
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

std::pair<std::vector<std::shared_ptr<Mat<R>>>, std::vector<std::shared_ptr<Mat<R>>>>& : previous state
uint index : embedding observation

Outputs
-------

std::tuple<std::pair<vector<shared_ptr<Mat<R>>>, vector<shared_ptr<Mat<R>>>>, shared_ptr<Mat<R>>, R> out :
    pair of LSTM hidden and cell states, and probabilities from the decoder, and memory usage.

**/
template<typename R>
typename StackedGatedModel<R>::activation_t StackedGatedModel<R>::activate(
        state_type& previous_state,
        const uint& index) const {
    activation_t out;

    auto input_vector = this->embedding->row_pluck(index);
    auto memory       = gate.activate(input_vector, previous_state.second[this->stack_size-1]);
    input_vector      = input_vector->eltmul_broadcast_rowwise(memory);

    std::get<0>(out) = this->stacked_lstm->activate(previous_state, input_vector);
    std::get<1>(out) = softmax(this->decoder->activate(input_vector, std::get<0>(out).second));
    std::get<2>(out) = memory;
    return out;
}

template<typename R>
typename StackedGatedModel<R>::activation_t StackedGatedModel<R>::activate(
        state_type& previous_state,
        const eigen_index_block indices) const {
    activation_t out;
    auto input_vector = this->embedding->rows_pluck(indices);
    auto memory       = gate.activate(input_vector, previous_state.second[this->stack_size-1]);
    input_vector      = input_vector->eltmul_broadcast_rowwise(memory);

    std::get<0>(out) = this->stacked_lstm->activate(previous_state, input_vector);
    std::get<1>(out) = softmax(this->decoder->activate(input_vector, std::get<0>(out).second));
    std::get<2>(out) = memory;

    return out;
}

template<typename R>
std::vector<int> StackedGatedModel<R>::reconstruct(
        Indexing::Index example,
        int eval_steps,
        int symbol_offset) const {

    graph::NoBackprop nb;
    auto initial_state = get_final_activation(example);

    SHARED_MAT input_vector;
    SHARED_MAT memory;
    vector<int> outputs;
    auto last_symbol = argmax(this->decoder->activate(input_vector, initial_state.second));
    outputs.emplace_back(last_symbol);
    last_symbol += symbol_offset;
    for (uint j = 0; j < eval_steps - 1; j++) {
            input_vector  = this->embedding->row_pluck(last_symbol);
            memory        = gate.activate(input_vector, initial_state.second[this->stack_size-1]);
            input_vector  = input_vector->eltmul_broadcast_rowwise(memory);
            initial_state = this->stacked_lstm->activate(initial_state, input_vector);
            last_symbol   = argmax(this->decoder->activate(input_vector, initial_state.second));
            outputs.emplace_back(last_symbol);
            last_symbol += symbol_offset;
    }
    return outputs;
}

template<typename R>
std::vector<utils::OntologyBranch::shared_branch> StackedGatedModel<R>::reconstruct_lattice(
        Indexing::Index example,
        utils::OntologyBranch::shared_branch root,
        int eval_steps) const {
    graph::NoBackprop nb;
    SHARED_MAT input_vector;
    SHARED_MAT memory;
    auto pos = root;
    auto initial_state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
            // pick this letter from the embedding
            input_vector  = this->embedding->row_pluck(example[i]);
            memory        = gate.activate(input_vector, initial_state.second[0]);
            input_vector  = input_vector->eltmul_broadcast_rowwise(memory);
            // pass this letter to the LSTM for processing
            initial_state = this->stacked_lstm->activate(initial_state, input_vector);
            // decoder takes as input the final hidden layer's activation:
    }
    vector<utils::OntologyBranch::shared_branch> outputs;
    // Take the argmax over the available options (0 for go back to
    // root, and 1..n for the different children of the current position)
    auto last_turn = argmax_slice(this->decoder->activate(input_vector, initial_state.second), 0, pos->children.size() + 1);
    // if the turn is 0 go back to root, else go to one of the children using
    // the lattice pointers:
    pos = (last_turn == 0) ? root : pos->children[last_turn-1];
    // add this decision to the output :
    outputs.emplace_back(pos);
    for (uint j = 0; j < eval_steps - 1; j++) {
            input_vector  = this->embedding->row_pluck(pos->id);
            memory        = gate.activate(input_vector, initial_state.second[0]);
            input_vector  = input_vector->eltmul_broadcast_rowwise(memory);
            initial_state = this->stacked_lstm->activate(initial_state, input_vector);
            last_turn     = argmax_slice(this->decoder->activate(input_vector, initial_state.second), 0, pos->children.size() + 1);
            pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
            outputs.emplace_back(pos);
    }
    return outputs;
}

template class StackedGatedModel<float>;
template class StackedGatedModel<double>;
