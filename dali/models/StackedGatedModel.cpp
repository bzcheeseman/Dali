#include "StackedGatedModel.h"


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

template<typename Z>
typename StackedGatedModel<Z>::config_t StackedGatedModel<Z>::configuration() const  {
    auto config = StackedModel<Z>::configuration();
    config["memory_penalty"].emplace_back(to_string(memory_penalty));
    return config;
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
StackedGatedModel<Z>::MaskedActivation::MaskedActivation(Mat<Z> _prediction_error, Mat<Z> _memory_error) :
    prediction_error(_prediction_error), memory_error(_memory_error) {}

template<typename Z>
StackedGatedModel<Z>::MaskedActivation::operator std::tuple<Mat<Z>&, Mat<Z>&>() {
    return std::make_tuple(std::ref(prediction_error), std::ref(memory_error));
}

template<typename Z>
typename StackedGatedModel<Z>::MaskedActivation StackedGatedModel<Z>::masked_predict_cost(
        Mat<int> data,
        Mat<int> target_data,
        Mat<Z> mask,
        Z drop_prob,
        int temporal_offset,
        uint softmax_offset) const {

    utils::Timer mpc("masked_predict_cost");
    auto state = this->initial_states();
    mat total_error(1,1);
    mat memory;
    mat memory_error(1,1);

    auto n = data.dims(0);
    assert (temporal_offset < n);
    assert (target_data.dims(0) >= data.dims(0));

    for (uint timestep = 0; timestep < n - temporal_offset; ++timestep) {
        // pick this letter from the embedding
        utils::Timer gte("get the embeddings");
        auto input_vector = this->embedding[data[timestep]];
        memory = gate.activate(
            {
                input_vector,
                state.back().hidden
            }).sigmoid();
        input_vector = input_vector.eltmul_broadcast_colwise(memory);
        gte.stop();

        utils::Timer flstm("forward lstm");
        state = this->stacked_lstm.activate(
            state,
            input_vector,
            drop_prob
        );
        flstm.stop();

        // classifier takes as input the final hidden layer's activation:
        utils::Timer decode_tm("decode");
        auto logprobs = this->decode(input_vector, state);
        decode_tm.stop();

        auto target = target_data[timestep + temporal_offset];
        if (softmax_offset > 0) {
            target -= softmax_offset;
        }

        utils::Timer softmax_tm("softmax cross entropy");
        auto errors = MatOps<Z>::softmax_cross_entropy_rowwise(logprobs, target);
        softmax_tm.stop();

        utils::Timer masking_tm("masking");
        errors *= mask[timestep + temporal_offset].T();
        memory *= mask[timestep + temporal_offset].T();
        masking_tm.stop();

        total_error += errors.sum();
        memory_error += memory.sum() * memory_penalty;
    }
    mpc.stop();

    return MaskedActivation(total_error, memory_error);
}

template<typename Z>
typename StackedGatedModel<Z>::MaskedActivation StackedGatedModel<Z>::masked_predict_cost(const Batch<Z>& batch,
                                            Z drop_prob,
                                            int temporal_offset,
                                            uint softmax_offset) const {
    return masked_predict_cost(batch.data, batch.target, batch.mask,
                               drop_prob, temporal_offset, softmax_offset);
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
      gate({input_size, hidden_size}, 1) {
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
        {
            from_string<int>(
                config.at("input_size")[0]
            ),
            from_string<int>(
                config.at("hidden_sizes")[config.at("hidden_sizes").size()-1]
            )
        },
        1
    ) {
}

template<typename Z>
StackedGatedModel<Z>::StackedGatedModel (
    int vocabulary_size,
    int input_size,
    int output_size,
    const std::vector<int>& hidden_sizes,
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
        gate({input_size, hidden_sizes[0]}, 1) {
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
    auto state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = this->embedding[example[i]];
        memory        = gate.activate(
            {
                input_vector,
                state.back().hidden
            }
        ).sigmoid();
        if (graph::backprop_enabled() && memory_penalty > 0) {
            // add this sum to objective function
            (memory * memory_penalty).grad();
        }
        input_vector  = input_vector.eltmul_broadcast_colwise(memory);
        // pass this letter to the LSTM for processing
        state = this->stacked_lstm.activate(state, input_vector);
    }
    return state;
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
typename StackedGatedModel<Z>::State StackedGatedModel<Z>::activate(
        state_type& previous_state,
        const uint& index) const {
    return activate(previous_state, Indexing::Index({index}));
}

template<typename Z>
typename StackedGatedModel<Z>::State StackedGatedModel<Z>::activate(
        state_type& previous_state,
        const Indexing::Index indices) const {
    State out;
    auto input_vector = this->embedding[indices];

    out.memory       = gate.activate(
        {
            input_vector,
            previous_state.back().hidden
        }
    ).sigmoid();

    input_vector      = input_vector.eltmul_broadcast_colwise(out.memory);

    out.lstm_state = this->stacked_lstm.activate(
        previous_state,
        input_vector);

    out.prediction = MatOps<Z>::softmax_rowwise(
            this->decode(
                input_vector,
                out.lstm_state
            )
        );

    return out;
}

template<typename Z>
std::vector<int> StackedGatedModel<Z>::reconstruct(
        Indexing::Index example,
        int eval_steps,
        int symbol_offset) const {

    graph::NoBackprop nb;
    auto state = get_final_activation(example);

    mat input_vector;
    mat memory;
    vector<int> outputs;
    auto last_symbol = this->decode(
      input_vector,
      state
    ).argmax();
    outputs.emplace_back(last_symbol);
    last_symbol += symbol_offset;
    for (uint j = 0; j < eval_steps - 1; j++) {
            input_vector  = this->embedding[last_symbol];
            memory        = gate.activate(
                {
                    input_vector,
                    state.back().hidden
                }
            ).sigmoid();
            input_vector  = input_vector.eltmul_broadcast_colwise(memory);
            state = this->stacked_lstm.activate(
                state,
                input_vector);
            last_symbol   = this->decode(
                input_vector,
                state
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
    auto initial_state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        input_vector  = this->embedding[example[i]];
        memory        = gate.activate(
            {
                input_vector,
                initial_state.back().hidden
            }).sigmoid();
        input_vector  = input_vector.eltmul_broadcast_colwise(memory);
        initial_state = this->stacked_lstm.activate(initial_state, input_vector);
    }
    vector<utils::OntologyBranch::shared_branch> outputs;
    // Take the argmax over the available options (0 for go back to
    // root, and 1..n for the different children of the current position)
    auto pos = root;
    auto last_turn = this->decode(
        input_vector,
        initial_state
    ).argmax_slice(
        0,
        pos->children.size() + 1
    );
    // if the turn is 0 go back to root, else go to one of the children using
    // the lattice pointers:
    pos = (last_turn == 0) ? root : pos->children[last_turn-1];
    // add this decision to the output :
    outputs.emplace_back(pos);
    for (int step = 0; step < eval_steps - 1; step++) {
        input_vector  = this->embedding[pos->id];
        memory        = gate.activate(
            {
                input_vector,
                initial_state.back().hidden
            }).sigmoid();
        input_vector  = input_vector.eltmul_broadcast_colwise(memory);
        initial_state = this->stacked_lstm.activate(initial_state, input_vector);
        last_turn     = this->decode(
            input_vector,
            initial_state
        ).argmax_slice(
            0,
            pos->children.size() + 1
        );
        // take an action
        pos = (last_turn == 0) ? root : pos->children[last_turn-1];
        outputs.emplace_back(pos);
    }
    return outputs;
}

template class StackedGatedModel<float>;
template class StackedGatedModel<double>;
