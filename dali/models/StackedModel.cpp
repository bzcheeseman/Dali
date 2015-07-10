#include "StackedModel.h"

DEFINE_int32(stack_size,        4,    "How many LSTMs should I stack ?");
DEFINE_int32(input_size,        100,  "Size of the word vectors");
DEFINE_int32(hidden,            100,  "How many Cells and Hidden Units should each LSTM have ?");
DEFINE_double(decay_rate,       0.95, "What decay rate should RMSProp use ?");
DEFINE_double(rho,              0.95, "What rho / learning rate should the Solver use ?");
DEFINE_bool(shortcut,           true, "Use a Stacked LSTM with shortcuts");
DEFINE_bool(memory_feeds_gates, true, "LSTM's memory cell also control gate outputs");

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;
using utils::from_string;

#define GET_STATE_HIDDENS(X) LSTM<Z>::State::hiddens(X)

template<typename Z>
vector<Mat<Z>> StackedModel<Z>::parameters() const {
    auto parameters = RecurrentEmbeddingModel<Z>::parameters();
    auto decoder_params = decoder.parameters();
    parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());
    auto stacked_lstm_params = stacked_lstm.parameters();
    parameters.insert(parameters.end(), stacked_lstm_params.begin(), stacked_lstm_params.end());
    return parameters;
}

template<typename Z>
typename StackedModel<Z>::config_t StackedModel<Z>::configuration() const  {
    auto config = RecurrentEmbeddingModel<Z>::configuration();
    config["use_shortcut"].emplace_back(to_string(use_shortcut ? 1 : 0));
    config["memory_feeds_gates"].emplace_back(to_string(memory_feeds_gates ? 1 : 0));
    return config;
}

template<typename Z>
vector<int> StackedModel<Z>::decoder_initialization(int input_size, vector<int> hidden_sizes, bool use_shortcut, bool input_vector_to_decoder) {
    vector<int> sizes;
    if (use_shortcut) {
        sizes.reserve((input_vector_to_decoder ? 1 : 0) + hidden_sizes.size());
        if (input_vector_to_decoder) {
            sizes.emplace_back(input_size);
        }
        for (auto& s : hidden_sizes)
            sizes.emplace_back(s);
        return sizes;
    } else {
        sizes.emplace_back(hidden_sizes.back());
        return sizes;
    }
}

template<typename Z>
vector<int> StackedModel<Z>::decoder_initialization(int input_size, const vector<string>& hidden_sizes, bool use_shortcut, bool input_vector_to_decoder) {
    vector<int> sizes;
    if (use_shortcut) {
        sizes.reserve((input_vector_to_decoder ? 1 : 0) + hidden_sizes.size());
        if (input_vector_to_decoder) {
            sizes.emplace_back(input_size);
        }
        for (auto& s : hidden_sizes)
            sizes.emplace_back(from_string<int>(s));
        return sizes;
    } else {
        sizes.emplace_back(from_string<int>(hidden_sizes.back()));
        return sizes;
    }
}

template<typename Z>
StackedModel<Z> StackedModel<Z>::build_from_CLI(
        string load_location,
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

template<typename Z>
StackedModel<Z> StackedModel<Z>::load(std::string dirname) {
    // fname should be a directory:
    utils::ensure_directory(dirname);
    // load the configuration file
    auto config_name = dirname + "config.md";

    auto config = utils::text_to_map(config_name);

    utils::assert_map_has_key(config, "input_size");
    utils::assert_map_has_key(config, "hidden_sizes");
    utils::assert_map_has_key(config, "vocabulary_size");
    utils::assert_map_has_key(config, "output_size");
    if (config.find("use_shortcut") == config.end()) {
        std::cout << "Warning: Could not find `use_shortcut` "
                     "parameter, using non shortcut StackedLSTMs"
                  << std::endl;
    }
    if (config.find("memory_feeds_gates") == config.end()) {
        std::cout << "Warning: Could not find `memory_feeds_gates` "
                     "parameter,"" using LSTMs where memory does "
                     "not connect back to gates."
                  << std::endl;
    }
    // construct the model using the map
    auto model =  StackedModel<Z>(config);
    // get the current parameters of the model.
    auto params = model.parameters();
    // get the new parameters from the saved numpy files
    utils::load_matrices(params, dirname);

    return model;
}

template<typename Z>
Mat<Z> StackedModel<Z>::decode(
    Mat<Z> input_vector,
    state_type& states,
    Z drop_prob
    ) const {
    if (use_shortcut) {
        if (drop_prob > 0.0) {
            vector<Mat<Z>> dropped_states;
            std::transform(states.begin(), states.end(), std::back_inserter(dropped_states), [&drop_prob](const typename LSTM<Z>::State& state) {
                return MatOps<Z>::dropout_normalized(state.hidden, drop_prob);
            });
            if (_input_vector_to_decoder) {
                return this->decoder.activate(
                    MatOps<Z>::dropout_normalized(input_vector, drop_prob),
                    dropped_states
                );
            } else {
                return this->decoder.activate(
                    dropped_states
                );
            }
        } else {
            if (_input_vector_to_decoder) {
                return this->decoder.activate(
                    input_vector,
                    LSTM<Z>::State::hiddens(states)
                );
            } else {
                return this->decoder.activate(
                    LSTM<Z>::State::hiddens(states)
                );
            }
        }
    } else {
        if (drop_prob > 0.0) {
            return this->decoder.activate(
                MatOps<Z>::dropout_normalized(states.back().hidden, drop_prob)
            );
        } else {
            return this->decoder.activate(
                states.back().hidden
            );
        }
    }
}

template<typename Z>
Mat<Z> StackedModel<Z>::masked_predict_cost(
        Mat<int> data,
        Mat<int> target_data,
        Mat<Z> mask,
        Z drop_prob,
        int temporal_offset,
        uint softmax_offset) const {

    utils::Timer mpc("masked_predict_cost");
    auto state = this->initial_states();
    mat total_error(1,1);

    auto n = data.dims(0);
    assert (temporal_offset < n);
    assert (target_data.dims(0) >= data.dims(0));

    for (uint timestep = 0; timestep < n - temporal_offset; ++timestep) {
        // pick this letter from the embedding
        utils::Timer gte("get the embeddings");
        auto input_vector = this->embedding[data[timestep]];
        gte.stop();
        // pass this letter to the LSTM for processing

        utils::Timer flstm("forward lstm");
        state = stacked_lstm.activate(
            state,
            input_vector,
            drop_prob
        );
        flstm.stop();

        // classifier takes as input the final hidden layer's activation:
        utils::Timer decode_tm("decode");
        auto logprobs = decode(input_vector, state);
        decode_tm.stop();

        auto target = target_data[timestep + temporal_offset];
        if (softmax_offset > 0) {
            target -= softmax_offset;
        }

        utils::Timer softmax_tm("softmax cross entropy");
        auto errors = MatOps<Z>::softmax_cross_entropy(logprobs, target);
        softmax_tm.stop();

        utils::Timer masking_tm("masking");
        errors *= mask[timestep + temporal_offset].T();
        masking_tm.stop();

        total_error += errors.sum();
    }
    mpc.stop();

    return total_error;
}

// Private method that names the parameters
// For better debugging and reference
template<typename Z>
void StackedModel<Z>::name_parameters() {
    this->embedding.set_name("Embedding");
    decoder.b.set_name("Decoder Bias");
}

template<typename Z>
StackedModel<Z>::StackedModel (
    int vocabulary_size,
    int input_size,
    int hidden_size,
    int stack_size,
    int output_size,
    bool _use_shortcut,
    bool _memory_feeds_gates)
    : RecurrentEmbeddingModel<Z>(
        vocabulary_size,
        input_size,
        hidden_size,
        stack_size,
        output_size),
    use_shortcut(_use_shortcut),
    memory_feeds_gates(_memory_feeds_gates),
    decoder(decoder_initialization(input_size, this->hidden_sizes, _use_shortcut, _input_vector_to_decoder), output_size) {
    stacked_lstm = StackedLSTM<Z>(
            input_size,
            this->hidden_sizes,
            use_shortcut,
            memory_feeds_gates);
    name_parameters();
}

template<typename Z>
StackedModel<Z>::StackedModel (const typename StackedModel<Z>::config_t& config)
    :
    use_shortcut( config.find("use_shortcut") != config.end() ? (from_string<int>(config.at("use_shortcut")[0]) > 0) : false ),
    memory_feeds_gates( config.find("memory_feeds_gates") != config.end() ? (from_string<int>(config.at("memory_feeds_gates")[0]) > 0) : false ),
    RecurrentEmbeddingModel<Z>(config) {
    decoder = StackedInputLayer<Z>(
        decoder_initialization(from_string<int>(config.at("input_size")[0]), this->hidden_sizes, use_shortcut, _input_vector_to_decoder),
        this->output_size
    );
    stacked_lstm = StackedLSTM<Z>(
        from_string<int>(config.at("input_size")[0]),
        this->hidden_sizes,
        use_shortcut,
        memory_feeds_gates
    );
    name_parameters();
}

template<typename Z>
StackedModel<Z>::StackedModel(
    int vocabulary_size,
    int input_size,
    int output_size,
    std::vector<int>& hidden_sizes,
    bool _use_shortcut,
    bool _memory_feeds_gates)
    : RecurrentEmbeddingModel<Z>(vocabulary_size, input_size, hidden_sizes, output_size),
    use_shortcut(_use_shortcut), memory_feeds_gates(_memory_feeds_gates) {
    decoder = StackedInputLayer<Z> (
        decoder_initialization(input_size, this->hidden_sizes, use_shortcut, _input_vector_to_decoder),
        this->output_size
    );
    stacked_lstm = StackedLSTM<Z>(
        input_size,
        this->hidden_sizes,
        use_shortcut,
        memory_feeds_gates
    );
    name_parameters();
}

template<typename Z>
StackedModel<Z>::StackedModel (const StackedModel<Z>& model, bool copy_w, bool copy_dw)
    : RecurrentEmbeddingModel<Z>(model, copy_w, copy_dw),
      use_shortcut(model.use_shortcut),
      _input_vector_to_decoder(model.input_vector_to_decoder()),
      memory_feeds_gates(model.memory_feeds_gates),
      decoder(model.decoder, copy_w, copy_dw) {

    stacked_lstm = StackedLSTM<Z>(
        model.stacked_lstm, copy_w, copy_dw
    );
    name_parameters();
}

template<typename Z>
StackedModel<Z> StackedModel<Z>::shallow_copy() const {
    return StackedModel<Z>(*this, false, true);
}

template<typename Z>
typename StackedModel<Z>::state_type StackedModel<Z>::get_final_activation(
    Indexing::Index example,
    Z drop_prob) const {
    mat input_vector;
    auto initial_state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = this->embedding[example[i]];
        // pass this letter to the LSTM for processing
        initial_state = stacked_lstm.activate(
            initial_state,
            input_vector,
            drop_prob
        );
        // decoder takes as input the final hidden layer's activation:
    }
    return initial_state;
}

template<typename Z>
std::vector<int> StackedModel<Z>::reconstruct(
    Indexing::Index example,
    int eval_steps,
    int symbol_offset) const {

    graph::NoBackprop nb;
    auto initial_state = get_final_activation(example);
    vector<int> outputs;
    auto input_vector = this->embedding[example[example.size() - 1]];
    auto last_symbol = decode(
        input_vector,
        initial_state
    ).argmax();
    outputs.emplace_back(last_symbol);
    last_symbol += symbol_offset;

    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = this->embedding[last_symbol];
        initial_state = stacked_lstm.activate(initial_state, input_vector);
        last_symbol   = decode(
            input_vector,
            initial_state
        ).argmax();
        outputs.emplace_back(last_symbol);
        last_symbol += symbol_offset;
    }
    return outputs;
}

template<typename Z>
typename StackedModel<Z>::activation_t StackedModel<Z>::activate(
    state_type& previous_state,
    const uint& index) const {
    activation_t out;
    auto input_vector = this->embedding[index];
    std::get<0>(out)  = stacked_lstm.activate(previous_state, input_vector);
    std::get<1>(out)  = graph::backprop_enabled() ?
        MatOps<Z>::softmax(
            decode(
                input_vector,
                std::get<0>(out)
            )
        ) :
        MatOps<Z>::softmax_no_grad(
            decode(
                input_vector,
                std::get<0>(out)
            )
        );
    return out;
}

template<typename Z>
typename StackedModel<Z>::activation_t StackedModel<Z>::activate(
    state_type& previous_state,
    Indexing::Index indices) const {
    activation_t out;
    auto input_vector = this->embedding[indices];
    std::get<0>(out)  = stacked_lstm.activate(previous_state, input_vector);
    std::get<1>(out)  = graph::backprop_enabled() ?
        MatOps<Z>::softmax(
            decode(
                input_vector,
                std::get<0>(out)
            )
        ) :
        MatOps<Z>::softmax_no_grad(
            decode(
                input_vector,
                std::get<0>(out)
            )
        );
    return out;
}

template<typename Z>
std::vector<utils::OntologyBranch::shared_branch> StackedModel<Z>::reconstruct_lattice(
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
        input_vector  = this->embedding[example[i]];
        // pass this letter to the LSTM for processing
        initial_state = stacked_lstm.activate(initial_state, input_vector);
        // decoder takes as input the final hidden layer's activation:
    }
    vector<utils::OntologyBranch::shared_branch> outputs;
    // Rake the argmax over the available options (0 for go back to
    // root, and 1..n for the different children of the current position)
    auto last_turn = decode(
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
    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = this->embedding[pos->id];
        initial_state = stacked_lstm.activate(initial_state, input_vector);
        last_turn     = decode(
            input_vector,
            initial_state
        ).argmax_slice(
            0,
            pos->children.size() + 1
        );
        pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
        outputs.emplace_back(pos);
    }
    return outputs;
}

template<typename Z>
const bool& StackedModel<Z>::input_vector_to_decoder() const {
    return _input_vector_to_decoder;
}

template<typename Z>
void StackedModel<Z>::input_vector_to_decoder(bool should_input_feed_to_decoder) {
    if (_input_vector_to_decoder != should_input_feed_to_decoder) {
        _input_vector_to_decoder = should_input_feed_to_decoder;
        decoder.input_sizes(decoder_initialization(this->embedding.dims(1), this->hidden_sizes, use_shortcut, _input_vector_to_decoder));
    }
}

template class StackedModel<float>;
template class StackedModel<double>;
