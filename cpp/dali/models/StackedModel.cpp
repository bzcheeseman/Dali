#include "StackedModel.h"

DEFINE_int32(stack_size, 4, "How many LSTMs should I stack ?");
DEFINE_int32(input_size, 100, "Size of the word vectors");
DEFINE_int32(hidden, 100, "How many Cells and Hidden Units should each LSTM have ?");
DEFINE_double(decay_rate, 0.95, "What decay rate should RMSProp use ?");
DEFINE_double(rho, 0.95, "What rho / learning rate should the Solver use ?");
DEFINE_bool(shortcut, true, "Use a Stacked LSTM with shortcuts");

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;
using utils::from_string;

template<typename R>
vector<Mat<R>> StackedModel<R>::parameters() const {
    auto parameters = RecurrentEmbeddingModel<R>::parameters();
    auto decoder_params = decoder->parameters();
    parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());
    auto stacked_lstm_params = stacked_lstm->parameters();
    parameters.insert(parameters.end(), stacked_lstm_params.begin(), stacked_lstm_params.end());
    return parameters;
}


template<typename R>
typename StackedModel<R>::config_t StackedModel<R>::configuration() const  {
    auto config = RecurrentEmbeddingModel<R>::configuration();
    config["use_shortcut"].emplace_back(to_string(use_shortcut ? 1 : 0));
    return config;
}

template<typename R>
vector<int> StackedModel<R>::decoder_initialization(int input_size, vector<int> hidden_sizes) {
    vector<int> sizes;
    sizes.reserve(1 + hidden_sizes.size());
    sizes.emplace_back(input_size);
    for (auto& s : hidden_sizes) sizes.emplace_back(s);
    return sizes;
}

template<typename R>
vector<int> StackedModel<R>::decoder_initialization(int input_size, const vector<string>& hidden_sizes) {
    vector<int> sizes;
    sizes.reserve(1 + hidden_sizes.size());
    sizes.emplace_back(input_size);
    for (auto& s : hidden_sizes) sizes.emplace_back(from_string<int>(s));
    return sizes;
}

template<typename R>
StackedModel<R> StackedModel<R>::build_from_CLI(string load_location,
                                                int vocab_size,
                                                int output_size,
                                                bool verbose) {
        if (verbose)
                std::cout << "Load location         = " << ((load_location == "") ? "N/A" : load_location)       << std::endl;
        // Load or Construct the model
        auto model = (load_location != "") ?
                StackedModel<R>::load(load_location) :
                StackedModel<R>(
                        vocab_size,
                        FLAGS_input_size,
                        FLAGS_hidden,
                        FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
                        output_size,
                        FLAGS_shortcut);
        if (verbose) {
                std::cout << ((load_location == "") ? "Constructed Stacked LSTMs" : "Loaded Model") << std::endl;
                std::cout << "Vocabulary size       = " << model.embedding.dims(0)  << std::endl;
                std::cout << "Input size            = " << model.input_size         << std::endl;
                std::cout << "Output size           = " << model.output_size        << std::endl;
                std::cout << "Stack size            = " << model.stack_size         << std::endl;
        }
        return model;
}

template<typename R>
StackedModel<R> StackedModel<R>::load(std::string dirname) {
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
        std::cout << "Warning: Could not find `use_shortcut` parameter, using non shortcut StackedLSTMs" <<std::endl;
    }

    // construct the model using the map
    auto model =  StackedModel<R>(config);

    // get the current parameters of the model.
    auto params = model.parameters();

    // get the new parameters from the saved numpy files
    utils::load_matrices(params, dirname);

    return model;
}

template<typename R>
R StackedModel<R>::masked_predict_cost(
    shared_index_mat data,
    shared_index_mat target_data,
    shared_eigen_index_vector start_loss,
    shared_eigen_index_vector codelens,
    uint offset,
    R drop_prob) {
    auto initial_state    = this->initial_states();
    mat input_vector;
    mat memory;
    mat logprobs;

    R cost = 0.0;

    auto n = data->cols();
    for (uint i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector = this->embedding.rows_pluck(data->col(i));
        // pass this letter to the LSTM for processing
        initial_state = stacked_lstm->activate(initial_state, input_vector, drop_prob);
        // classifier takes as input the final hidden layer's activation:
        logprobs      = decoder->activate(input_vector, std::get<1>(initial_state));
        cost += graph::backprop_enabled ? masked_cross_entropy(
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
    }
    return cost;
}

template<typename R>
R StackedModel<R>::masked_predict_cost(
    shared_index_mat data,
    shared_index_mat target_data,
    uint start_loss,
    shared_eigen_index_vector codelens,
    uint offset,
    R drop_prob) {

    auto initial_state    = this->initial_states();

    mat input_vector;
    mat memory;
    mat logprobs;

    R cost = 0.0;

    auto n = data->cols();
    for (uint i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector = this->embedding.rows_pluck(data->col(i));
        // pass this letter to the LSTM for processing
        initial_state = stacked_lstm->activate(initial_state, input_vector, drop_prob);
        // classifier takes as input the final hidden layer's activation:
        logprobs      = decoder->activate(input_vector, std::get<1>(initial_state));
        cost += graph::backprop_enabled ? masked_cross_entropy(
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
    }
    return cost;
}

// Private method that names the parameters
// For better debugging and reference
template<typename R>
void StackedModel<R>::name_parameters() {
    this->embedding.set_name("Embedding");
    decoder->b.set_name("Decoder Bias");
    // TODO: do layer naming!!
    // if (use_shortcut) {
    //     decoder->W->set_name("Decoder W");
    // } else {
    //     for (auto& mat : decoder->matrices) {

    //     }
    // }

}

template<typename R>
StackedModel<R>::StackedModel (int vocabulary_size, int input_size, int hidden_size, int stack_size, int output_size, bool _use_shortcut)
    : RecurrentEmbeddingModel<R>(vocabulary_size, input_size, hidden_size, stack_size, output_size),
    use_shortcut(_use_shortcut) {
    if (use_shortcut) {
        decoder = make_shared<StackedInputLayer<R>>(
            decoder_initialization(this->input_size, this->hidden_sizes),
            this->output_size);
        stacked_lstm = make_shared<StackedShortcutLSTM<R>>(
            this->input_size,
            this->hidden_sizes);
    } else {
        decoder = make_shared<Layer<R>>(
            this->hidden_sizes[this->hidden_sizes.size() - 1],
            this->output_size);
        stacked_lstm = make_shared<StackedLSTM<R>>(
            this->input_size,
            this->hidden_sizes);
    }
    name_parameters();
}

template<typename R>
StackedModel<R>::StackedModel (const typename StackedModel<R>::config_t& config)
    :
    use_shortcut( config.find("use_shortcut") != config.end() ? (from_string<int>("use_shortcut") > 0) : false ),
    RecurrentEmbeddingModel<R>(config) {

    if (use_shortcut) {
        decoder = make_shared<StackedInputLayer<R>>(
            decoder_initialization(this->input_size, this->hidden_sizes),
            this->output_size
        );
        stacked_lstm = make_shared<StackedShortcutLSTM<R>>(
            this->input_size,
            this->hidden_sizes
        );
    } else {
        decoder = make_shared<Layer<R>>(
            this->hidden_sizes[this->hidden_sizes.size() - 1],
            this->output_size
        );
        stacked_lstm = make_shared<StackedLSTM<R>>(
            this->input_size,
            this->hidden_sizes
        );
    }
    name_parameters();
}

template<typename R>
StackedModel<R>::StackedModel (int vocabulary_size, int input_size, int output_size, std::vector<int>& hidden_sizes, bool _use_shortcut)
    : RecurrentEmbeddingModel<R>(vocabulary_size, input_size, hidden_sizes, output_size),
    use_shortcut(_use_shortcut) {
    if (use_shortcut) {
        decoder = make_shared<StackedInputLayer<R>>(decoder_initialization(this->input_size, this->hidden_sizes), this->output_size);
        stacked_lstm = make_shared<StackedShortcutLSTM<R>>(this->input_size, this->hidden_sizes);
    } else {
        decoder = make_shared<Layer<R>>(this->hidden_sizes[this->hidden_sizes.size() - 1], this->output_size);
        stacked_lstm = make_shared<StackedLSTM<R>>(this->input_size, this->hidden_sizes);
    }
    name_parameters();
}

template<typename R>
StackedModel<R>::StackedModel (const StackedModel<R>& model, bool copy_w, bool copy_dw)
    : RecurrentEmbeddingModel<R>(model, copy_w, copy_dw),
      use_shortcut(model.use_shortcut) {

    if (use_shortcut) {
        decoder = make_shared<StackedInputLayer<R>>(
            *dynamic_cast<StackedInputLayer<R>*>(model.decoder.get()), copy_w, copy_dw
        );
        stacked_lstm = make_shared<StackedShortcutLSTM<R>>(
            *dynamic_cast<StackedShortcutLSTM<R>*>(model.stacked_lstm.get()), copy_w, copy_dw
        );
    } else {
        decoder = make_shared<Layer<R>>(
            *dynamic_cast<Layer<R>*>(model.decoder.get()), copy_w, copy_dw
        );
        stacked_lstm = make_shared<StackedLSTM<R>>(
            *dynamic_cast<StackedLSTM<R>*>(model.stacked_lstm.get()), copy_w, copy_dw
        );
    }

    name_parameters();
}

template<typename R>
StackedModel<R> StackedModel<R>::shallow_copy() const {
    return StackedModel<R>(*this, false, true);
}

template<typename R>
typename StackedModel<R>::state_type StackedModel<R>::get_final_activation(
    Indexing::Index example,
    R drop_prob) const {
    mat input_vector;
    auto initial_state = this->initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = this->embedding.row_pluck(example[i]);
        // pass this letter to the LSTM for processing
        initial_state = stacked_lstm->activate(initial_state, input_vector, drop_prob);
        // decoder takes as input the final hidden layer's activation:
    }
    return initial_state;
}

template<typename R>
std::vector<int> StackedModel<R>::reconstruct(
    Indexing::Index example,
    int eval_steps,
    int symbol_offset) const {

    graph::NoBackprop nb;
    auto initial_state = get_final_activation(example);
    vector<int> outputs;
    auto input_vector = this->embedding.row_pluck(example[example.size() - 1]);
    auto last_symbol = decoder->activate(input_vector, std::get<1>(initial_state)).argmax();
    outputs.emplace_back(last_symbol);
    last_symbol += symbol_offset;

    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = this->embedding.row_pluck(last_symbol);
        initial_state = stacked_lstm->activate(initial_state, input_vector);
        last_symbol   = decoder->activate(input_vector, std::get<1>(initial_state)).argmax();
        outputs.emplace_back(last_symbol);
        last_symbol += symbol_offset;
    }
    return outputs;
}

template<typename R>
typename StackedModel<R>::activation_t StackedModel<R>::activate(
    state_type& previous_state,
    const uint& index) const {
    activation_t out;
    auto input_vector = this->embedding.row_pluck(index);
    std::get<0>(out)  = stacked_lstm->activate(previous_state, input_vector);
    std::get<1>(out)  = MatOps<R>::softmax_no_grad(decoder->activate(input_vector, std::get<1>(std::get<0>(out))));
    return out;
}

template<typename R>
typename StackedModel<R>::activation_t StackedModel<R>::activate(
    state_type& previous_state,
    const eigen_index_block indices) const {
    activation_t out;
    auto input_vector = this->embedding.rows_pluck(indices);
    std::get<0>(out)  = stacked_lstm->activate(previous_state, input_vector);
    std::get<1>(out)  = MatOps<R>::softmax_no_grad(decoder->activate(input_vector, std::get<1>(std::get<0>(out))));
    return out;
}

template<typename R>
std::vector<utils::OntologyBranch::shared_branch> StackedModel<R>::reconstruct_lattice(
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
        // pass this letter to the LSTM for processing
        initial_state = stacked_lstm->activate(initial_state, input_vector);
        // decoder takes as input the final hidden layer's activation:
    }
    vector<utils::OntologyBranch::shared_branch> outputs;
    // Rake the argmax over the available options (0 for go back to
    // root, and 1..n for the different children of the current position)
    auto last_turn = decoder->activate(input_vector, std::get<1>(initial_state)).argmax_slice(0, pos->children.size() + 1);
    // if the turn is 0 go back to root, else go to one of the children using
    // the lattice pointers:
    pos = (last_turn == 0) ? root : pos->children[last_turn-1];
    // add this decision to the output :
    outputs.emplace_back(pos);
    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = this->embedding.row_pluck(pos->id);
        initial_state = stacked_lstm->activate(initial_state, input_vector);
        last_turn     = decoder->activate(input_vector, std::get<1>(initial_state)).argmax_slice( 0, pos->children.size() + 1);
        pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
        outputs.emplace_back(pos);
    }
    return outputs;
}

template class StackedModel<float>;
template class StackedModel<double>;
