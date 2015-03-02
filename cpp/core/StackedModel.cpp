#include "StackedModel.h"

DEFINE_int32(stack_size, 4, "How many LSTMs should I stack ?");
DEFINE_int32(input_size, 100, "Size of the word vectors");
DEFINE_int32(hidden, 100, "How many Cells and Hidden Units should each LSTM have ?");
DEFINE_double(decay_rate, 0.95, "What decay rate should RMSProp use ?");
DEFINE_double(rho, 0.95, "What rho / learning rate should the Solver use ?");

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;
using utils::from_string;

template<typename T>
vector<typename StackedModel<T>::shared_mat> StackedModel<T>::parameters() const {
    vector<shared_mat> parameters;
    parameters.push_back(embedding);

    auto decoder_params = decoder.parameters();
    parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());
    for (auto& cell : cells) {
        auto cell_params = cell.parameters();
        parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
    }
    return parameters;
}

template<typename T>
typename StackedModel<T>::config_t StackedModel<T>::configuration() const {
    config_t config;
    config["output_size"].emplace_back(to_string(output_size));
    config["input_size"].emplace_back(to_string(input_size));
    config["vocabulary_size"].emplace_back(to_string(vocabulary_size));
    for (auto& v : hidden_sizes)
        config["hidden_sizes"].emplace_back(to_string(v));
    return config;
}

template<typename T>
void StackedModel<T>::save_configuration(std::string fname) const {

    auto config = configuration();
    utils::map_to_file(config, fname);
}

template<typename T>
void StackedModel<T>::save(std::string dirname) const {
        utils::ensure_directory(dirname);
        // Save the matrices:
        auto params = parameters();
        utils::save_matrices(params, dirname);
        dirname += "config.md";
        save_configuration(dirname);
}

template<typename T>
StackedModel<T> StackedModel<T>::build_from_CLI(string load_location,
                                                                                            int vocab_size,
                                                                                            int output_size,
                                                                                            bool verbose) {
        if (verbose)
                std::cout << "Load location         = " << ((load_location == "") ? "N/A" : load_location)       << std::endl;
        // Load or Construct the model
        auto model = (load_location != "") ?
                StackedModel<T>::load(load_location) :
                StackedModel<T>(
                        vocab_size,
                        FLAGS_input_size,
                        FLAGS_hidden,
                        FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
                        output_size);
        if (verbose) {
                std::cout << ((load_location == "") ? "Constructed Stacked LSTMs" : "Loaded Model") << std::endl;
                std::cout << "Vocabulary size       = " << model.embedding->n      << std::endl;
                std::cout << "Input size            = " << model.input_size        << std::endl;
                std::cout << "Output size           = " << model.output_size       << std::endl;
                std::cout << "Stack size            = " << model.stack_size        << std::endl;
        }
        return model;
}

template<typename T>
StackedModel<T> StackedModel<T>::load(std::string dirname) {
    // fname should be a directory:
    utils::ensure_directory(dirname);
    // load the configuration file
    auto config_name = dirname + "config.md";

    auto config = utils::text_to_map(config_name);

    utils::assert_map_has_key(config, "input_size");
    utils::assert_map_has_key(config, "hidden_sizes");
    utils::assert_map_has_key(config, "vocabulary_size");
    utils::assert_map_has_key(config, "output_size");

    // construct the model using the map
    auto model =  StackedModel<T>(config);

    // get the current parameters of the model.
    auto params = model.parameters();

    // get the new parameters from the saved numpy files
    utils::load_matrices(params, dirname);

    return model;
}

template<typename T>
T StackedModel<T>::masked_predict_cost(
    graph_t& G,
    shared_index_mat data,
    shared_index_mat target_data,
    shared_eigen_index_vector start_loss,
    shared_eigen_index_vector codelens,
    uint offset) {

    auto initial_state    = initial_states();
    auto num_hidden_sizes = hidden_sizes.size();

    shared_mat input_vector;
    shared_mat memory;
    shared_mat logprobs;
    // shared_mat probs;
    T cost = 0.0;

    auto n = data->cols();
    for (uint i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector = G.rows_pluck(embedding, data->col(i));
        // pass this letter to the LSTM for processing
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
        // classifier takes as input the final hidden layer's activation:
        logprobs      = decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
        cost += G.needs_backprop ? masked_cross_entropy(
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

template<typename T>
T StackedModel<T>::masked_predict_cost(
    graph_t& G,
    shared_index_mat data,
    shared_index_mat target_data,
    uint start_loss,
    shared_eigen_index_vector codelens,
    uint offset) {

    auto initial_state    = initial_states();
    auto num_hidden_sizes = hidden_sizes.size();

    shared_mat input_vector;
    shared_mat memory;
    shared_mat logprobs;
    // shared_mat probs;
    T cost = 0.0;

    auto n = data->cols();
    for (uint i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector = G.rows_pluck(embedding, data->col(i));
        // pass this letter to the LSTM for processing
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
        // classifier takes as input the final hidden layer's activation:
        logprobs      = decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
        cost += G.needs_backprop ? masked_cross_entropy(
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
template<typename T>
void StackedModel<T>::name_parameters() {
    embedding->set_name("Embedding");
    decoder.W->set_name("Decoder W");
    decoder.b->set_name("Decoder Bias");
}

template<typename T>
void StackedModel<T>::construct_LSTM_cells() {
    cells = StackedCells<lstm>(input_size, hidden_sizes);
}

template<typename T>
void StackedModel<T>::construct_LSTM_cells(const vector<StackedModel<T>::lstm>& _cells, bool copy_w, bool copy_dw) {
    cells = StackedCells<lstm>(_cells, copy_w, copy_dw);
}

template<typename T>
StackedModel<T>::StackedModel (int _vocabulary_size, int _input_size, int hidden_size, int _stack_size, int _output_size)
    :
    input_size(_input_size),
    output_size(_output_size),
    vocabulary_size(_vocabulary_size),
    stack_size(_stack_size),
    decoder(hidden_size, _output_size) {

    embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
    for (int i = 0; i < stack_size;i++)
        hidden_sizes.emplace_back(hidden_size);
    construct_LSTM_cells();
    name_parameters();
}

template<typename T>
StackedModel<T>::StackedModel (const typename StackedModel<T>::config_t& config)
    :
    vocabulary_size(from_string<int>(config.at("vocabulary_size")[0])),
    output_size(from_string<int>(config.at("output_size")[0])),
    input_size(from_string<int>(config.at("input_size")[0])),
    stack_size(config.at("hidden_sizes").size()),
    decoder(
        from_string<int>(config.at("hidden_sizes")[config.at("hidden_sizes").size()-1]),
        from_string<int>(config.at("output_size")[0])) {
    embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
    for (auto& v : config.at("hidden_sizes"))
        hidden_sizes.emplace_back(from_string<int>(v));

    construct_LSTM_cells();
    name_parameters();
}

template<typename T>
StackedModel<T>::StackedModel (int _vocabulary_size, int _input_size, int _output_size, std::vector<int>& _hidden_sizes)
    :
    input_size(_input_size),
    output_size(_output_size),
    vocabulary_size(_vocabulary_size),
    stack_size(_hidden_sizes.size()),
    hidden_sizes(_hidden_sizes),
    decoder(_hidden_sizes[_hidden_sizes.size()-1], _output_size) {

    embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
    construct_LSTM_cells();
    name_parameters();
}

template<typename T>
StackedModel<T>::StackedModel (const StackedModel<T>& model, bool copy_w, bool copy_dw) :
    input_size(model.input_size),
    output_size(model.output_size),
    vocabulary_size(model.vocabulary_size),
    stack_size(model.stack_size),
    hidden_sizes(model.hidden_sizes),
    decoder(model.decoder, copy_w, copy_dw)
    {
    embedding = make_shared<mat>(*model.embedding, copy_w, copy_dw);
    construct_LSTM_cells(model.cells, copy_w, copy_dw);
    name_parameters();
}

template<typename T>
StackedModel<T> StackedModel<T>::shallow_copy() const {
    return StackedModel<T>(*this, false, true);
}

template<typename T>
typename StackedModel<T>::state_type StackedModel<T>::get_final_activation(
    graph_t& G,
    Indexing::Index example) const {
    shared_mat input_vector;
    auto initial_state = initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = G.row_pluck(embedding, example[i]);
        // pass this letter to the LSTM for processing
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
        // decoder takes as input the final hidden layer's activation:
    }
    return initial_state;
}

// Nested Templates !!
template<typename T>
std::vector<int> StackedModel<T>::reconstruct(
    Indexing::Index example,
    int eval_steps,
    int symbol_offset) {

    graph_t G(false);
    auto initial_state = get_final_activation(G, example);
    vector<int> outputs;
    auto last_symbol = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
    outputs.emplace_back(last_symbol);
    last_symbol += symbol_offset;

    shared_mat input_vector;

    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = G.row_pluck(embedding, last_symbol);
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
        last_symbol   = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
        outputs.emplace_back(last_symbol);
        last_symbol += symbol_offset;
    }
    return outputs;
}

template<typename T>
typename StackedModel<T>::activation_t StackedModel<T>::activate(
    graph_t& G,
    state_type& previous_state,
    const uint& index) const {
    activation_t out;
    out.first =  forward_LSTMs(G, G.row_pluck(embedding, index), previous_state, cells);
    out.second = softmax(decoder.activate(G, out.first.second[stack_size-1]));

    return out;
}

template<typename T>
typename StackedModel<T>::activation_t StackedModel<T>::activate(
    graph_t& G,
    state_type& previous_state,
    const eigen_index_block indices) const {
    activation_t out;

    out.first =  forward_LSTMs(G, G.rows_pluck(embedding, indices), previous_state, cells);
    out.second = softmax(decoder.activate(G, out.first.second[stack_size-1]));

    return out;
}

template<typename T>
typename StackedModel<T>::state_type StackedModel<T>::initial_states() const {
    return lstm::initial_states(hidden_sizes);
}

template<typename T>
std::vector<utils::OntologyBranch::shared_branch> StackedModel<T>::reconstruct_lattice(
    Indexing::Index example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {

    graph_t G(false);
    shared_mat input_vector;
    shared_mat memory;
    auto pos = root;
    auto initial_state = initial_states();
    auto n = example.size();
    for (uint i = 0; i < n; ++i) {
        // pick this letter from the embedding
        input_vector  = G.row_pluck(embedding, example[i]);
        // pass this letter to the LSTM for processing
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
        // decoder takes as input the final hidden layer's activation:
    }
    vector<utils::OntologyBranch::shared_branch> outputs;
    // Take the argmax over the available options (0 for go back to
    // root, and 1..n for the different children of the current position)
    auto last_turn = argmax_slice(decoder.activate(G, initial_state.second[stack_size-1]), 0, pos->children.size() + 1);
    // if the turn is 0 go back to root, else go to one of the children using
    // the lattice pointers:
    pos = (last_turn == 0) ? root : pos->children[last_turn-1];
    // add this decision to the output :
    outputs.emplace_back(pos);
    for (uint j = 0; j < eval_steps - 1; j++) {
        input_vector  = G.row_pluck(embedding, pos->id);
        initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
        last_turn     = argmax_slice(decoder.activate(G, initial_state.second[stack_size-1]), 0, pos->children.size() + 1);
        pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
        outputs.emplace_back(pos);
    }
    return outputs;
}

// Nested Templates !!
template<typename T>
string StackedModel<T>::reconstruct_string(
    Indexing::Index example,
    const utils::Vocab& lookup_table,
    int eval_steps,
    int symbol_offset) {
    auto reconstruction = reconstruct(example, eval_steps, symbol_offset);
    stringstream rec;
    for (auto& cat : reconstruction) {
        rec << (
            (cat < lookup_table.index2word.size()) ?
                lookup_table.index2word.at(cat) :
                (
                    cat == lookup_table.index2word.size() ? "**END**" : "??"
                )
            ) << ", ";
    }
    return rec.str();
}

// Nested Templates !!
template<typename T>
string StackedModel<T>::reconstruct_lattice_string(
    Indexing::Index example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {
    auto reconstruction = reconstruct_lattice(example, root, eval_steps);
    stringstream rec;
    for (auto& cat : reconstruction)
        rec << ((&(*cat) == &(*root)) ? "⟲" : cat->name) << ", ";
    return rec.str();
}

template class StackedModel<float>;
template class StackedModel<double>;
