#include "StackedShortcutModel.h"

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;
using utils::from_string;

template<typename T>
vector<typename StackedShortcutModel<T>::shared_mat> StackedShortcutModel<T>::parameters() const {
        vector<shared_mat> parameters;
        parameters.push_back(embedding);

        auto decoder_params = decoder.parameters();
        parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());
        auto base_cell_params = base_cell.parameters();
        parameters.insert(parameters.end(), base_cell_params.begin(), base_cell_params.end());
        for (auto& cell : cells) {
                auto cell_params = cell.parameters();
                parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
        }
        return parameters;
}

template<typename T>
typename StackedShortcutModel<T>::config_t StackedShortcutModel<T>::configuration() const {
        config_t config;
        config["output_size"].emplace_back(to_string(output_size));
        config["input_size"].emplace_back(to_string(input_size));
        config["vocabulary_size"].emplace_back(to_string(vocabulary_size));
        for (auto& v : hidden_sizes)
                config["hidden_sizes"].emplace_back(to_string(v));
        return config;
}

template<typename T>
void StackedShortcutModel<T>::save_configuration(std::string fname) const {

        auto config = configuration();
        utils::map_to_file(config, fname);
}

template<typename T>
void StackedShortcutModel<T>::save(std::string dirname) const {
        utils::ensure_directory(dirname);
        // Save the matrices:
        auto params = parameters();
        utils::save_matrices(params, dirname);
        dirname += "config.md";
        save_configuration(dirname);
}

template<typename T>
StackedShortcutModel<T> StackedShortcutModel<T>::build_from_CLI(string load_location,
                                                                                                                                int vocab_size,
                                                                                                                                int output_size,
                                                                                                                                bool verbose) {
        if (verbose)
                std::cout << "Load location         = " << ((load_location == "") ? "N/A" : load_location)       << std::endl;
        // Load or Construct the model
        auto model = (load_location != "") ?
                StackedShortcutModel<T>::load(load_location) :
                StackedShortcutModel<T>(
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
std::vector<int> StackedShortcutModel<T>::decoder_initialization(int input_size, int hidden_size, int stack_size) {
        vector<int> sizes;
        sizes.reserve(1 + stack_size);
        sizes.emplace_back(input_size);
        for (int i = 0; i < stack_size; i++) sizes.emplace_back(hidden_size);
        return sizes;
}

template<typename T>
vector<int> StackedShortcutModel<T>::decoder_initialization(int input_size, vector<int> hidden_sizes) {
        vector<int> sizes;
        sizes.reserve(1 + hidden_sizes.size());
        sizes.emplace_back(input_size);
        for (auto& s : hidden_sizes) sizes.emplace_back(s);
        return sizes;
}

template<typename T>
vector<int> StackedShortcutModel<T>::decoder_initialization(int input_size, const vector<string>& hidden_sizes) {
        vector<int> sizes;
        sizes.reserve(1 + hidden_sizes.size());
        sizes.emplace_back(input_size);
        for (auto& s : hidden_sizes) sizes.emplace_back(from_string<int>(s));
        return sizes;
}

template<typename T>
StackedShortcutModel<T> StackedShortcutModel<T>::load(std::string dirname) {
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
        auto model =  StackedShortcutModel<T>(config);

        // get the current parameters of the model.
        auto params = model.parameters();

        // get the new parameters from the saved numpy files
        utils::load_matrices(params, dirname);

        return model;
}

template<typename T>
T StackedShortcutModel<T>::masked_predict_cost(
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
        T cost = 0.0;

        auto n = data->cols();
        for (uint i = 0; i < n-1; ++i) {
                // pick this letter from the embedding
                input_vector = G.rows_pluck(embedding, data->col(i));
                // pass this letter to the LSTM for processing
                initial_state = forward_LSTMs(G, input_vector, initial_state, base_cell, cells);
                // classifier takes as input the final hidden layer's activation:
                #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                        logprobs      = decoder.activate(G, input_vector, initial_state.second);
                #else
                        logprobs      = decoder.activate(G, initial_state.second[initial_state.second.size() - 1]);
                #endif
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
typename StackedShortcutModel<T>::state_type StackedShortcutModel<T>::initial_states() const {
    return lstm::initial_states(hidden_sizes);
}

template<typename T>
T StackedShortcutModel<T>::masked_predict_cost(
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
                initial_state = forward_LSTMs(G, input_vector, initial_state, base_cell, cells);
                // classifier takes as input the final hidden layer's activation:
                #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                        logprobs      = decoder.activate(G, input_vector, initial_state.second);
                #else
                        logprobs      = decoder.activate(G, initial_state.second[initial_state.second.size() - 1]);
                #endif
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
void StackedShortcutModel<T>::name_parameters() {
        embedding->set_name("Embedding");
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                int i = 1;
                for (auto& W : decoder.matrices) {
                        string name = "Decoder W_" + std::to_string(i++);
                        W->set_name(name);
                }
        #else
                decoder.W->set_name("Decoder W");
        #endif
        decoder.b->set_name("Decoder Bias");
}

template<typename T>
void StackedShortcutModel<T>::construct_LSTM_cells() {
        auto hidden_sizes_subset = vector<int>(hidden_sizes.begin() + 1, hidden_sizes.end());
        if (hidden_sizes.size() > 0)
                cells = StackedCells<shortcut_lstm>(hidden_sizes[0], input_size, hidden_sizes_subset);
}

template<typename T>
void StackedShortcutModel<T>::construct_LSTM_cells(const vector<StackedShortcutModel<T>::shortcut_lstm>& _cells, bool copy_w, bool copy_dw) {
        if (_cells.size() > 0)
                cells = StackedCells<shortcut_lstm>(_cells, copy_w, copy_dw);
}

template<typename T>
StackedShortcutModel<T>::StackedShortcutModel (int _vocabulary_size, int _input_size, int hidden_size, int _stack_size, int _output_size)
        :
        input_size(_input_size),
        output_size(_output_size),
        vocabulary_size(_vocabulary_size),
        stack_size(_stack_size),
        base_cell(_input_size, hidden_size),
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                decoder(decoder_initialization(
                        _input_size,
                        hidden_size,
                        _stack_size
                ), _output_size)
        #else
                decoder(hidden_size, _output_size)
        #endif
        {
        embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
        for (int i = 0; i < stack_size;i++)
                hidden_sizes.emplace_back(hidden_size);
        construct_LSTM_cells();
        name_parameters();
}

template<typename T>
StackedShortcutModel<T>::StackedShortcutModel (
        const typename StackedShortcutModel<T>::config_t& config)
        :
        vocabulary_size(from_string<int>(config.at("vocabulary_size")[0])),
        output_size(from_string<int>(config.at("output_size")[0])),
        input_size(from_string<int>(config.at("input_size")[0])),
        stack_size(config.at("hidden_sizes").size()),
        base_cell(
                from_string<int>(config.at("input_size")[0]),
                from_string<int>(config.at("hidden_sizes")[0])),
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                decoder(
                decoder_initialization(
                        from_string<int>(config.at("input_size")[0]),
                        config.at("hidden_sizes")
                ), from_string<int>(config.at("output_size")[0]))
        #else
                decoder(from_string<int>(config.at("hidden_sizes")[config.at("hidden_sizes").size() - 1])
                        , from_string<int>(config.at("output_size")[0]))
        #endif
        {
        embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
        for (auto& v : config.at("hidden_sizes"))
                hidden_sizes.emplace_back(from_string<int>(v));

        construct_LSTM_cells();
        name_parameters();
}

template<typename T>
StackedShortcutModel<T>::StackedShortcutModel (int _vocabulary_size, int _input_size, int _output_size, std::vector<int>& _hidden_sizes)
        :
        input_size(_input_size),
        output_size(_output_size),
        vocabulary_size(_vocabulary_size),
        stack_size(_hidden_sizes.size()),
        hidden_sizes(_hidden_sizes),
        base_cell(_input_size, _hidden_sizes[0]),
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                decoder(decoder_initialization(
                        _input_size,
                        _hidden_sizes
                ), _output_size)
        #else
                decoder(_hidden_sizes[ _hidden_sizes.size() - 1], _output_size)
        #endif
        {

        embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
        construct_LSTM_cells();
        name_parameters();
}

template<typename T>
StackedShortcutModel<T>::StackedShortcutModel (const StackedShortcutModel<T>& model, bool copy_w, bool copy_dw) :
    input_size(model.input_size),
        output_size(model.output_size),
        vocabulary_size(model.vocabulary_size),
        stack_size(model.stack_size),
        hidden_sizes(model.hidden_sizes),
        base_cell(model.base_cell, copy_w, copy_dw),
        decoder(model.decoder, copy_w, copy_dw)
    {
    embedding = make_shared<mat>(*model.embedding, copy_w, copy_dw);
    construct_LSTM_cells(model.cells, copy_w, copy_dw);
    name_parameters();
}

template<typename T>
StackedShortcutModel<T> StackedShortcutModel<T>::shallow_copy() const {
    return StackedShortcutModel<T>(*this, false, true);
}

template<typename T>
template<typename K>
typename StackedShortcutModel<T>::state_type StackedShortcutModel<T>::get_final_activation(
        graph_t& G,
        const K& example) const {
        shared_mat input_vector;
        auto initial_state = initial_states();
        auto n = example.cols() * example.rows();
        for (uint i = 0; i < n; ++i) {
                // pick this letter from the embedding
                input_vector  = G.row_pluck(embedding, example(i));
                // pass this letter to the LSTM for processing
                initial_state = forward_LSTMs(G, input_vector, initial_state, base_cell, cells);
        }
        return initial_state;
}

// Nested Templates !!
template<typename T>
template<typename K>
std::vector<int> StackedShortcutModel<T>::reconstruct(
    K example,
    int eval_steps,
    int symbol_offset) {

        graph_t G(false);

        auto initial_state = get_final_activation(G, example);
        vector<int> outputs;

        auto input_vector = G.row_pluck(embedding, example((example.cols() * example.rows()) -1));
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                auto last_symbol = argmax(decoder.activate(G, input_vector, initial_state.second));
        #else
                auto last_symbol = argmax(decoder.activate(G, initial_state.second[initial_state.second.size()-1] ));
        #endif
        outputs.emplace_back(last_symbol);
        last_symbol += symbol_offset;

        for (uint j = 0; j < eval_steps - 1; j++) {
                input_vector  = G.row_pluck(embedding, last_symbol);
                initial_state = forward_LSTMs(G, input_vector, initial_state, base_cell, cells);
                #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                        last_symbol   = argmax(decoder.activate(G, input_vector, initial_state.second));
                #else
                        last_symbol   = argmax(decoder.activate(G, initial_state.second[initial_state.second.size() - 1]));
                #endif
                outputs.emplace_back(last_symbol);
                last_symbol += symbol_offset;
        }
        return outputs;
}

template<typename T>
typename StackedShortcutModel<T>::activation_t StackedShortcutModel<T>::activate(
    graph_t& G,
    state_type& previous_state,
    const uint& index) const {
    activation_t out;
    auto input_vector = G.row_pluck(embedding, index);
    out.first = forward_LSTMs(G, input_vector, previous_state, base_cell, cells);

    #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
        out.second = softmax(decoder.activate(G, input_vector, out.first.second));
    #else
        out.second = softmax(decoder.activate(G, out.first.second[out.first.second.size() - 1] ));
    #endif

    return out;
}

template<typename T>
typename StackedShortcutModel<T>::activation_t StackedShortcutModel<T>::activate(
    graph_t& G,
    state_type& previous_state,
    const eigen_index_block indices) const {
    activation_t out;
    auto input_vector = G.rows_pluck(embedding, indices);
    out.first = forward_LSTMs(G, input_vector, previous_state, base_cell, cells);

    #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
        out.second = softmax(decoder.activate(G, input_vector, out.first.second));
    #else
        out.second = softmax(decoder.activate(G, out.first.second[out.first.second.size() - 1] ));
    #endif

    return out;
}

template<typename T>
template<typename K>
std::vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<T>::reconstruct_lattice(
    K example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {

        graph_t G(false);
        shared_mat input_vector;
        shared_mat memory;
        auto pos = root;
        auto initial_state = initial_states();
        auto n = example.cols() * example.rows();
        for (uint i = 0; i < n; ++i) {
                // pick this letter from the embedding
                input_vector  = G.row_pluck(embedding, example(i));
                // pass this letter to the LSTM for processing
                initial_state = forward_LSTMs(G, input_vector, initial_state, base_cell, cells);
                // decoder takes as input the final hidden layer's activation:
        }
        vector<utils::OntologyBranch::shared_branch> outputs;
        // Take the argmax over the available options (0 for go back to
        // root, and 1..n for the different children of the current position)
        #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                auto last_turn = argmax_slice(decoder.activate(G, input_vector, initial_state.second), 0, pos->children.size() + 1);
        #else
                auto last_turn = argmax_slice(decoder.activate(G, initial_state.second[initial_state.second.size() - 1] ), 0, pos->children.size() + 1);
        #endif
        // if the turn is 0 go back to root, else go to one of the children using
        // the lattice pointers:
        pos = (last_turn == 0) ? root : pos->children[last_turn-1];
        // add this decision to the output :
        outputs.emplace_back(pos);
        for (uint j = 0; j < eval_steps - 1; j++) {
                input_vector  = G.row_pluck(embedding, pos->id);
                initial_state = forward_LSTMs(G, input_vector, initial_state, base_cell, cells);
                #ifdef SHORTCUT_DECODE_ACROSS_LAYERS
                        last_turn     = argmax_slice(decoder.activate(G, input_vector, initial_state.second), 0, pos->children.size() + 1);
                #else
                        last_turn     = argmax_slice(decoder.activate(G, initial_state.second[initial_state.second.size() - 1] ), 0, pos->children.size() + 1);
                #endif
                pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
                outputs.emplace_back(pos);
        }
        return outputs;
}

// Nested Templates !!
template<typename T>
template<typename K>
string StackedShortcutModel<T>::reconstruct_string(
    K example,
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
template<typename K>
string StackedShortcutModel<T>::reconstruct_lattice_string(
    K example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {
        auto reconstruction = reconstruct_lattice(example, root, eval_steps);
        stringstream rec;
        for (auto& cat : reconstruction)
                rec << ((&(*cat) == &(*root)) ? "⟲" : cat->name) << ", ";
        return rec.str();
}

typedef Eigen::Block< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, 1, Eigen::Dynamic, !Eigen::RowMajor> index_row;
typedef Eigen::VectorBlock< index_row, Eigen::Dynamic> sliced_row;

typedef Eigen::Block< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Dynamic, 1, !Eigen::RowMajor> index_col;
typedef Eigen::VectorBlock< index_col, Eigen::Dynamic> sliced_col;

template string StackedShortcutModel<float>::reconstruct_string(sliced_row, const utils::Vocab&, int, int);
template string StackedShortcutModel<double>::reconstruct_string(sliced_row, const utils::Vocab&, int, int);

template string StackedShortcutModel<float>::reconstruct_string(index_row, const utils::Vocab&, int, int);
template string StackedShortcutModel<double>::reconstruct_string(index_row, const utils::Vocab&, int, int);

template string StackedShortcutModel<float>::reconstruct_string(sliced_col, const utils::Vocab&, int, int);
template string StackedShortcutModel<double>::reconstruct_string(sliced_col, const utils::Vocab&, int, int);

template string StackedShortcutModel<float>::reconstruct_string(index_col, const utils::Vocab&, int, int);
template string StackedShortcutModel<double>::reconstruct_string(index_col, const utils::Vocab&, int, int);

template vector<int> StackedShortcutModel<float>::reconstruct(sliced_row, int, int);
template vector<int> StackedShortcutModel<double>::reconstruct(sliced_row, int, int);

template vector<int> StackedShortcutModel<float>::reconstruct(index_row, int, int);
template vector<int> StackedShortcutModel<double>::reconstruct(index_row, int, int);

template vector<int> StackedShortcutModel<float>::reconstruct(sliced_col, int, int);
template vector<int> StackedShortcutModel<double>::reconstruct(sliced_col, int, int);

template vector<int> StackedShortcutModel<float>::reconstruct(index_col, int, int);
template vector<int> StackedShortcutModel<double>::reconstruct(index_col, int, int);

template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const index_col&) const;
template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const index_row&) const;
template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const sliced_row&) const;
template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const sliced_col&) const;
template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const eigen_index_block_scalar&) const;

template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const index_col&) const;
template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const index_row&) const;
template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const sliced_row&) const;
template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const sliced_col&) const;
template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const eigen_index_block_scalar&) const;

typedef Eigen::VectorBlock< Eigen::Matrix<uint, Eigen::Dynamic, 1>, Eigen::Dynamic> vector_block;
typedef Eigen::VectorBlock< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Dynamic> submatrix_block;

template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const vector_block&) const;
template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const vector_block&) const;
template StackedShortcutModel<float>::state_type StackedShortcutModel<float>::get_final_activation(Graph<float>&, const submatrix_block&) const;
template StackedShortcutModel<double>::state_type StackedShortcutModel<double>::get_final_activation(Graph<double>&, const submatrix_block&) const;

template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<float>::reconstruct_lattice(sliced_row, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<double>::reconstruct_lattice(sliced_row, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<float>::reconstruct_lattice(index_row, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<double>::reconstruct_lattice(index_row, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<float>::reconstruct_lattice(sliced_col, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<double>::reconstruct_lattice(sliced_col, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<float>::reconstruct_lattice(index_col, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedShortcutModel<double>::reconstruct_lattice(index_col, utils::OntologyBranch::shared_branch, int);

template string StackedShortcutModel<float>::reconstruct_lattice_string(sliced_row, utils::OntologyBranch::shared_branch, int);
template string StackedShortcutModel<double>::reconstruct_lattice_string(sliced_row, utils::OntologyBranch::shared_branch, int);

template string StackedShortcutModel<float>::reconstruct_lattice_string(index_row, utils::OntologyBranch::shared_branch, int);
template string StackedShortcutModel<double>::reconstruct_lattice_string(index_row, utils::OntologyBranch::shared_branch, int);

template string StackedShortcutModel<float>::reconstruct_lattice_string(sliced_col, utils::OntologyBranch::shared_branch, int);
template string StackedShortcutModel<double>::reconstruct_lattice_string(sliced_col, utils::OntologyBranch::shared_branch, int);

template string StackedShortcutModel<float>::reconstruct_lattice_string(index_col, utils::OntologyBranch::shared_branch, int);
template string StackedShortcutModel<double>::reconstruct_lattice_string(index_col, utils::OntologyBranch::shared_branch, int);

template class StackedShortcutModel<float>;
template class StackedShortcutModel<double>;
