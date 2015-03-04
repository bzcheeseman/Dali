#include "core/RecurrentEmbeddingModel.h"

using std::string;
using std::vector;
using std::stringstream;
using std::make_shared;
using utils::from_string;
using std::to_string;

template<typename T>
string RecurrentEmbeddingModel<T>::reconstruct_string(
    Indexing::Index example,
    const utils::Vocab& lookup_table,
    int eval_steps,
    int symbol_offset) const {
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

template<typename T>
void RecurrentEmbeddingModel<T>::save(std::string dirname) const {
    utils::ensure_directory(dirname);
    // Save the matrices:
    auto params = parameters();
    utils::save_matrices(params, dirname);
    dirname += "config.md";
    save_configuration(dirname);
}

template<typename T>
typename RecurrentEmbeddingModel<T>::state_type RecurrentEmbeddingModel<T>::initial_states() const {
    return LSTM<T>::initial_states(hidden_sizes);
}

template<typename T>
vector<typename RecurrentEmbeddingModel<T>::shared_mat> RecurrentEmbeddingModel<T>::parameters() const {
        vector<shared_mat> parameters;
        parameters.push_back(this->embedding);
        return parameters;
}

template<typename T>
void RecurrentEmbeddingModel<T>::save_configuration(std::string fname) const {
    auto config = configuration();
    utils::map_to_file(config, fname);
}

template<typename T>
string RecurrentEmbeddingModel<T>::reconstruct_lattice_string(
    Indexing::Index example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) const {
    auto reconstruction = reconstruct_lattice(example, root, eval_steps);
    stringstream rec;
    for (auto& cat : reconstruction)
        rec << ((&(*cat) == &(*root)) ? "⟲" : cat->name) << ", ";
    return rec.str();
}

template<typename T>
RecurrentEmbeddingModel<T>::RecurrentEmbeddingModel(
    int _vocabulary_size, int _input_size, int hidden_size, int _stack_size, int _output_size) :
    input_size(_input_size),
    vocabulary_size(_vocabulary_size),
    stack_size(_stack_size),
    output_size(_output_size) {

    embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
    for (int i = 0; i < stack_size;i++)
        hidden_sizes.emplace_back(hidden_size);
}

template<typename T>
RecurrentEmbeddingModel<T>::RecurrentEmbeddingModel(
    int _vocabulary_size, int _input_size, const std::vector<int>& _hidden_sizes, int _output_size) :
    input_size(_input_size),
    vocabulary_size(_vocabulary_size),
    stack_size(_hidden_sizes.size()),
    output_size(_output_size),
    hidden_sizes(_hidden_sizes) {
    embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
}

template<typename T>
RecurrentEmbeddingModel<T>::RecurrentEmbeddingModel(const typename RecurrentEmbeddingModel<T>::config_t& config) :
    vocabulary_size(from_string<int>(config.at("vocabulary_size")[0])),
    output_size(from_string<int>(config.at("output_size")[0])),
    input_size(from_string<int>(config.at("input_size")[0])),
    stack_size(config.at("hidden_sizes").size()) {

    for (auto& v : config.at("hidden_sizes"))
        hidden_sizes.emplace_back(from_string<int>(v));
    embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
}

template<typename T>
typename RecurrentEmbeddingModel<T>::config_t RecurrentEmbeddingModel<T>::configuration() const  {
    config_t config;
    config["output_size"].emplace_back(to_string(output_size));
    config["input_size"].emplace_back(to_string(input_size));
    config["vocabulary_size"].emplace_back(to_string(vocabulary_size));
    for (auto& v : hidden_sizes)
        config["hidden_sizes"].emplace_back(to_string(v));
    return config;
}

template class RecurrentEmbeddingModel<float>;
template class RecurrentEmbeddingModel<double>;
