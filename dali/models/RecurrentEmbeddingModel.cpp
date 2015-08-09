#include "RecurrentEmbeddingModel.h"

using std::string;
using std::vector;
using std::stringstream;
using std::make_shared;
using utils::from_string;
using std::to_string;

template<typename R>
void RecurrentEmbeddingModel<R>::save(std::string dirname) const {
    utils::ensure_directory(dirname);
    // Save the matrices:
    auto params = parameters();
    utils::save_matrices(params, dirname);
    dirname += "config.md";
    save_configuration(dirname);
}

template<typename R>
vector<typename RecurrentEmbeddingModel<R>::mat> RecurrentEmbeddingModel<R>::parameters() const {
    vector<mat> parameters;
    parameters.push_back(this->embedding);
    return parameters;
}

template<typename R>
void RecurrentEmbeddingModel<R>::save_configuration(std::string fname) const {
    auto config = configuration();
    utils::map_to_file(config, fname);
}

template<typename R>
RecurrentEmbeddingModel<R>::RecurrentEmbeddingModel(
    int _vocabulary_size, int _input_size, int hidden_size, int _stack_size, int _output_size) :
    vocabulary_size(_vocabulary_size),
    output_size(_output_size) {

    embedding = mat(vocabulary_size, _input_size, weights<R>::uniform(-0.05, 0.05));
    for (int i = 0; i < _stack_size;i++)
        hidden_sizes.emplace_back(hidden_size);
}

template<typename R>
RecurrentEmbeddingModel<R>::RecurrentEmbeddingModel(
    int _vocabulary_size, int _input_size, const std::vector<int>& _hidden_sizes, int _output_size) :
    vocabulary_size(_vocabulary_size),
    output_size(_output_size),
    hidden_sizes(_hidden_sizes) {
    embedding = mat(vocabulary_size, _input_size, weights<R>::uniform(-0.05, 0.05));
}

template<typename R>
RecurrentEmbeddingModel<R>::RecurrentEmbeddingModel(const RecurrentEmbeddingModel& model, bool copy_w, bool copy_dw) :
    vocabulary_size(model.vocabulary_size),
    output_size(model.output_size),
    hidden_sizes(model.hidden_sizes),
    embedding(model.embedding, copy_w, copy_dw) {
}

template<typename R>
RecurrentEmbeddingModel<R>::RecurrentEmbeddingModel(const typename RecurrentEmbeddingModel<R>::config_t& config) :
    vocabulary_size(from_string<int>(config.at("vocabulary_size")[0])),
    output_size(from_string<int>(config.at("output_size")[0])),
    embedding(
        from_string<int>(config.at("vocabulary_size")[0]),
        from_string<int>(config.at("input_size")[0]),
        weights<R>::uniform(-0.05, 0.05)) {
    for (auto& v : config.at("hidden_sizes"))
        hidden_sizes.emplace_back(from_string<int>(v));
}

template<typename R>
typename RecurrentEmbeddingModel<R>::config_t RecurrentEmbeddingModel<R>::configuration() const  {
    config_t config;
    config["output_size"].emplace_back(to_string(output_size));
    config["vocabulary_size"].emplace_back(to_string(vocabulary_size));
    config["input_size"].emplace_back(to_string(embedding.dims(1)));
    for (auto& v : hidden_sizes)
        config["hidden_sizes"].emplace_back(to_string(v));
    return config;
}

template<typename R>
void maybe_save_model(RecurrentEmbeddingModel<R>* model,
                      const string& base_path,
                      const string& label) {
    if (base_path.empty() && FLAGS_save.empty()) return;
    if (FLAGS_save != "") {
        model_save_throttled.maybe_run(std::chrono::seconds(FLAGS_save_frequency_s),
                [model,&base_path,&label]() {
            std::stringstream filename;
            if(!base_path.empty()) {
                filename << base_path;
            } else if (!FLAGS_save.empty()) {
                filename << FLAGS_save;
            }
            if(!label.empty())
                filename << "_" << label;

            filename << "_" << model_snapshot_no;
            model_snapshot_no += 1;

            std::cout << "Saving model to \""
                      << filename.str() << "/\"" << std::endl;

            model->save(filename.str());
        });
    }
}

template void maybe_save_model (RecurrentEmbeddingModel<float>*, const string& base_path, const string& label);
template void maybe_save_model (RecurrentEmbeddingModel<double>*, const string& base_path, const string& label);


template class RecurrentEmbeddingModel<float>;
template class RecurrentEmbeddingModel<double>;
