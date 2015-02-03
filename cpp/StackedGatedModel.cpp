#include "StackedGatedModel.h"

#include <iostream>
#include <fstream>

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;

/**
Parameters
----------

Create a vector of shared pointers to the underlying matrices
of the model. Useful for saving, loading parameters all at once
and for telling Solvers which parameters should be updated
during each training loop.

Outputs
-------

std::vector<std::shared_ptr<Mat<T>>> parameters : vector of model parameters

**/

template<typename T>
vector<typename StackedGatedModel<T>::shared_mat> StackedGatedModel<T>::parameters() {
	vector<shared_mat> parameters;
	parameters.push_back(embedding);

	auto gate_params = gate.parameters();
	parameters.insert(parameters.end(), gate_params.begin(), gate_params.end());

	auto decoder_params = decoder.parameters();
	parameters.insert(parameters.end(), decoder_params.begin(), decoder_params.end());
	for (auto& cell : cells) {
		auto cell_params = cell.parameters();
		parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
	}
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

std::unordered_map<std::string, std::vector< std::string >> config : configuration map

**/

template<typename T>
typename StackedGatedModel<T>::config_t StackedGatedModel<T>::configuration() {
	config_t config;
	config["output_size"].emplace_back(to_string(output_size));
	config["input_size"].emplace_back(to_string(input_size));
	config["memory_penalty"].emplace_back(to_string(memory_penalty));
	config["vocabulary_size"].emplace_back(to_string(vocabulary_size));
	for (auto& v : hidden_sizes)
		config["hidden_sizes"].emplace_back(to_string(v));
	return config;
}

/**
Save Configuration
------------------

Save model configuration as a text file with key value pairs.
Values are vectors of string, and keys are known by the model.

Input
-----

std::string fname : where to save the configuration

**/
template<typename T>
void StackedGatedModel<T>::save_configuration(std::string fname) {
	
	auto config = configuration();
	utils::map_to_file(config, fname);
}

template<typename T>
void StackedGatedModel<T>::save(std::string dirname) {
	utils::ensure_directory(dirname);
	// Save the matrices:
	auto params = parameters();
	utils::save_matrices(params, dirname);
	dirname += "config.md";
	save_configuration(dirname);
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

StackedGatedModel<T> model : the saved model

**/
template<typename T>
StackedGatedModel<T> StackedGatedModel<T>::load(std::string dirname) {
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
	auto model =  StackedGatedModel<T>(config);

	// get the current parameters of the model.
	auto params = model.parameters();

	// get the new parameters from the saved numpy files
	utils::load_matrices(params, dirname);

	return model;
}

template<typename T>
std::tuple<T, T> StackedGatedModel<T>::cost_fun(
	graph_t& G,
	shared_index_mat data,
	shared_eigen_index_vector start_loss,
	shared_eigen_index_vector codelens,
	uint offset) {

	auto initial_state    = lstm::initial_states(hidden_sizes);
	auto num_hidden_sizes = hidden_sizes.size();

	shared_mat input_vector;
	shared_mat memory;
	shared_mat logprobs;
	// shared_mat probs;
	std::tuple<T, T> cost(0.0, 0.0);

	auto n = data->cols();
	for (uint i = 0; i < n-1; ++i) {
		// pick this letter from the embedding
		input_vector = G.rows_pluck(embedding, data->col(i));
		memory = gate.activate(G, input_vector, initial_state.second[0]);
		memory->set_name("Memory Gated Input");
		input_vector = G.eltmul_broadcast_rowwise(input_vector, memory);
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
		std::get<0>(cost) += masked_cross_entropy(
			logprobs,
			i,
			start_loss,
			codelens,
			(data->col(i+1).array() - offset).matrix());

		std::get<1>(cost) += masked_sum(memory, i, 0, start_loss, memory_penalty);
	}
	return cost;
}

// Private method that names the parameters
// For better debugging and reference
template<typename T>
void StackedGatedModel<T>::name_parameters() {
	embedding->set_name("Embedding");
	decoder.W->set_name("Decoder W");
	decoder.b->set_name("Decoder Bias");
}

// Private method for building lstm cells:
template<typename T>
void StackedGatedModel<T>::construct_LSTM_cells() {
	cells = StackedCells<lstm>(input_size, hidden_sizes);
}

template<typename T>
StackedGatedModel<T>::StackedGatedModel (int _vocabulary_size, int _input_size, int hidden_size, int _stack_size, int _output_size, T _memory_penalty)
	:
	input_size(_input_size),
	output_size(_output_size),
	vocabulary_size(_vocabulary_size),
	memory_penalty(_memory_penalty),
	stack_size(_stack_size),
	gate(_input_size, hidden_size),
	decoder(hidden_size, _output_size) {

	embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
	for (int i = 0; i < stack_size;i++)
		hidden_sizes.emplace_back(hidden_size);
	construct_LSTM_cells();
	name_parameters();
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

std::unordered_map<std::string, std::vector<std::string>& config : model hyperparameters

**/
template<typename T>
StackedGatedModel<T>::StackedGatedModel (
	const typename StackedGatedModel<T>::config_t& config)
	:
	memory_penalty(from_string<T>(config.at("memory_penalty")[0])),
	vocabulary_size(from_string<int>(config.at("vocabulary_size")[0])),
	output_size(from_string<int>(config.at("output_size")[0])),
	input_size(from_string<int>(config.at("input_size")[0])),
	stack_size(config.at("hidden_sizes").size()),
	decoder(
		from_string<int>(config.at("hidden_sizes")[config.at("hidden_sizes").size()-1]),
		from_string<int>(config.at("output_size")[0])),
	gate(
		from_string<int>(config.at("input_size")[0]),
		from_string<int>(config.at("hidden_sizes")[config.at("hidden_sizes").size()-1]))
{
	embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
	for (auto& v : config.at("hidden_sizes"))
		hidden_sizes.emplace_back(from_string<int>(v));

	construct_LSTM_cells();
	name_parameters();
}

template<typename T>
StackedGatedModel<T>::StackedGatedModel (int _vocabulary_size, int _input_size, int _output_size, std::vector<int>& _hidden_sizes, T _memory_penalty)
	:
	input_size(_input_size),
	output_size(_output_size),
	vocabulary_size(_vocabulary_size),
	memory_penalty(_memory_penalty),
	stack_size(_hidden_sizes.size()),
	hidden_sizes(_hidden_sizes),
	gate(_input_size, _hidden_sizes[0]),
	decoder(_hidden_sizes[_hidden_sizes.size()-1], _output_size) {

	embedding = make_shared<mat>(vocabulary_size, input_size, (T) -0.05, (T) 0.05);
	construct_LSTM_cells();
	name_parameters();
}

// Nested Templates !!
template<typename T>
template<typename K>
std::vector<int> StackedGatedModel<T>::reconstruct_fun(
    K example,
    utils::Vocab& lookup_table,
    int eval_steps,
    int symbol_offset) {

	graph_t G(false);
	shared_mat input_vector;
	shared_mat memory;
	auto initial_state = lstm::initial_states(hidden_sizes);
	auto n = example.cols() * example.rows();
	for (uint i = 0; i < n; ++i) {
		// pick this letter from the embedding
		input_vector  = G.row_pluck(embedding, example(i));
		memory        = gate.activate(G, input_vector, initial_state.second[0]);
		input_vector  = G.eltmul_broadcast_rowwise(input_vector, memory);
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// decoder takes as input the final hidden layer's activation:
	}
	vector<int> outputs;
	auto last_symbol = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
	outputs.emplace_back(last_symbol);
	last_symbol += symbol_offset;
	for (uint j = 0; j < eval_steps - 1; j++) {
		input_vector  = G.row_pluck(embedding, last_symbol);
		memory        = gate.activate(G, input_vector, initial_state.second[0]);
		input_vector  = G.eltmul_broadcast_rowwise(input_vector, memory);
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		last_symbol   = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
		outputs.emplace_back(last_symbol);
		last_symbol += symbol_offset;
	}
	return outputs;
}

typedef Eigen::Block< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, 1, Eigen::Dynamic, !Eigen::RowMajor> index_row;
typedef Eigen::VectorBlock< index_row, Eigen::Dynamic> sliced_row;

typedef Eigen::Block< Eigen::Matrix<uint, Eigen::Dynamic, Eigen::Dynamic>, Eigen::Dynamic, 1, !Eigen::RowMajor> index_col;
typedef Eigen::VectorBlock< index_col, Eigen::Dynamic> sliced_col;

template std::vector<int> StackedGatedModel<float>::reconstruct_fun(sliced_row, utils::Vocab&, int, int);
template std::vector<int> StackedGatedModel<double>::reconstruct_fun(sliced_row, utils::Vocab&, int, int);

template std::vector<int> StackedGatedModel<float>::reconstruct_fun(index_row, utils::Vocab&, int, int);
template std::vector<int> StackedGatedModel<double>::reconstruct_fun(index_row, utils::Vocab&, int, int);

template std::vector<int> StackedGatedModel<float>::reconstruct_fun(sliced_col, utils::Vocab&, int, int);
template std::vector<int> StackedGatedModel<double>::reconstruct_fun(sliced_col, utils::Vocab&, int, int);

template std::vector<int> StackedGatedModel<float>::reconstruct_fun(index_col, utils::Vocab&, int, int);
template std::vector<int> StackedGatedModel<double>::reconstruct_fun(index_col, utils::Vocab&, int, int);

template class StackedGatedModel<float>;
template class StackedGatedModel<double>;