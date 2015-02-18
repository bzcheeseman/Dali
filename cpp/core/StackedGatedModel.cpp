#include "StackedGatedModel.h"

DEFINE_double(memory_penalty, 0.3, "L1 Penalty on Input Gate activation.");

using std::shared_ptr;
using std::vector;
using std::make_shared;
using std::ofstream;
using std::to_string;
using std::stringstream;
using std::string;

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

std::map<std::string, std::vector< std::string >> config : configuration map

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

template<typename T>
StackedGatedModel<T> StackedGatedModel<T>::build_from_CLI(int vocab_size, int output_size, bool verbose) {
	if (verbose)
		std::cout << "Load location         = " << ((FLAGS_load == "") ? "N/A" : FLAGS_load)  << std::endl;
	// Load or Construct the model
	auto model = (FLAGS_load != "") ?
		StackedGatedModel<T>::load(FLAGS_load) :
		StackedGatedModel<T>(
			vocab_size,
			FLAGS_input_size,
			FLAGS_hidden,
			FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size,
			output_size,
			FLAGS_memory_penalty);
	if (verbose) {
		std::cout << ((FLAGS_load == "") ? "Constructed Stacked LSTMs" : "Loaded Model") << std::endl;
		std::cout << "Vocabulary size       = " << model.embedding->n      << std::endl;
		std::cout << "Input size            = " << model.input_size        << std::endl;
		std::cout << "Output size           = " << model.output_size       << std::endl;
		std::cout << "Stack size            = " << model.stack_size        << std::endl;
		std::cout << "Memory Penalty        = " << model.memory_penalty    << std::endl;
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
std::tuple<T, T> StackedGatedModel<T>::masked_predict_cost(
	graph_t& G,
	shared_index_mat data,
	shared_index_mat target_data,
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
		input_vector = G.eltmul_broadcast_rowwise(input_vector, memory);
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = decoder.activate(G, initial_state.second[num_hidden_sizes-1]);

		std::get<0>(cost) += G.needs_backprop ? masked_cross_entropy(
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
		std::get<1>(cost) += G.needs_backprop ? masked_sum(
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

template<typename T>
std::tuple<T, T> StackedGatedModel<T>::masked_predict_cost(
	graph_t& G,
	shared_index_mat data,
	shared_index_mat target_data,
	uint start_loss,
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
		memory = gate.activate(G, input_vector, initial_state.second[stack_size-1]);
		input_vector = G.eltmul_broadcast_rowwise(input_vector, memory);
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = decoder.activate(G, initial_state.second[stack_size-1]);
		std::get<0>(cost) += G.needs_backprop ? masked_cross_entropy(
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
		std::get<1>(cost) += G.needs_backprop ? masked_sum(
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

// Private method that names the parameters
// For better debugging and reference
template<typename T>
void StackedGatedModel<T>::name_parameters() {
	embedding->set_name("Embedding");
	decoder.W->set_name("Decoder W");
	decoder.b->set_name("Decoder Bias");
}

/**
Construct LSTM Cells (private)
------------------------------

Construct LSTM cells using the provided hidden sizes and
the input size to the Stacked LSTMs.

**/
template<typename T>
void StackedGatedModel<T>::construct_LSTM_cells() {
	cells = StackedCells<lstm>(input_size, hidden_sizes);
}

/**
Construct LSTM Cells (private)
------------------------------

Constructs cells using either deep or shallow copies from
other cells.

Inputs
------

const std::vector<LSTM<T>>& cells : cells for copy
                      bool copy_w : should each LSTM copy the parameters or share them
                     bool copy_dw : should each LSTM copy the gradient memory `dw` or share it.


**/
template<typename T>
void StackedGatedModel<T>::construct_LSTM_cells(const vector<StackedGatedModel<T>::lstm>& _cells, bool copy_w, bool copy_dw) {
	cells = StackedCells<lstm>(_cells, copy_w, copy_dw);
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

std::map<std::string, std::vector<std::string>& config : model hyperparameters

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

/**
StackedGatedModel<T>::StackedGatedModel
-------------

Copy constructor with option to make a shallow
or deep copy of the underlying parameters.

If the copy is shallow then the parameters are shared
but separate gradients `dw` are used for each of
thread StackedGatedModel<T>.

Shallow copies are useful for Hogwild and multithreaded
training

See `Mat<T>::shallow_copy`, `examples/character_prediction.cpp`,
`StackedGatedModel<T>::shallow_copy`

Inputs
------

      StackedGatedModel<T> l : StackedGatedModel from which to source parameters and dw
    bool copy_w : whether parameters for new StackedGatedModel should be copies
                  or shared
   bool copy_dw : whether gradients for new StackedGatedModel should be copies
                  shared (Note: sharing `dw` should be used with
                  caution and can lead to unpredictable behavior
                  during optimization).

Outputs
-------

StackedGatedModel<T> out : the copied StackedGatedModel with deep or shallow copy of parameters

**/
template<typename T>
StackedGatedModel<T>::StackedGatedModel (const StackedGatedModel<T>& model, bool copy_w, bool copy_dw) :
    input_size(model.input_size),
	output_size(model.output_size),
	vocabulary_size(model.vocabulary_size),
	memory_penalty(model.memory_penalty),
	stack_size(model.stack_size),
	hidden_sizes(model.hidden_sizes),
	gate(model.gate, copy_w, copy_dw),
	decoder(model.decoder, copy_w, copy_dw)
    {
    embedding = make_shared<mat>(*model.embedding, copy_w, copy_dw);
    construct_LSTM_cells(model.cells, copy_w, copy_dw);
    name_parameters();
}

/**
Shallow Copy
------------

Perform a shallow copy of a StackedGatedModel<T> that has
the same parameters but separate gradients `dw`
for each of its parameters.

Shallow copies are useful for Hogwild and multithreaded
training

See `StackedGatedModel<T>::shallow_copy`, `examples/character_prediction.cpp`.

Outputs
-------

StackedGatedModel<T> out : the copied layer with sharing parameters,
                           but with separate gradients `dw`

**/
template<typename T>
StackedGatedModel<T> StackedGatedModel<T>::shallow_copy() const {
    return StackedGatedModel<T>(*this, false, true);
}

template<typename T>
template<typename K>
typename StackedGatedModel<T>::lstm_activation_t StackedGatedModel<T>::get_final_activation(
	graph_t& G,
	const K& example) {
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

Graph<T>& G : computation graph
std::pair<std::vector<std::shared_ptr<Mat<T>>>, std::vector<std::shared_ptr<Mat<T>>>>& : previous state
uint index : embedding observation

Outputs
-------

std::tuple<std::pair<vector<shared_ptr<Mat<T>>>, vector<shared_ptr<Mat<T>>>>, shared_ptr<Mat<T>>, T> out :
    pair of LSTM hidden and cell states, and probabilities from the decoder, and memory usage.

**/
template<typename T>
typename StackedGatedModel<T>::activation_t StackedGatedModel<T>::activate(
	graph_t& G,
	lstm_activation_t& previous_state,
	const uint& index) {
	activation_t out;

	auto input_vector = G.row_pluck(embedding, index);
	auto memory       = gate.activate(G, input_vector, previous_state.second[stack_size-1]);
	input_vector  = G.eltmul_broadcast_rowwise(input_vector, memory);

	std::get<0>(out) = forward_LSTMs(G, input_vector, previous_state, cells);
	std::get<1>(out) = softmax(decoder.activate(G, std::get<0>(out).second[stack_size-1]));
	std::get<2>(out) = memory->w(0);

	return out;
}

// Nested Templates !!
template<typename T>
template<typename K>
std::vector<int> StackedGatedModel<T>::reconstruct(
    K example,
    int eval_steps,
    int symbol_offset) {

	graph_t G(false);
	auto initial_state = get_final_activation(G, example);

	shared_mat input_vector;
	shared_mat memory;
	vector<int> outputs;
	auto last_symbol = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
	outputs.emplace_back(last_symbol);
	last_symbol += symbol_offset;
	for (uint j = 0; j < eval_steps - 1; j++) {
		input_vector  = G.row_pluck(embedding, last_symbol);
		memory        = gate.activate(G, input_vector, initial_state.second[stack_size-1]);
		input_vector  = G.eltmul_broadcast_rowwise(input_vector, memory);
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		last_symbol   = argmax(decoder.activate(G, initial_state.second[stack_size-1]));
		outputs.emplace_back(last_symbol);
		last_symbol += symbol_offset;
	}
	return outputs;
}

template<typename T>
template<typename K>
std::vector<utils::OntologyBranch::shared_branch> StackedGatedModel<T>::reconstruct_lattice(
    K example,
    utils::OntologyBranch::shared_branch root,
    int eval_steps) {

	graph_t G(false);
	shared_mat input_vector;
	shared_mat memory;
	auto pos = root;
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
		memory        = gate.activate(G, input_vector, initial_state.second[0]);
		input_vector  = G.eltmul_broadcast_rowwise(input_vector, memory);
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		last_turn     = argmax_slice(decoder.activate(G, initial_state.second[stack_size-1]), 0, pos->children.size() + 1);
		pos           = (last_turn == 0) ? root : pos->children[last_turn-1];
		outputs.emplace_back(pos);
	}
	return outputs;
}

// Nested Templates !!
template<typename T>
template<typename K>
string StackedGatedModel<T>::reconstruct_string(
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
string StackedGatedModel<T>::reconstruct_lattice_string(
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

template string StackedGatedModel<float>::reconstruct_string(sliced_row, const utils::Vocab&, int, int);
template string StackedGatedModel<double>::reconstruct_string(sliced_row, const utils::Vocab&, int, int);

template string StackedGatedModel<float>::reconstruct_string(index_row, const utils::Vocab&, int, int);
template string StackedGatedModel<double>::reconstruct_string(index_row, const utils::Vocab&, int, int);

template string StackedGatedModel<float>::reconstruct_string(sliced_col, const utils::Vocab&, int, int);
template string StackedGatedModel<double>::reconstruct_string(sliced_col, const utils::Vocab&, int, int);

template string StackedGatedModel<float>::reconstruct_string(index_col, const utils::Vocab&, int, int);
template string StackedGatedModel<double>::reconstruct_string(index_col, const utils::Vocab&, int, int);

template vector<int> StackedGatedModel<float>::reconstruct(sliced_row, int, int);
template vector<int> StackedGatedModel<double>::reconstruct(sliced_row, int, int);

template vector<int> StackedGatedModel<float>::reconstruct(index_row, int, int);
template vector<int> StackedGatedModel<double>::reconstruct(index_row, int, int);

template vector<int> StackedGatedModel<float>::reconstruct(sliced_col, int, int);
template vector<int> StackedGatedModel<double>::reconstruct(sliced_col, int, int);

template vector<int> StackedGatedModel<float>::reconstruct(index_col, int, int);
template vector<int> StackedGatedModel<double>::reconstruct(index_col, int, int);

template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<float>::reconstruct_lattice(sliced_row, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<double>::reconstruct_lattice(sliced_row, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<float>::reconstruct_lattice(index_row, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<double>::reconstruct_lattice(index_row, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<float>::reconstruct_lattice(sliced_col, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<double>::reconstruct_lattice(sliced_col, utils::OntologyBranch::shared_branch, int);

template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<float>::reconstruct_lattice(index_col, utils::OntologyBranch::shared_branch, int);
template vector<utils::OntologyBranch::shared_branch> StackedGatedModel<double>::reconstruct_lattice(index_col, utils::OntologyBranch::shared_branch, int);

template string StackedGatedModel<float>::reconstruct_lattice_string(sliced_row, utils::OntologyBranch::shared_branch, int);
template string StackedGatedModel<double>::reconstruct_lattice_string(sliced_row, utils::OntologyBranch::shared_branch, int);

template string StackedGatedModel<float>::reconstruct_lattice_string(index_row, utils::OntologyBranch::shared_branch, int);
template string StackedGatedModel<double>::reconstruct_lattice_string(index_row, utils::OntologyBranch::shared_branch, int);

template string StackedGatedModel<float>::reconstruct_lattice_string(sliced_col, utils::OntologyBranch::shared_branch, int);
template string StackedGatedModel<double>::reconstruct_lattice_string(sliced_col, utils::OntologyBranch::shared_branch, int);

template string StackedGatedModel<float>::reconstruct_lattice_string(index_col, utils::OntologyBranch::shared_branch, int);
template string StackedGatedModel<double>::reconstruct_lattice_string(index_col, utils::OntologyBranch::shared_branch, int);

template class StackedGatedModel<float>;
template class StackedGatedModel<double>;
