#include <fstream>
#include <random>
#include <iostream>
#include <cstdio>
#include <thread>
#include <iterator>
#include "core/StackedModel.h"

// test file for character prediction
using std::vector;
using std::make_shared;
using std::shared_ptr;
using utils::assign_cli_argument;
using std::fstream;
using std::thread;

typedef float REAL_t;
typedef LSTM<REAL_t> lstm;
typedef Graph<REAL_t> graph_t;
typedef Layer<REAL_t> classifier_t;
typedef Mat<REAL_t> mat;
typedef shared_ptr<mat> shared_mat;

vector<vector<int>> get_character_sequences(const char* filename, int& prepad, int& postpad, int& vocab_size) {
	char ch;
	char linebreak = '\n';
	fstream file;
	file.open(filename);
	vector<vector<int>> lines;
	lines.emplace_back(2);
	vector<int>* line = &lines[0];
	line->push_back(prepad);
	while(file) {
		ch = file.get();
		if (ch == linebreak) {
			line->push_back(postpad);
			lines.emplace_back(2);
			line = &(lines.back());
			line->push_back(prepad);
			continue;
		}
		if (ch == EOF) {
			break;
		}
		line->push_back(std::min(vocab_size-1, (int)ch));
	}
	return lines;
}

template<typename T>
T validation_error(
	StackedModel<T>& model,
    vector<vector<int>>& data_set) {
	Graph<T> G(false);

	auto initial_state = lstm::initial_states(model.hidden_sizes);
	auto num_hidden_sizes = model.hidden_sizes.size();

	shared_mat input_vector;
	shared_mat logprobs;
	shared_mat probs;

	T cost = 0.0;
	for (auto& example: data_set) {
		auto n = example.size();
		REAL_t example_cost = 0.0;
		for (int i = 0; i < n-1; ++i) {
			// pick this letter from the embedding
			input_vector  = G.row_pluck(model.embedding, example[i]);
			// pass this letter to the LSTM for processing
			initial_state = forward_LSTMs(G, input_vector, initial_state, model.cells);
			// classifier takes as input the final hidden layer's activation:
			logprobs      = model.decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
			example_cost -= cross_entropy(logprobs, example[i+1]);

		}
		cost += example_cost / (n-1);
	}
	return cost / data_set.size();
}


template<typename T>
T cost_fun(
	Graph<T>& G,
	StackedModel<T>& model,
    vector<int>& indices) {

	auto initial_state = lstm::initial_states(model.hidden_sizes);
	auto num_hidden_sizes = model.hidden_sizes.size();

	shared_mat input_vector;
	shared_mat logprobs;
	shared_mat probs;

	T cost = 0.0;
	auto n = indices.size();

	for (int i = 0; i < n-1; ++i) {
		// pick this letter from the embedding
		input_vector  = G.row_pluck(model.embedding, indices[i]);
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, model.cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = model.decoder.activate(G, initial_state.second[num_hidden_sizes-1]);
		cost -= cross_entropy(logprobs, indices[i+1]);
	}
	return cost / (n-1);
}



int main (int argc, char *argv[]) {
	// Do not sync with stdio when using C++
	std::ios_base::sync_with_stdio(0);

	auto epochs           = 2000;
	auto input_size       = 5;
	auto report_frequency = 5;
	REAL_t std            = 0.1;
	auto hidden_size      = 20;
	auto vocab_size       = 300;
	auto num_threads      = 5;
	int  stack_size       = 2;
	auto minibatch_size   = 20;

	if (argc > 1) assign_cli_argument(argv[1], num_threads,     "num_threads");
	if (argc > 2) assign_cli_argument(argv[2], minibatch_size,  "minibatch_size");
	if (argc > 2) assign_cli_argument(argv[3], epochs,          "epochs");
	if (argc > 3) assign_cli_argument(argv[4], input_size,      "input size");
	if (argc > 5) assign_cli_argument(argv[5], hidden_size,     "hidden size");
	if (argc > 6) assign_cli_argument(argv[6], stack_size,      "stack size");
	if (argc > 7) assign_cli_argument(argv[7], vocab_size,      "vocab_size");

	auto model = StackedModel<REAL_t>(vocab_size, input_size, hidden_size, stack_size, vocab_size);
	auto parameters = model.parameters();

/*
	for (auto& param : parameters) {
		param->npy_load(stdin);
	}
*/
	auto prepad = 0;
	auto postpad = vocab_size-1;
	auto sentences = get_character_sequences("../paulgraham_text.txt", prepad, postpad, vocab_size);
	int train_size = (int)(sentences.size() * 0.9);
	int valid_size = sentences.size() - train_size;
	vector<vector<int>> train_set(sentences.begin(), sentences.begin() + train_size);
	vector<vector<int>> valid_set(sentences.begin() + train_size, sentences.end());

	static std::random_device rd;
    static std::mt19937 seed(rd());
    static std::uniform_int_distribution<> uniform(0, train_set.size() - 1);


	// Main training loop:
	REAL_t cost = 0.0;
	vector<thread> ts;

	int total_epochs = 0;

	Solver::AdaDelta<REAL_t> solver(parameters);
	// Solver::RMSProp<REAL_t> solver(thread_parameters, 0.999, 1e-9, 5.0);
	// Solver::SGD<REAL_t> solver;


	for (int t=0; t<num_threads; ++t) {
		ts.emplace_back([&](int thread_id) {
			auto thread_model = model.shallow_copy();

			auto thread_parameters = thread_model.parameters();




			for (auto i = 0; i < epochs / num_threads / minibatch_size; ++i) {
				auto G = graph_t(true);      // create a new graph for each loop

				for (auto mb = 0; mb < minibatch_size; ++mb)
					cost_fun(
						G,                       // to keep track of computation
						thread_model,            // what model should collect errors
						train_set[uniform(seed)] // the sequence to predict
					);
				G.backward();                // backpropagate

				// solve it.
				// RMS prop
				//solver.step(thread_parameters, 0.01, 0.0);
				// AdaDelta
				solver.step(thread_parameters, 0.0);
				// SGD
				// solver.step(thread_parameters, 0.3/minibatch_size, 0.0);
				if (++total_epochs % report_frequency == 0) {
					cost = validation_error(model, valid_set);

					std::cout << "epoch (" << total_epochs << ") perplexity = "
										  << std::fixed
		                                  << std::setw( 5 ) // keep 7 digits
		                                  << std::setprecision( 3 ) // use 3 decimals
		                                  << std::setfill( ' ' ) << cost << std::endl;
		        }
		    }
		}, t);

	}

	for(auto& t: ts) t.join();
/*
	for (auto& param : parameters) {
		param->npy_save(stdout);
	}
*/
	// utils::save_matrices(parameters, "paul_graham_params");

	// outputs:
	//> epoch (0) perplexity = -5.70376
	//> epoch (100) perplexity = -2.54203
}
