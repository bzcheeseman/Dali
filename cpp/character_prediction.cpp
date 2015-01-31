#include "Layers.h"
#include <fstream>
#include  <random>
#include  <iterator>
// test file for character prediction
using std::vector;
using std::make_shared;
using std::shared_ptr;
using utils::assign_cli_argument;
using std::fstream;

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

REAL_t cost_fun(
	graph_t& G,
	vector<int>& hidden_sizes,
    vector<lstm>& cells,
    shared_mat embedding,
    classifier_t& classifier,
    vector<int>& indices) {

	auto initial_state = lstm::initial_states(hidden_sizes);
	auto num_hidden_sizes = hidden_sizes.size();

	shared_mat input_vector;
	shared_mat logprobs;
	shared_mat probs;

	REAL_t cost = 0.0;
	auto n = indices.size();

	for (int i = 0; i < n-1; ++i) {
		// pick this letter from the embedding
		input_vector  = G.row_pluck(embedding, indices[i]);
		// pass this letter to the LSTM for processing
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);
		// classifier takes as input the final hidden layer's activation:
		logprobs      = classifier.activate(G, initial_state.second[num_hidden_sizes-1]);
		cost -= cross_entropy(logprobs, indices[i+1]);
	}
	return cost / (n-1);
}



int main (int argc, char *argv[]) {
	auto epochs              = 101;
	auto input_size          = 5;
	auto report_frequency    = 100;
	REAL_t std               = 0.1;
	vector<int> hidden_sizes = {20, 20};

	if (argc > 1) assign_cli_argument(argv[1], epochs,          "epochs");
	if (argc > 2) assign_cli_argument(argv[2], input_size,      "input size");
	if (argc > 3) assign_cli_argument(argv[3], std,             "standard deviation");
	if (argc > 4) assign_cli_argument(argv[4], hidden_sizes[0], "hidden size 1");
	if (argc > 5) assign_cli_argument(argv[5], hidden_sizes[1], "hidden size 2");


	auto vocab_size = 300;
	auto cells = StackedCells<lstm>(input_size, hidden_sizes);
	classifier_t classifier(hidden_sizes[hidden_sizes.size() - 1], vocab_size);
	auto embedding = make_shared<mat>(vocab_size, input_size, std);
	vector<shared_mat> parameters;
	parameters.push_back(embedding);
	auto classifier_params = classifier.parameters();
	parameters.insert(parameters.end(), classifier_params.begin(), classifier_params.end());

	for (auto& cell : cells) {
		auto cell_params = cell.parameters();
		parameters.insert(parameters.end(), cell_params.begin(), cell_params.end());
	}

	auto prepad = 0;
	auto postpad = vocab_size-1;
	auto sentences = get_character_sequences("../paulgraham_text.txt", prepad, postpad, vocab_size);

	static std::random_device rd;
    static std::mt19937 seed(rd());
    static std::uniform_int_distribution<> uniform(0, sentences.size() - 1);

	//Gradient descent optimizer:
	Solver<REAL_t> solver(parameters, 0.999, 1e-9, 5.0);
	
	// Main training loop:
	for (auto i = 0; i < epochs; ++i) {
		auto G = graph_t(true);      // create a new graph for each loop
		auto cost = cost_fun(
			G,                       // to keep track of computation
			hidden_sizes,            // to construct initial states
			cells,                   // LSTMs
			embedding,               // character embedding
			classifier,              // decoder for LSTM final hidden layer
			sentences[uniform(seed)] // the sequence to predict
		);
		G.backward();                // backpropagate
		// solve it.
		solver.step(parameters, 0.01, 0.0);
		if (i % report_frequency == 0)
			std::cout << "epoch (" << i << ") perplexity = " << cost << std::endl;
	}

	parameters[0]->npy_save("embedding.npy", "w");

	parameters[1]->npy_save("classifier_matrix.npy", "w");
	parameters[2]->npy_save("classifier_bias.npy", "w");
	
	// outputs:
	//> epoch (0) perplexity = -5.70376
	//> epoch (100) perplexity = -2.54203
}