#include "Layers.h"
#include <sstream>

// test file for stacked LSTM cells:
using std::pair;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::cerr;
using std::istringstream;

typedef float REAL_t;
typedef LSTM<REAL_t> lstm;
typedef Graph<REAL_t> graph_t;
typedef Mat<REAL_t> mat;
typedef shared_ptr<mat> shared_mat;
typedef vector<shared_mat> cell_outputs;
typedef pair<cell_outputs, cell_outputs> paired_cell_outputs;

paired_cell_outputs forward_lstms(
	graph_t& G,
	shared_mat input_vector,
	paired_cell_outputs& previous_state,
	vector<lstm>& cells) {

	auto previous_state_cells = previous_state.first;
	auto previous_state_hiddens = previous_state.second;

	auto cell_iter = previous_state_cells.begin();
	auto hidden_iter = previous_state_hiddens.begin();

	paired_cell_outputs out_state;
	out_state.first.reserve(cells.size());
	out_state.second.reserve(cells.size());

	auto layer_input = input_vector;

	for (auto& layer : cells) {

		auto layer_out = layer.activate(G, layer_input, *cell_iter, *hidden_iter);

		out_state.first.push_back(layer_out.first);
		out_state.second.push_back(layer_out.second);

		++cell_iter;
		++hidden_iter;

		layer_input = layer_out.second;
	}

	return out_state;
}

template<typename T>
void assign_cli_argument(char * source, T& target, T default_val, std::string variable_name ) {
	// Takes an input, a default value, and tries to extract from a character sequence
	// an assignment. If it fails it notifies the user and switches back to the default.
	// Default is copied so a copy is an original is always available
	// for fallback (even if target and default originated from the same place).
	istringstream ss(source);
	if (!(ss >> target)) {
	    cerr << "Invalid " << variable_name << " => \""<< source << "\"\n";
	    cerr << "Using default (" << default_val << ") instead\n";
	    target = default_val;
	}
}

int main (int argc, char *argv[]) {
	auto batch_size          = 2;
	auto input_size          = 50;
	auto timesteps           = 2;
	REAL_t std               = 2.0;
	vector<int> hidden_sizes = {100, 100, 100};

	if (argc > 1) assign_cli_argument(argv[1], timesteps,       timesteps,       "timesteps");
	if (argc > 2) assign_cli_argument(argv[2], batch_size,      batch_size,      "batch size");
	if (argc > 3) assign_cli_argument(argv[3], input_size,      input_size,      "input size");
	if (argc > 4) assign_cli_argument(argv[4], std,             std,             "standard deviation");
	if (argc > 5) assign_cli_argument(argv[5], hidden_sizes[0], hidden_sizes[0], "hidden size 1");
	if (argc > 6) assign_cli_argument(argv[6], hidden_sizes[1], hidden_sizes[1], "hidden size 2");
	if (argc > 7) assign_cli_argument(argv[7], hidden_sizes[2], hidden_sizes[2], "hidden size 3");
	
	auto cells = StackedCells<lstm>(input_size, hidden_sizes);
	paired_cell_outputs initial_state = lstm::initial_states(hidden_sizes);
	graph_t G;

	auto input_vector = make_shared<mat>(input_size, batch_size, std);

	for (auto i = 0; i < timesteps; ++i)
		initial_state = forward_lstms(G, input_vector, initial_state, cells);
	
	// backpropagate
	G.backward();
}