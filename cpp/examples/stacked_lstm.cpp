#include "core/Layers.h"

// test file for stacked LSTM cells:
using std::vector;
using std::make_shared;
using utils::assign_cli_argument;

typedef float REAL_t;
typedef LSTM<REAL_t> lstm;
typedef Graph<REAL_t> graph_t;
typedef Mat<REAL_t> mat;

int main (int argc, char *argv[]) {
	auto batch_size          = 2;
	auto input_size          = 50;
	auto timesteps           = 2;
	REAL_t std               = 2.0;
	vector<int> hidden_sizes = {100, 100, 100};

	if (argc > 1) assign_cli_argument(argv[1], timesteps,       "timesteps");
	if (argc > 2) assign_cli_argument(argv[2], batch_size,      "batch size");
	if (argc > 3) assign_cli_argument(argv[3], input_size,      "input size");
	if (argc > 4) assign_cli_argument(argv[4], std,             "standard deviation");
	if (argc > 5) assign_cli_argument(argv[5], hidden_sizes[0], "hidden size 1");
	if (argc > 6) assign_cli_argument(argv[6], hidden_sizes[1], "hidden size 2");
	if (argc > 7) assign_cli_argument(argv[7], hidden_sizes[2], "hidden size 3");

	auto cells = StackedCells<lstm>(input_size, hidden_sizes);
	auto initial_state = lstm::initial_states(hidden_sizes);
	graph_t G;

	auto input_vector = make_shared<mat>(input_size, batch_size, std);

	for (auto i = 0; i < timesteps; ++i)
		initial_state = forward_LSTMs(G, input_vector, initial_state, cells);

	// backpropagate
	G.backward();
}
