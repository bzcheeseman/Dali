#include <gflags/gflags.h>

#include "dali/core.h"
#include "dali/utils.h"

// test file for stacked LSTM cells:
using std::vector;

typedef float REAL_t;
typedef LSTM<REAL_t> lstm;
typedef Mat<REAL_t>   mat;


DEFINE_int32(timesteps, 2, "How many steps to run the simulation for ?");
DEFINE_int32(batch_size, 2, "How big is the batch ?");
DEFINE_int32(input_size, 50, "How big is input size ?");
DEFINE_double(std, 2.0, "Standard deviation of random data.");
DEFINE_int32(hidden_size, 100, "How big is the hidden size ?");
DEFINE_int32(stack_size, 3, "How many LSTMs are stacked at each time step ?");

int main (int argc, char *argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Backprop stacked LSTMs\n"
        "----------------------\n"
        "Test backprop on random data.\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date February 18th 2015\n"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    vector<int> hidden_sizes;
    for (int i=0; i < FLAGS_stack_size; ++i)
        hidden_sizes.emplace_back(FLAGS_hidden_size);

    auto cells = StackedCells<lstm>(FLAGS_input_size, hidden_sizes, true, true);
    auto initial_state = lstm::initial_states(hidden_sizes);

    auto input_vector = mat(FLAGS_input_size, FLAGS_batch_size, (float)FLAGS_std);

    for (auto i = 0; i < FLAGS_timesteps; ++i)
        initial_state = forward_LSTMs(input_vector, initial_state, cells);

    // backpropagate
    graph::backward();
}
