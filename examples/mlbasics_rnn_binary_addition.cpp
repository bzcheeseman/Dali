#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstring>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>

#include "dali/core.h"
#include "dali/utils.h"

typedef double R;

using std::bitset;
using std::chrono::milliseconds;
using std::make_shared;
using std::max;
using std::vector;


int main( int argc, char* argv[]) {
    sane_crashes::activate();

    GFLAGS_NAMESPACE::SetUsageMessage(
    "RNN Kindergarden - Lesson 2 - RNN learning to add binary numbers.\n"
    "\n"
    "Here we will train recurrent neural network to add two numbers,  by looking\n"
    "at their binary representations. At each time-step we input one bit of first number\n"
    "one bit of second number and we output one bit of the result.\n");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Throttled throttled;

    // How many triples of form (a,b,a+b) do we use each training epoch
    const int ITERATIONS_PER_EPOCH = 30;
    // Biggest number to add.
    const int NUM_BITS = 30;
    // What is the learning rate.
    double LR = 0.1 ;

    const int SEED = 80085;

    // We have two bits as an input
    const int INPUT_SIZE = 2;
    // We have one bit of output
    const int OUTPUT_SIZE = 1;
    // How big is hidden state
    const int MEMORY_SIZE = 5;

    // Initialize random number generator.
    srand(SEED);

    // HINT: Since we are doing output = map * input, we put output
    //       dimension first.

    // Defines an RNN that takes input of size INPUT_SIZE and stores it in memory
    // Here rnn corresponds to a simple mapping H' = W * [X, H] + b
    // (where X is input, W,b are weights and bias, H, H' are hidden state
    // and new hidden state). Note it does not contain a nonlinearity!
    RNN<R>    rnn(INPUT_SIZE, MEMORY_SIZE);
    // Defines a mapping from memory to result bit. It computes a mapping Y=W*X+b
    Layer<R>  classifier(MEMORY_SIZE, OUTPUT_SIZE);
    // Defines initial state of a memory (HINT: it's also a parameter)
    Mat<R>    rnn_initial(1, MEMORY_SIZE);

    // For convenience we store all the parameters that we are going to update during training
    // in a vector params. Rnn and classifier are Layers and already provide a piece
    // of code that extracts all the relevant parameters.
    vector<Mat<R>> params;
    auto rnn_params = rnn.parameters();
    auto classifier_params = classifier.parameters();
    params.push_back(rnn_initial);
    params.insert(params.end(), rnn_params.begin(), rnn_params.end());
    params.insert(params.end(), classifier_params.begin(), classifier_params.end());

    uint patience = 0;

    Solver::SGD<R> solver(params);
    solver.step_size = FLAGS_learning_rate;

    for (int epoch = 0; ; ++epoch) {
        // Average cross entropy bit error per bit.
        double epoch_error = 0;
        // Average number of bits flipped.
        double epoch_bit_error = 0;

        // a, b are the numbers we are adding
        // res is basically a+b
        // predicted_res is the output of our neural network.
        int a, b, res, predicted_res;
        // bitwise representation of variables above.
        bitset<NUM_BITS> a_bits, b_bits, res_bits, predicted_res_bits;

        for (int iter = 0; iter < ITERATIONS_PER_EPOCH; ++iter) {
            a = rand()%(1<<(NUM_BITS-1));
            b = rand()%(1<<(NUM_BITS-1));
            res = a+b;
            predicted_res = 0;

            a_bits =                bitset<NUM_BITS> (a);
            b_bits =                bitset<NUM_BITS> (b);
            res_bits =              bitset<NUM_BITS> (res);
            predicted_res_bits =    bitset<NUM_BITS> (predicted_res);

            // Now we are going to iterate over all the bits and use the RNN
            // to compute the prediction. "prev_hidden" will always how the
            // previous hidden/memory state. "error" will accumulate cross
            // entropy error.
            Mat<R> prev_hidden = rnn_initial;
            Mat<R> error(1,1);

            for (int i=0; i< NUM_BITS; ++i) {
                // soon we will support constant and need for converting this
                // to matrix class will disappear.
                Mat<R> input_i(1, INPUT_SIZE);
                input_i.w(0) = a_bits[i];
                input_i.w(1) = b_bits[i];

                // advance RNN. We have a choice to use different nonlinearities,
                // but in this case we use tanh.
                prev_hidden = rnn.activate(input_i, prev_hidden).tanh();
                // compute the output bit, apply sigmoid to trap it in [0,1] interval.
                Mat<R> output_i = classifier.activate(prev_hidden).sigmoid();
                // output bit can be any number between 0, 1, so we round it.
                predicted_res_bits[i] = output_i.w(0) < 0.5 ? 0 : 1;

                // update errors
                epoch_bit_error += res_bits[i] != predicted_res_bits[i];
                error = error + MatOps<R>::binary_cross_entropy(output_i, (R)res_bits[i]);

                // Alternatively we could use square error - it converges slightly slower.
                // error = error + (output_i.sigmoid() - (R)res_bits[i]).square();
            }
            // Make sure we are looking at average error.
            error = error / (R)NUM_BITS;
            predicted_res = predicted_res_bits.to_ulong();
            epoch_error += error.w(0);
            // compute gradient
            error.grad();

            // apply backpropagation. This adds current gradient to already accumulated
            // gradient for every parameter.
            graph::backward();
        }

        epoch_error /= ITERATIONS_PER_EPOCH;

        solver.step(params);

        // Output status update every 500 ms.
        throttled.maybe_run(milliseconds(500), [&]() {
            epoch_error /= ITERATIONS_PER_EPOCH;
            epoch_bit_error /= ITERATIONS_PER_EPOCH;
            std::cout << "Epoch " << epoch << std::endl;
            std::cout << "        Argument1 " << a << "\t" << a_bits << std::endl;
            std::cout << "        Argument2 " << b << "\t" << b_bits << std::endl;
            std::cout << "        Predicted " << predicted_res << "\t"
                                              << predicted_res_bits << std::endl;
            std::cout << "        Expected  " << res << "\t"
                                              << res_bits << std::endl;
            std::cout << "    Training error: " << epoch_error << std::endl;
            std::cout << "    Average bits flipped: " << epoch_bit_error << std::endl;


            if (epoch_bit_error < 1e-6)
                patience += 1;
            else
                patience = 0; // bad chappie !! bad !!
        });


        // If for two consecutive epochs error was < 1e-6 break.
        if (patience >= 2) {
            std::cout << "       Training Complete      \n"
                         "    ┌───────────────────────┐ \n"
                         "    │Achievement Unlocked !!│ \n"
                         "    └───────────────────────┘ \n"
                         "                              \n";
            exit(0);
        }
    }
}
