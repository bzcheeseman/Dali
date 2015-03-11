#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstring>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>

#include "core/Mat.h"
#include "core/Reporting.h"
#include "core/Seq.h"
#include "core/Layers.h"
#include "core/SaneCrashes.h"
#include "core/utils.h"

typedef double R;

using std::bitset;
using std::chrono::seconds;
using std::make_shared;
using std::max;
using std::vector;


int main( int argc, char* argv[]) {
    sane_crashes::activate();

    GFLAGS_NAMESPACE::SetUsageMessage(
    "RNN Kindergarden - Lesson 2 - RNN learning to add binary numbers.");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Throttled throttled;

    // How many iterations of gradient descent to run.
    const int NUM_EPOCHS = 100000;
    const int ITERATIONS_PER_EPOCH = 30;
    // Biggest number to add.
    const int NUM_BITS = 30;
    // What is the learning rate.
    double LR = 0.1 ;

    const int SEED = 80085;
    const int INPUT_SIZE = 2;
    const int OUTPUT_SIZE = 1;
    const int MEMORY_SIZE = 5;

    // Initialize random number generator.
    srand(SEED);

    // TIP: Since we are doing output = map * input, we put output
    //      dimension first.
    RNN<R>    rnn(INPUT_SIZE, MEMORY_SIZE);
    Layer<R>  classifier(MEMORY_SIZE, OUTPUT_SIZE);
    Mat<R>    rnn_initial(MEMORY_SIZE, 1);

    auto rnn_params = rnn.parameters();
    auto classifier_params = classifier.parameters();
    vector<Mat<R>> params;
    params.push_back(rnn_initial);
    params.insert(params.end(), rnn_params.begin(), rnn_params.end());
    params.insert(params.end(), classifier_params.begin(), classifier_params.end());

    for (int epoch = 0; epoch <= NUM_EPOCHS; ++epoch) {
        // Cross entropy bit error
        double epoch_error = 0;
        // Average number of bits wrong.
        double epoch_bit_error = 0;
        int a, b, res, predicted_res;
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

            Mat<R> prev_hidden = rnn_initial;
            Mat<R> error(1,1);
            for (int i=0; i< NUM_BITS; ++i) {
                Mat<R> input_i(INPUT_SIZE, 1);
                input_i.w()(0,0) = a_bits[i];
                input_i.w()(1,0) = b_bits[i];

                prev_hidden = rnn.activate(input_i, prev_hidden).tanh();
                Mat<R> output_i = classifier.activate(prev_hidden);
                predicted_res_bits[i] = output_i.w()(0,0) < 0.5 ? 0 : 1;
                epoch_bit_error += res_bits[i] != predicted_res_bits[i];

                // error = error + MatOps<R>::sigmoid_binary_cross_entropy(output_i, (R)res_bits[i]);
                // error = error + MatOps<R>::binary_cross_entropy(output_i.sigmoid(), (R)res_bits[i]);
                // error = error + (output_i.sigmoid() - (R)res_bits[i]).square();
            }
            predicted_res = predicted_res_bits.to_ulong();
            epoch_error += error.w()(0,0);

            error.grad();
            graph::backward();
        }

        for (auto param: params) {
            param.w() -= (LR / ITERATIONS_PER_EPOCH) * param.dw();
            param.dw().fill(0);
        }


        throttled.maybe_run(seconds(2), [&]() {
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
        });
    }
}
