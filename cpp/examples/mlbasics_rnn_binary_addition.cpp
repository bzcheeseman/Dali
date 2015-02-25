#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cstring>
#include <gflags/gflags.h>
#include <iostream>
#include <memory>
#include <vector>

#include "core/Graph.h"
#include "core/Mat.h"
#include "core/Reporting.h"

typedef Mat<double> mat;
typedef std::shared_ptr<mat> shared_mat;

using std::bitset;
using std::chrono::seconds;
using std::make_shared;
using std::max;
using std::vector;

class AffineMap {
    int input_size;
    int output_size;
    shared_mat mult;
    shared_mat bias;

    public:
        AffineMap(int input_size, int output_size, double bound=0.2) :
                input_size(input_size),
                output_size(output_size) {
            mult = make_shared<mat>(output_size, input_size, -bound/2.0, bound/2.0);
            bias = make_shared<mat>(output_size, 1,          -bound/2.0, bound/2.0);
        }

        shared_mat f(Graph<double>& G, shared_mat input) {
            return G.add(G.mul(mult, input), bias);
        }

        void push_params(vector<shared_mat>& destination) const {
            for(auto& param: {mult, bias})
                destination.push_back(param);
        }
};

class RnnMap {
    int input_size;
    int output_size;
    int memory_size;

    AffineMap input_map;
    AffineMap output_map;
    AffineMap memory_map;
    shared_mat first_memory;

    shared_mat prev_memory;

    public:
        RnnMap(int input_size, int output_size, int memory_size, double bound=0.2) :
                input_size(input_size),
                output_size(output_size),
                memory_size(memory_size),
                input_map(input_size, memory_size, bound),
                output_map(memory_size, output_size, bound),
                memory_map(memory_size, memory_size, bound) {
            first_memory = make_shared<mat>(memory_size, 1, -bound/2.0, bound/2.0);
            reset();
        }

        void reset() {
            prev_memory = first_memory;
        }

        // output is in range 0, 1
        shared_mat f(Graph<double>& G, shared_mat input) {
            shared_mat memory_in;
            memory_in = memory_map.f(G, prev_memory);
            shared_mat input_in = input_map.f(G, input);

            shared_mat memory = G.tanh(G.add(input_in, memory_in));

            prev_memory = memory;

            return G.sigmoid(output_map.f(G, memory));
        }

        void push_params(vector<shared_mat>& destination) const {
            destination.push_back(first_memory);

            for(const AffineMap& affine_map: {input_map, memory_map, output_map})
                affine_map.push_params(destination);
        }

};


int main( int argc, char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
    "RNN Kindergarden - Lesson 2 - RNN learning to add binary numbers.");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Throttled throttled;
    shared_mat one = make_shared<mat>(1, 1);
    one->w(0,0) = 1.0;
    shared_mat zero = make_shared<mat>(1, 1);
    zero->w(0,0) = 0.0;
    shared_mat eps = make_shared<mat>(1, 1);
    eps->w(0,0) = 1e-5;

    // How many iterations of gradient descent to run.
    const int NUM_EPOCHS = 100000;
    const int ITERATIONS_PER_EPOCH = 30;
    // Biggest number to add.
    const int NUM_BITS = 30;
    // What is the learning rate.
    double LR = 0.01;

    const int SEED = 80085;
    const int INPUT_SIZE = 2;
    const int OUTPUT_SIZE = 1;
    const int MEMORY_SIZE = 10;
    const int HIDDEN_SIZE = 5;

    // Initialize random number generator.
    srand(SEED);

    // TIP: Since we are doing output = map * input, we put output
    //      dimension first.
    RnnMap rnn(INPUT_SIZE, HIDDEN_SIZE, MEMORY_SIZE);
    RnnMap rnn2(HIDDEN_SIZE, OUTPUT_SIZE, MEMORY_SIZE);

    vector<shared_mat> params;
    rnn.push_params(params);
    rnn2.push_params(params);

    Solver::AdaDelta<double> solver(params);

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
            predicted_res_bits =    bitset<NUM_BITS>(predicted_res);

            Graph<double> G(true);
            rnn.reset();
            rnn2.reset();

            shared_mat error = make_shared<mat>(1, 1);

            error->w.fill(0);


            for (int i=0; i< NUM_BITS; ++i) {
                shared_mat input_i = make_shared<mat>(INPUT_SIZE, 1);
                input_i->w(0,0) = a_bits[i];
                input_i->w(1,0) = b_bits[i];

                auto expected_output_i = make_shared<mat>(OUTPUT_SIZE, 1);
                expected_output_i->w(0,0) = res_bits[i];

                shared_mat hidden_i = rnn.f(G, input_i);
                shared_mat output_i = rnn2.f(G, hidden_i);

                predicted_res_bits[i] = output_i->w(0,0) < 0.5 ? 0 : 1;

                shared_mat partial_error;
                if (res_bits[i] == 1) {
                    partial_error = G.sub(zero, G.log(G.add(eps, output_i)));
                } else {
                    assert(res_bits[i] == 0);
                    partial_error = G.sub(zero, G.log(G.add(eps, G.sub(one, output_i))));
                }

                error = G.add(error, partial_error);
            }
            predicted_res = predicted_res_bits.to_ulong();

            for (int i=0; i<NUM_BITS; ++i)
                epoch_bit_error += res_bits[i] != predicted_res_bits[i];
            epoch_error += error->w(0,0);
            error->grad();
            G.backward();
        }
        solver.step(params, 0.0);

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
