#include <cstdio>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>

#include "dali/core.h"
#include "dali/models/StackedModel.h"
#include "dali/utils/stacked_model_builder.h"

auto default_paul_graham_location = utils::dir_join({ STR(DALI_DATA_DIR) , "paul_graham", "train.txt" });


DEFINE_int32(num_threads,     5, "How many threads to run ?");
DEFINE_int32(epochs,       2000, "How many epochs to run for ?");
DEFINE_int32(minibatch_size, 20, "How big is the minibatch ?");
DEFINE_int32(hidden_size,    20, "How many hidden states and cells should network use?");
DEFINE_int32(vocab_size,    300, "How many characters should be modeled in the embedding.");
DEFINE_string(train,        default_paul_graham_location, "Location of the dataset");

// test file for character prediction
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::fstream;
using std::thread;

typedef float               REAL_t;
typedef LSTM<REAL_t>          lstm;
typedef Layer<REAL_t> classifier_t;
typedef Mat<REAL_t>            mat;

vector<vector<uint>> get_character_sequences(std::string filename, uint& prepad, uint& postpad, uint& vocab_size) {
    char ch;
    char linebreak = '\n';
    fstream file;
    file.open(filename);
    vector<vector<uint>> lines;
    lines.emplace_back(2);
    vector<uint>* line = &lines[0];
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
        line->push_back(std::min(vocab_size-1, (uint)ch));
    }
    return lines;
}

template<typename T>
T validation_error(
    StackedModel<T>& model,
    vector<vector<uint>>& data_set) {

    graph::NoBackprop nb;

    auto initial_state = model.initial_states();
    auto num_hidden_sizes = model.hidden_sizes.size();

    mat input_vector;
    mat logprobs;
    mat probs;

    T cost = 0.0;
    for (auto& example: data_set) {
        auto n = example.size();
        auto error = Mat<T>(1,1);
        for (int i = 0; i < n-1; ++i) {
            // pick this letter from the embedding
            input_vector  = model.embedding[example[i]];
            // pass this letter to the LSTM for processing
            initial_state = model.stacked_lstm.activate(initial_state, input_vector);
            // classifier takes as input the final hidden layer's activation:
            logprobs      = model.decode(input_vector, initial_state);
            error = error + MatOps<T>::softmax_cross_entropy_rowwise(logprobs, example[i+1]);

        }
        cost += error.w(0) / (n-1);
    }
    return cost / data_set.size();
}


template<typename T>
Mat<T> cost_fun(
    StackedModel<T>& model,
    vector<uint>& indices) {

    auto initial_state    = model.initial_states();

    mat input_vector;
    mat logprobs;
    mat probs;

    auto cost = Mat<T>(1,1);
    auto n = indices.size();

    for (int i = 0; i < n-1; ++i) {
        // pick this letter from the embedding
        input_vector  = model.embedding[indices[i]];
        // pass this letter to the LSTM for processing
        initial_state = model.stacked_lstm.activate(initial_state, input_vector);
        // classifier takes as input the final hidden layer's activation:
        logprobs      = model.decode(input_vector, initial_state);
        cost          = cost + MatOps<T>::softmax_cross_entropy_rowwise(logprobs, indices[i+1]);
    }
    return cost / (n-1);
}



int main (int argc, char *argv[]) {
        // Do not sync with stdio when using C++
        std::ios_base::sync_with_stdio(0);

        auto model = StackedModel<REAL_t>(FLAGS_vocab_size,
                                          FLAGS_input_size,
                                          FLAGS_hidden_size,
                                          FLAGS_stack_size,
                                          FLAGS_vocab_size);
        auto parameters = model.parameters();
        /*
            for (auto& param : parameters) {
                    param->npy_load(stdin);
            }
        */
        uint prepad = 0;
        uint postpad = FLAGS_vocab_size-1;
        uint vocab_size = FLAGS_vocab_size;
        auto sentences = get_character_sequences(
            FLAGS_train,
            prepad,
            postpad,
            vocab_size
        );
        int train_size = (int)(sentences.size() * 0.9);
        int valid_size = sentences.size() - train_size;
        vector<vector<uint>> train_set(sentences.begin(), sentences.begin() + train_size);
        vector<vector<uint>> valid_set(sentences.begin() + train_size, sentences.end());

        static std::random_device rd;
        static std::mt19937 seed(rd());
        static std::uniform_int_distribution<> uniform(0, train_set.size() - 1);


        // Main training loop:
        REAL_t cost = 0.0;
        vector<thread> ts;

        int total_epochs = 0;

        Solver::AdaDelta<REAL_t> solver(parameters);

        for (int t=0; t<FLAGS_num_threads; ++t) {
            ts.emplace_back([&](int thread_id) {
                auto thread_model = model.shallow_copy();
                auto thread_parameters = thread_model.parameters();
                for (auto i = 0; i < FLAGS_epochs / FLAGS_num_threads / FLAGS_minibatch_size; ++i) {
                    for (auto mb = 0; mb < FLAGS_minibatch_size; ++mb) {
                        cost_fun(
                            thread_model,            // what model should collect errors
                            train_set[uniform(seed)] // the sequence to predict
                        ).grad();
                    }
                    graph::backward(); // backpropagate

                    // solve it.
                    // RMS prop
                    //solver.step(thread_parameters, 0.01, 0.0);
                    // AdaDelta
                    solver.step(thread_parameters);
                    // SGD
                    // solver.step(thread_parameters, 0.3/minibatch_size, 0.0);
                    cost = validation_error(model, valid_set);

                    std::cout << "epoch (" << total_epochs << ") perplexity = "
                                                              << std::fixed
                              << std::setw( 5 ) // keep 7 digits
                              << std::setprecision( 3 ) // use 3 decimals
                              << std::setfill( ' ' ) << cost << std::endl;

                }
            }, t);
        }

        for(auto& t: ts)
            t.join();
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
