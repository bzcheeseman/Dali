#include <algorithm>
#include <atomic>
#include <Eigen/Eigen>
#include <fstream>
#include <iterator>
#include <chrono>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/data_processing/SST.h"
#include "dali/models/StackedModel.h"
#include "dali/models/StackedGatedModel.h"

using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
using std::string;
using std::min;
using utils::Vocab;
using utils::tokenized_uint_labeled_dataset;
using std::atomic;
using std::chrono::seconds;
using SST::Databatch;
using utils::ConfusionMatrix;

typedef float REAL_t;
typedef Mat<REAL_t> mat;

DEFINE_int32(minibatch, 100, "What size should be used for the minibatches ?");
DEFINE_int32(patience, 5,    "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_double(dropout, 0.3, "How much dropout noise to add to the problem ?");

ThreadPool* pool;

int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Sentiment Analysis using multiple bidirectional LSTMs\n"
        "-----------------------------------------------------\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date March 13th 2015"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    auto epochs              = FLAGS_epochs;
    auto sentiment_treebank  = SST::load(FLAGS_train);

    auto word_vocab          = SST::get_word_vocab(sentiment_treebank, FLAGS_min_occurence);
    auto vocab_size          = word_vocab.index2word.size();

    // Load Dataset of Trees:
    std::cout << "Unique Treees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl;

    // Put trees into matrices:
    #ifdef USE_MASKED_LOSS
        vector<Databatch> dataset = SST::Databatch::create_labeled_dataset(
            sentiment_treebank,
            word_vocab,
            (size_t)FLAGS_minibatch);
        vector<Databatch> validation_set;

        {
            auto validation_treebank = SST::load(FLAGS_validation);
            validation_set = SST::Databatch::create_labeled_dataset(
                validation_treebank,
                word_vocab,
                (size_t)FLAGS_minibatch);
        }
    #else
        std::vector<std::vector<std::pair<std::vector<uint>, uint>>> dataset;

        auto to_index_pair = [&word_vocab](std::pair<std::vector<std::string>, uint>&& pair) {
            return std::pair<std::vector<uint>, uint>(
                word_vocab.transform(pair.first, true),
                pair.second);
        };

        auto add_to_dataset_in_minibatches = [&to_index_pair](
            std::vector<std::vector<std::pair<std::vector<uint>, uint>>>& dataset,
            std::vector<SST::AnnotatedParseTree::shared_tree>& trees
            ) {
            if (dataset.size() == 0)
                dataset.emplace_back(0);
            for (auto& tree : trees) {
                if (dataset[dataset.size()-1].size() == FLAGS_minibatch) {
                    dataset.emplace_back(0);
                    dataset.reserve(FLAGS_minibatch);
                }
                dataset[dataset.size()-1].emplace_back(
                    to_index_pair(
                        tree->to_labeled_pair()
                    )
                );

                for (auto& child : tree->general_children) {
                    if (dataset[dataset.size()-1].size() == FLAGS_minibatch) {
                        dataset.emplace_back(0);
                        dataset.reserve(FLAGS_minibatch);
                    }
                    dataset[dataset.size()-1].emplace_back(
                        to_index_pair(
                            child->to_labeled_pair()
                        )
                    );
                }
            }
        };

        add_to_dataset_in_minibatches(dataset, sentiment_treebank);

        std::vector<std::vector<std::pair<std::vector<uint>, uint>>> validation_set;

        {
            auto validation_treebank = SST::load(FLAGS_validation);
            add_to_dataset_in_minibatches(validation_set, validation_treebank);
        }
    #endif

    std::cout     << "        Max training epochs = " << FLAGS_epochs << std::endl;
    #ifdef USE_MASKED_LOSS
        std::cout     << "Number of training examples = " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].data->rows()) << std::endl;
    #else
        std::cout     << "Number of training examples = " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl;
    #endif

    pool = new ThreadPool(FLAGS_j);

    /*
    Create a model with an embedding, and several stacks:
    */

    Mat<REAL_t> embedding(
        word_vocab.index2word.size(),
        FLAGS_input_size,
        -1.0 / (REAL_t) FLAGS_input_size,
         1.0 / (REAL_t) FLAGS_input_size);
    std::vector<int> hidden_sizes;
    for (int i = 0; i < (FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size); i++) {
        hidden_sizes.emplace_back(FLAGS_hidden);
    }
    StackedLSTM<REAL_t> model(FLAGS_input_size, hidden_sizes);
    Layer<REAL_t> decoder(
        FLAGS_hidden,
        SST::label_names.size()
    );

    // what needs to be optimized:
    auto params = model.parameters();
    auto decoder_params = decoder.parameters();
    params.insert(params.end(), decoder_params.begin(), decoder_params.end());
    params.push_back(embedding);

    vector<Mat<REAL_t>> thread_embeddings;
    vector<StackedLSTM<REAL_t>> thread_models;
    vector<Layer<REAL_t>> thread_decoders;
    for (int i = 0; i < FLAGS_j; i++) {
        // create a copy for each training thread
        // (shared memory mode = Hogwild)
        thread_embeddings.push_back(embedding.shallow_copy());
        thread_models.push_back(model.shallow_copy());
        thread_decoders.push_back(decoder.shallow_copy());
    }

    // Rho value, eps value, and gradient clipping value:
    Solver::AdaDelta<REAL_t> solver(params, 0.95, 1e-9, 5.0);

    REAL_t best_validation_score = 0.0;
    int epoch = 0;
    double patience = 0;

    while (patience < FLAGS_patience && epoch < epochs) {

        stringstream ss;
        ss << "Epoch " << ++epoch;
        atomic<int> batches_processed(0);

        ReportProgress<double> journalist(
            ss.str(), // what to say first
            dataset.size() // how many steps to expect before being done with epoch
        );

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool->run([&params, batch_id, &thread_embeddings, &thread_decoders, &thread_models, &journalist, &solver, &dataset, &best_validation_score, &batches_processed]() {
                auto& embedding      = thread_embeddings[ThreadPool::get_thread_number()];
                auto& thread_model   = thread_models[ThreadPool::get_thread_number()];
                auto& thread_decoder = thread_decoders[ThreadPool::get_thread_number()];
                auto& minibatch      = dataset[batch_id];

                // many forward steps here:
                #ifdef USE_MASKED_LOSS
                    /*thread_model.masked_predict_cost(
                        minibatch.data, // the sequence to draw from
                        minibatch.data, // what to predict (the words offset by 1)
                        1,
                        minibatch.codelens,
                        0,
                        (REAL_t) FLAGS_dropout
                    );*/
                #else
                    for (auto & example : minibatch) {
                        int pass = 0;

                        vector<Mat<REAL_t>> observation_sequence;
                        for (auto& token : example.first) {
                            observation_sequence.emplace_back(
                                MatOps<REAL_t>::dropout_normalized(
                                    embedding.row_pluck(token),
                                    FLAGS_dropout
                                )
                            );
                        }
                        Mat<REAL_t> memory, hidden;
                        std::tie(memory, hidden) = thread_model.cells[0].initial_states();
                        for (auto& cell : thread_model.cells) {
                            if (pass != 0) {
                                std::tie(memory, hidden) = cell.initial_states();
                            }
                            if (pass % 2 == 0) {
                                auto seq_begin = observation_sequence.begin();
                                for (int i = 0; i < observation_sequence.size(); ++i) {
                                    std::tie(memory, hidden) = cell.activate(
                                        MatOps<REAL_t>::dropout_normalized(
                                            observation_sequence[i],
                                            FLAGS_dropout
                                        ),
                                        memory,
                                        hidden);
                                    // prepare the observation sequence to be fed to the next
                                    // level up:
                                    observation_sequence[i] = hidden;
                                }
                            } else {
                                auto seq_begin = observation_sequence.rbegin();
                                for (int i = observation_sequence.size() - 1; i > -1; --i) {
                                    std::tie(memory, hidden) = cell.activate(
                                        MatOps<REAL_t>::dropout_normalized(
                                            observation_sequence[i],
                                            FLAGS_dropout
                                        ),
                                        memory,
                                        hidden);
                                    // prepare the observation sequence to be fed to the next
                                    // level up:
                                    observation_sequence[i] = hidden;
                                }
                            }
                            pass+=1;
                        }
                        auto logprobs = thread_decoder.activate(hidden);
                        auto error = MatOps<REAL_t>::softmax_cross_entropy(logprobs, example.second);
                        error.grad();
                        graph::backward(); // backpropagate
                    }
                #endif
                solver.step(params); // One step of gradient descent

                journalist.tick(++batches_processed, best_validation_score);
            });
        }
        pool->wait_until_idle();

        double new_validation = 100.0;// * average_error(model, validation_set);

        if (new_validation < best_validation_score) {
            // lose patience:
            patience += 1;
        } else {
            // recover some patience:
            patience = std::max(patience - 1, 0.0);
            best_validation_score = new_validation;
        }
    }

}

