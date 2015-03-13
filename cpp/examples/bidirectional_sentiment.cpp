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

template<typename T>
class BidirectionalLSTM {
    public:

        Mat<T> embedding;
        StackedLSTM<T> stacked_lstm;
        Layer<T> decoder;

        BidirectionalLSTM(int vocabulary_size, int input_size, vector<int> hidden_sizes, int output_size)
            : embedding(vocabulary_size, input_size, -1.0 / (T) input_size, 1.0 / (T) input_size),
              stacked_lstm(input_size, hidden_sizes),
              decoder(hidden_sizes.back(), output_size) {}

        BidirectionalLSTM(const BidirectionalLSTM& model, bool copy_w, bool copy_dw)
            : embedding(model.embedding, copy_w, copy_dw),
              stacked_lstm(model.stacked_lstm, copy_w, copy_dw),
              decoder(model.decoder, copy_w, copy_dw) {}

        BidirectionalLSTM<T> shallow_copy() const {
            return BidirectionalLSTM(*this, false, true);
        }

        Mat<T> activate_sequence(Indexing::Index example, T drop_prob = 0.0) {
            int pass = 0;

            vector<Mat<T>> observation_sequence;
            for (auto& token : example) {
                observation_sequence.emplace_back(
                    MatOps<T>::dropout_normalized(
                        embedding.row_pluck(token),
                        drop_prob
                    )
                );
            }
            Mat<T> memory, hidden;
            std::tie(memory, hidden) = stacked_lstm.cells[0].initial_states();
            for (auto& cell : stacked_lstm.cells) {
                if (pass != 0) {
                    std::tie(memory, hidden) = cell.initial_states();
                }
                if (pass % 2 == 0) {
                    auto seq_begin = observation_sequence.begin();
                    for (int i = 0; i < observation_sequence.size(); ++i) {
                        std::tie(memory, hidden) = cell.activate(
                            MatOps<T>::dropout_normalized(
                                observation_sequence[i],
                                drop_prob
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
                            MatOps<T>::dropout_normalized(
                                observation_sequence[i],
                                drop_prob
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
            return thread_decoder.activate(hidden);
        }

        vector<Mat<T>> parameters() const {
            auto params = stacked_lstm.parameters();
            auto decoder_params = decoder.parameters();
            params.insert(params.end(), decoder_params.begin(), decoder_params.end());
            params.push_back(embedding);
            return params;
        }
    );
};

REAL_t average_recall(
    BidirectionalLSTM<REAL_t>& model,
    std::vector<std::vector<std::pair<std::vector<uint>, uint>>>& dataset) {
    ReportProgress<REAL_t> journalist("Average recall", dataset.size());
    atomic<int> seen_minibatches(0);
    atomic<int> correct(0);
    atomic<int> total(0);
    graph::NoBackprop nb;
    for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
        pool->run([batch_id, &model, &dataset, &correct, &total, &journalist]() {
            auto& minibatch = dataset[batch_id];
            for (auto& example : minibatch) {
                auto prediction = model.activate_sequence(example.first).argmax();
                if (prediction == example.second) {
                    correct += 1;
                }
                total += 1;
            }
            seen_minibatches += 1;
            journalist.tick(seen_minibatches, 100.0 *  ((REAL_t) correct / (REAL_t) total));
        });
    }
    return 100.0 * ((REAL_t) correct / (REAL_t) total);
}

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

    std::cout     << "        Max training epochs = " << FLAGS_epochs << std::endl;
    std::cout     << "Number of training examples = " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl;

    pool = new ThreadPool(FLAGS_j);

    /*
    Create a model with an embedding, and several stacks:
    */

    std::vector<int> hidden_sizes;
    for (int i = 0; i < (FLAGS_stack_size < 1 ? 1 : FLAGS_stack_size); i++) {
        hidden_sizes.emplace_back(FLAGS_hidden);
    }

    auto model = BidirectionalLSTM<REAL_t>(
         word_vocab.index2word.size(),
         FLAGS_input_size,
         hidden_sizes,
         SST::label_names.size());

    vector<vector<Mat<T>> thread_params;

    // what needs to be optimized:
    vector<BidirectionalLSTM<REAL_t>> thread_models;
    for (int i = 0; i < FLAGS_j; i++) {
        // create a copy for each training thread
        // (shared memory mode = Hogwild)
        thread_models.push_back(model.shallow_copy());
        thread_params.push_back(thread_models.back().parameters());
    }

    auto params = model.parameters();

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
            pool->run([&thread_params,&thread_models,  batch_id, &journalist, &solver, &dataset, &best_validation_score, &batches_processed]() {
                auto& thread_model   = thread_models[ThreadPool::get_thread_number()];
                auto& thread_parms   = thread_params[ThreadPool::get_thread_number()];
                auto& minibatch      = dataset[batch_id];
                // many forward steps here:
                for (auto & example : minibatch) {
                    auto logprobs = thread_model.activate_sequence(example.first, FLAGS_dropout);
                    auto error = MatOps<REAL_t>::softmax_cross_entropy(logprobs, example.second);
                    error.grad();
                    graph::backward(); // backpropagate
                }
                solver.step(params); // One step of gradient descent
                journalist.tick(++batches_processed, best_validation_score);
            });
        }
        pool->wait_until_idle();

        double new_validation = average_recall(model, validation_set);

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

