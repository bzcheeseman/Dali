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
DEFINE_bool(fast_dropout, false, "Use fast dropout?");
DEFINE_string(solver, "adadelta", "What solver to use (adadelta, sgd, adam)");
DEFINE_string(test, "", "Where is the test set?");
DEFINE_double(root_weight, 1.0, "By how much to weigh the roots in the objective function?");


ThreadPool* pool;

Mat<REAL_t> apply_dropout(Mat<REAL_t> X, REAL_t drop_prob) {
    if (drop_prob > 0) {
        if (FLAGS_fast_dropout) {
            return MatOps<REAL_t>::fast_dropout(X);
        } else {
            return MatOps<REAL_t>::dropout_normalized(X, drop_prob);
        }
    } else {
        return X;
    }
}

template<typename T>
class BidirectionalLSTM {
    public:

        Mat<T> embedding;
        StackedLSTM<T> stacked_lstm;
        Layer<T> decoder;

        BidirectionalLSTM(int vocabulary_size, int input_size, vector<int> hidden_sizes, int output_size, bool shortcut, bool memory_feeds_gates)
            : embedding(vocabulary_size, input_size, weights<T>::uniform(1.0 / (T) input_size)),
              stacked_lstm(input_size, hidden_sizes, shortcut, memory_feeds_gates),
              decoder(hidden_sizes.back(), output_size) {}

        BidirectionalLSTM(const BidirectionalLSTM& model, bool copy_w, bool copy_dw)
            : embedding(model.embedding, copy_w, copy_dw),
              stacked_lstm(model.stacked_lstm, copy_w, copy_dw),
              decoder(model.decoder, copy_w, copy_dw) {}

        BidirectionalLSTM<T> shallow_copy() const {
            return BidirectionalLSTM(*this, false, true);
        }

        Mat<T> activate_sequence(Indexing::Index example, T drop_prob = 0.0) {
            size_t pass = 0;

            vector<Mat<T>> X;
            for (size_t i = 0; i < example.size(); i++) {
                X.emplace_back(apply_dropout(embedding.row_pluck(example[i]), drop_prob));
            }
            auto state = stacked_lstm.cells[0].initial_states();
            for (auto& cell : stacked_lstm.cells) {
                if (pass != 0) {
                    state = cell.initial_states();
                }
                if (pass % 2 == 0) {
                    for (auto it = X.begin(); it != X.end(); ++it) {
                        state = cell.activate(
                            apply_dropout(*it, drop_prob),
                            state);
                        // prepare the observation sequence to be fed to the next
                        // level up:
                        *it = state.hidden;
                    }
                } else {
                    for (auto it = X.rbegin(); it != X.rend(); ++it) {
                        state = cell.activate(
                            apply_dropout(*it, drop_prob),
                            state);
                        // prepare the observation sequence to be fed to the next
                        // level up:
                        *it = state.hidden;
                    }
                }
                pass+=1;
            }
            return decoder.activate(state.hidden);
        }

        vector<Mat<T>> parameters() const {
            auto params = stacked_lstm.parameters();
            auto decoder_params = decoder.parameters();
            params.insert(params.end(), decoder_params.begin(), decoder_params.end());
            params.push_back(embedding);
            return params;
        }
};

REAL_t average_recall(
    BidirectionalLSTM<REAL_t>& model,
    std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>>& dataset) {
    std::cout << "Getting average_recall" << std::endl;
    ReportProgress<REAL_t> journalist("Average recall", dataset.size());
    atomic<int> seen_minibatches(0);
    atomic<int> correct(0);
    atomic<int> correct_root(0);
    atomic<int> total_root(0);
    atomic<int> total(0);
    graph::NoBackprop nb;
    for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
        pool->run([batch_id, &model, &dataset, &correct, &total, &correct_root, &total_root, &journalist, &seen_minibatches]() {
            graph::NoBackprop nb;
            auto& minibatch = dataset[batch_id];
            for (auto& example : minibatch) {
                auto prediction = model.activate_sequence(
                    std::get<0>(example), // see an example
                    0.0                   // activate without dropout
                ).argmax();               // no softmax needed, simply get best guess
                if (prediction == std::get<1>(example)) {
                    correct += 1;
                    if (std::get<2>(example)) {
                        correct_root +=1;
                    }
                }
                total += 1;
                if (std::get<2>(example)) {
                    total_root +=1;
                }
            }
            seen_minibatches += 1;
            journalist.tick(seen_minibatches, 100.0 * ((REAL_t) correct / (REAL_t) total));
        });
    }
    pool->wait_until_idle();
    journalist.done();
    std::cout << "Root nodes recall = " << 100.0 * (REAL_t) correct_root  / (REAL_t) total_root << "%" << std::endl;
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
    // Put trees into matrices:

    std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>> dataset;

    auto to_index_pair = [&word_vocab](std::pair<std::vector<std::string>, uint>&& pair, bool&& is_root) {
        return std::tuple<std::vector<uint>, uint, bool>(
            word_vocab.transform(pair.first, true),
            pair.second,
            is_root);
    };

    auto add_to_dataset_in_minibatches = [&to_index_pair](
        std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>>& dataset,
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
                    tree->to_labeled_pair(),
                    true
                )
            );

            for (auto& child : tree->general_children) {
                if (dataset[dataset.size()-1].size() == FLAGS_minibatch) {
                    dataset.emplace_back(0);
                    dataset.reserve(FLAGS_minibatch);
                }
                dataset[dataset.size()-1].emplace_back(
                    to_index_pair(
                        child->to_labeled_pair(),
                        false
                    )
                );
            }
        }
    };

    add_to_dataset_in_minibatches(dataset, sentiment_treebank);

    std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>> validation_set;

    {
        auto validation_treebank = SST::load(FLAGS_validation);
        add_to_dataset_in_minibatches(validation_set, validation_treebank);
    }


    std::cout << " Unique Trees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "           LSTM type : " << (FLAGS_memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << FLAGS_stack_size << std::endl
              << " # training examples : " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl;

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
         SST::label_names.size(),
         FLAGS_shortcut,
         FLAGS_memory_feeds_gates);

    vector<vector<Mat<REAL_t>>> thread_params;

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
    std::shared_ptr<Solver::AbstractSolver<REAL_t>> solver;
    if (FLAGS_solver == "adadelta") {
        std::cout << "Using AdaDelta" << std::endl;
        solver = make_shared<Solver::AdaDelta<REAL_t>>(params, 0.95, 1e-9, 100.0);
    } else if (FLAGS_solver == "adam") {
        std::cout << "Using Adam" << std::endl;
        solver = make_shared<Solver::Adam<REAL_t>>(params, 0.1, 0.001, 1e-9, 100.0);
    } else if (FLAGS_solver == "sgd") {
        std::cout << "Using vanilla SGD" << std::endl;
        solver = make_shared<Solver::SGD<REAL_t>>(params, 1e-9, 100.0);
    } else {
        utils::exit_with_message("Did not recognize this solver type.");
    }

    REAL_t best_validation_score = average_recall(model, validation_set);
    int epoch = 0;
    double patience = 0;

    while (patience < FLAGS_patience && epoch < epochs) {

        stringstream ss;
        ss << "Epoch " << ++epoch;
        atomic<int> batches_processed(0);

        ReportProgress<double> journalist(
            ss.str(),      // what to say first
            dataset.size() // how many steps to expect before being done with epoch
        );

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool->run([&thread_params, &thread_models,  batch_id, &journalist, &solver, &dataset, &best_validation_score, &batches_processed]() {
                auto& thread_model  = thread_models[ThreadPool::get_thread_number()];
                auto& params        = thread_params[ThreadPool::get_thread_number()];
                auto& minibatch     = dataset[batch_id];
                // many forward steps here:
                for (auto & example : minibatch) {
                    auto logprobs = thread_model.activate_sequence(std::get<0>(example), FLAGS_dropout);
                    auto error = MatOps<REAL_t>::softmax_cross_entropy(logprobs, std::get<1>(example));
                    if (std::get<2>(example) && FLAGS_root_weight != 1.0) {
                        error = error * FLAGS_root_weight;
                    }
                    error = error * (1.0 / minibatch.size());
                    error.grad();
                    graph::backward(); // backpropagate
                }
                solver->step(params); // One step of gradient descent
                journalist.tick(++batches_processed, best_validation_score);
            });
        }
        pool->wait_until_idle();
        journalist.done();
        double new_validation = average_recall(model, validation_set);

        if (new_validation < best_validation_score) {
            // lose patience:
            patience += 1;
        } else {
            // recover some patience:
            patience = std::max(patience - 1, 0.0);
            best_validation_score = new_validation;
        }
        if (best_validation_score != new_validation) {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << best_validation_score << "% ("<< new_validation << "%), patience = " << patience << std::endl;
        } else {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << best_validation_score << "%, patience = " << patience << std::endl;
        }
    }


    if (!FLAGS_test.empty()) {
        std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>> test_set;
        {
            auto test_treebank = SST::load(FLAGS_test);
            add_to_dataset_in_minibatches(test_set, test_treebank);
        }

        std::cout << "Done training" << std::endl;
        std::cout << "Test recall " << average_recall(model, test_set) << "%" << std::endl;
    }

}

