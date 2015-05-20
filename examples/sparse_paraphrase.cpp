#include <algorithm>
#include <atomic>
#include <Eigen/Eigen>
#include <fstream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <chrono>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/NlpUtils.h"
#include "dali/data_processing/Paraphrase.h"
#include "dali/data_processing/Glove.h"
#include "dali/models/StackedGatedModel.h"
#include "dali/visualizer/visualizer.h"

using std::atomic;
using std::chrono::seconds;
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::stringstream;
using std::string;
using std::ofstream;
using std::min;
using utils::Vocab;
using utils::assert2;

static const int ADADELTA_TYPE = 0;
static const int ADAGRAD_TYPE  = 1;
static const int SGD_TYPE      = 2;
static const int ADAM_TYPE     = 3;
static const int RMSPROP_TYPE  = 4;

typedef double REAL_t;
typedef Mat<REAL_t> mat;

DEFINE_int32(minibatch,           5,          "What size should be used for the minibatches ?");
DEFINE_int32(patience,            5,          "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_double(dropout,            0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(reg,                0.0,        "What penalty to place on L2 norm of weights?");
DEFINE_bool(fast_dropout,         true,       "Use fast dropout?");
DEFINE_string(solver,             "adadelta", "What solver to use (adadelta, sgd, adam)");
DEFINE_string(test,               "",         "Where is the test set?");
DEFINE_string(pretrained_vectors, "",         "Load pretrained word vectors?");
DEFINE_double(learning_rate,      0.01,       "Learning rate for SGD and Adagrad.");
DEFINE_string(results_file,       "",         "Where to save test performance.");
DEFINE_string(save_location,      "",         "Where to save test performance.");
DEFINE_double(embedding_learning_rate, -1.0,  "A separate learning rate for embedding layer");
DEFINE_bool(svd_init,             true,       "Initialize weights using SVD?");
DEFINE_bool(average_gradient,     false,      "Error during minibatch should be average or sum of errors.");
DEFINE_string(memory_penalty_curve, "flat",   "Type of annealing used on gate memory penalty (flat, linear, square)");

ThreadPool* pool;

int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Textual Similarity using single LSTM\n"
        "------------------------------------\n"
        "\n"
        " @author Jonathan Raiman & Szymon Sidor\n"
        " @date May 20th 2015"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    int memory_penalty_curve_type;
    if (FLAGS_memory_penalty_curve == "flat") {
        memory_penalty_curve_type = 0;
    } else if (FLAGS_memory_penalty_curve == "linear") {
        memory_penalty_curve_type = 1;
    } else if (FLAGS_memory_penalty_curve == "square") {
        memory_penalty_curve_type = 2;
    } else {
        utils::assert2(false, "memory_penalty_curve can only be flat, linear, or square.");
    }

    auto epochs = FLAGS_epochs;
    int rampup_time = 10;

    auto paraphrase_data = paraphrase::load(FLAGS_train);
    auto embedding       = Mat<REAL_t>(100, 0);
    auto word_vocab      = Vocab();
    if (!FLAGS_pretrained_vectors.empty()) {
        glove::load(FLAGS_pretrained_vectors, embedding, word_vocab, 50000);
    } else {
        word_vocab = Vocab(paraphrase::get_vocabulary(paraphrase_data, FLAGS_min_occurence), true);
    }
    auto vocab_size     = word_vocab.size();
    auto dataset        = paraphrase::convert_to_indexed_minibatches(
        word_vocab,
        paraphrase_data,
        FLAGS_minibatch
    );
    // No validation set yet...
    decltype(dataset) validation_set;
    {
        auto paraphrase_valid_data = paraphrase::load(FLAGS_validation);
        validation_set = paraphrase::convert_to_indexed_minibatches(
            word_vocab,
            paraphrase_valid_data,
            FLAGS_minibatch
        );
    }

    pool = new ThreadPool(FLAGS_j);
    // Create a model with an embedding, and several stacks:

    auto stack_size  = std::max(FLAGS_stack_size, 1);

    auto model = FLAGS_load.empty() ? StackedGatedModel<REAL_t>(
        FLAGS_pretrained_vectors.empty() ? word_vocab.size() : 0,
        FLAGS_pretrained_vectors.empty() ? FLAGS_hidden : embedding.dims(1),
        FLAGS_hidden,
        stack_size,
        1,
        (FLAGS_shortcut && stack_size > 1) ? FLAGS_shortcut : false,
        FLAGS_memory_feeds_gates,
        FLAGS_memory_penalty) : StackedGatedModel<REAL_t>::load(FLAGS_load);

    if (FLAGS_shortcut && stack_size == 1) {
        std::cout << "shortcut flag ignored: Shortcut connections only take effect with stack size > 1" << std::endl;
    }

    // don't send the input vector to the
    // decoder:
    model.input_vector_to_decoder(false);

    int solver_type;
    if (FLAGS_solver == "adadelta") {
        solver_type = ADADELTA_TYPE;
    } else if (FLAGS_solver == "adam") {
        solver_type = ADAM_TYPE;
    } else if (FLAGS_solver == "sgd") {
        solver_type = SGD_TYPE;
    } else if (FLAGS_solver == "adagrad") {
        solver_type = ADAGRAD_TYPE;
    } else {
        utils::exit_with_message("Did not recognize this solver type.");
    }
    if (dataset.size() == 0) utils::exit_with_message("Dataset is empty");

    std::cout << "     Vocabulary size : " << vocab_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << "        Dropout Prob : " << FLAGS_dropout << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "   First Hidden Size : " << model.hidden_sizes[0] << std::endl
              << "           LSTM type : " << (model.memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << model.hidden_sizes.size() << std::endl
              << " # training examples : " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl
              << " # layers -> decoder : " << model.decoder.matrices.size() << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;
    if (FLAGS_embedding_learning_rate > 0) {
        std::cout << " Embedding step size : " << FLAGS_embedding_learning_rate << std::endl;
    }

    if (!FLAGS_pretrained_vectors.empty()) {
        std::cout << "  Pretrained Vectors : " << FLAGS_pretrained_vectors << std::endl;
        model.embedding = embedding;
    }

    vector<vector<Mat<REAL_t>>> thread_params;
    vector<vector<Mat<REAL_t>>> thread_embedding_params;

    // what needs to be optimized:
    vector<StackedGatedModel<REAL_t>> thread_models;
    for (int i = 0; i < FLAGS_j; i++) {
        // create a copy for each training thread
        // (shared memory mode = Hogwild)
        thread_models.push_back(model.shallow_copy());

        auto thread_model_params = thread_models.back().parameters();
        // take a slice of all the parameters except for embedding.
        thread_params.emplace_back(
            thread_model_params.begin() + 1,
            thread_model_params.end()
        );
        // then add the embedding (the first parameter of StackeModel)
        thread_embedding_params.emplace_back(
            thread_model_params.begin(),
            thread_model_params.begin() + 1
        );
    }
    vector<Mat<REAL_t>> params;
    vector<Mat<REAL_t>> embedding_params;
    {
        auto temp  = model.parameters();
        params = vector<Mat<REAL_t>>(temp.begin()+1, temp.end());
        embedding_params.push_back(model.parameters()[0]);
    }

    // Rho value, eps value, and gradient clipping value:
    std::shared_ptr<Solver::AbstractSolver<REAL_t>> solver;
    std::shared_ptr<Solver::AbstractSolver<REAL_t>> embedding_solver;
    switch (solver_type) {
        case ADADELTA_TYPE:
            solver = make_shared<Solver::AdaDelta<REAL_t>>(params, 0.95, 1e-9, 100.0, (REAL_t) FLAGS_reg);
            embedding_solver = make_shared<Solver::AdaDelta<REAL_t>>(embedding_params, 0.95, 1e-9, 100.0, 0.0);
            break;
        case ADAM_TYPE:
            solver = make_shared<Solver::Adam<REAL_t>>(params, 0.1, 0.001, 1e-9, 100.0, (REAL_t) FLAGS_reg);
            embedding_solver = make_shared<Solver::Adam<REAL_t>>(embedding_params, 0.1, 0.001, 1e-9, 100.0, 0.0);
            break;
        case SGD_TYPE:
            solver = make_shared<Solver::SGD<REAL_t>>(params, 100.0, (REAL_t) FLAGS_reg);
            embedding_solver = make_shared<Solver::SGD<REAL_t>>(embedding_params, 100.0, 0.0);
            dynamic_cast<Solver::SGD<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
            dynamic_cast<Solver::SGD<REAL_t>*>(embedding_solver.get())->step_size = (
                FLAGS_embedding_learning_rate > 0 ? FLAGS_embedding_learning_rate : FLAGS_learning_rate);
            break;
        case ADAGRAD_TYPE:
            solver = make_shared<Solver::AdaGrad<REAL_t>>(params, 1e-9, 100.0, (REAL_t) FLAGS_reg);
            embedding_solver = make_shared<Solver::AdaGrad<REAL_t>>(embedding_params, 1e-9, 100.0, 0.0);
            dynamic_cast<Solver::AdaGrad<REAL_t>*>(solver.get())->step_size = FLAGS_learning_rate;
            dynamic_cast<Solver::AdaGrad<REAL_t>*>(embedding_solver.get())->step_size = (
                FLAGS_embedding_learning_rate > 0 ? FLAGS_embedding_learning_rate : FLAGS_learning_rate);
            break;
        default:
            utils::exit_with_message("Did not recognize this solver type.");
    }

    REAL_t best_validation_score = 0.0;
    int epoch = 0;
    int best_epoch = 0;
    double patience = 0;
    string best_file = "";
    REAL_t best_score = 0.0;

    shared_ptr<Visualizer> visualizer;
    if (!FLAGS_visualizer.empty()) {
        try {
            visualizer = make_shared<Visualizer>(FLAGS_visualizer, true);
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl; // could not connect to redis.
        }
    }

    auto pred_fun = [&model](vector<uint>& sentence1, vector<uint>& sentence2) {
        // Write me
        return 2.0;
    };

    // if no training should occur then use the validation set
    // to see how good the loaded model is.
    if (epochs == 0) {
        best_validation_score = paraphrase::pearson_correlation(validation_set, pred_fun, FLAGS_j);
        std::cout << "correlation = " << best_validation_score << std::endl;
    }

    while (patience < FLAGS_patience && epoch < epochs) {

        if (memory_penalty_curve_type == 1) { // linear
            model.memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
            );
            for (auto& thread_model : thread_models) {
                thread_model.memory_penalty = std::min(
                    FLAGS_memory_penalty,
                    (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
                );
            }
        } else if (memory_penalty_curve_type == 2) { // square
            model.memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
            );
            for (auto& thread_model : thread_models) {
                thread_model.memory_penalty = std::min(
                    FLAGS_memory_penalty,
                    (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
                );
            }
        }

        stringstream ss;
        ss << "Epoch " << ++epoch;
        atomic<int> batches_processed(0);

        ReportProgress<double> journalist(
            ss.str(),      // what to say first
            dataset.size() // how many steps to expect before being done with epoch
        );

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool->run([&word_vocab, &visualizer, &solver_type, &thread_embedding_params, &thread_params, &thread_models, batch_id, &journalist, &solver, &embedding_solver, &dataset, &best_validation_score, &batches_processed]() {
                auto& thread_model     = thread_models[ThreadPool::get_thread_number()];
                auto& params           = thread_params[ThreadPool::get_thread_number()];
                auto& embedding_params = thread_embedding_params[ThreadPool::get_thread_number()];
                auto& minibatch        = dataset[batch_id];
                // many forward steps here:
                for (auto & example : minibatch) {
                    auto error = MatOps<REAL_t>::consider_constant(Mat<REAL_t>(1,1));
                    auto state = thread_model.initial_states();
                    Mat<REAL_t> memory;
                    Mat<REAL_t> probs;
                    vector<Mat<REAL_t>> memories;
                    memories.reserve(std::get<0>(example).size());

                    for (int ex_idx = 0; ex_idx < std::get<0>(example).size(); ex_idx++) {
                        std::tie(state, probs, memory) = thread_model.activate(state, std::get<0>(example)[ex_idx]);
                        memories.emplace_back(memory);
                        error = error + MatOps<REAL_t>::cross_entropy(
                            probs,
                            std::get<1>(example)[ex_idx]
                        );
                    }
                    // total error is prediction error + memory usage.
                    if (thread_model.memory_penalty > 0) {
                        error = error + MatOps<REAL_t>::add(memories) * thread_model.memory_penalty;
                    }
                    error.grad();
                    graph::backward(); // backpropagate
                }
                // One step of gradient descent
                solver->step(params);
                // no L2 penalty on embedding:
                embedding_solver->step(embedding_params);
                if (visualizer != nullptr) {
                    visualizer->throttled_feed(seconds(5), [&word_vocab, &visualizer, &minibatch, &thread_model]() {
                        // pick example
                        auto& example = std::get<0>(minibatch[utils::randint(0, minibatch.size()-1)]);
                        // visualization does not backpropagate.
                        graph::NoBackprop nb;

                        auto state = thread_model.initial_states();
                        Mat<REAL_t> memory;
                        Mat<REAL_t> probs;

                        vector<Mat<REAL_t>> memories;
                        vector<uint> prediction;

                        for (auto& el : example) {
                            std::tie(state, probs, memory) = thread_model.activate(state, el);
                            memories.emplace_back(memory);
                            prediction.emplace_back(probs.argmax());
                        }

                        auto input_sentence = make_shared<visualizable::Sentence<REAL_t>>(word_vocab.decode(example));
                        input_sentence->set_weights(MatOps<REAL_t>::hstack(memories));

                        // Write me

                        auto psentence = visualizable::ParallelSentence<REAL_t>(
                            input_sentence,
                            input_sentence
                        );
                        return psentence.to_json();
                    });
                }
                // report minibatch completion to progress bar
                journalist.tick(++batches_processed, best_validation_score);
            });
        }
        pool->wait_until_idle();
        journalist.done();
        auto new_validation = paraphrase::pearson_correlation(validation_set, pred_fun, FLAGS_j);
        if (solver_type == ADAGRAD_TYPE) {
            solver->reset_caches(params);
            embedding_solver->reset_caches(embedding_params);
        }
        if (new_validation + 1e-6 < best_validation_score) {
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
            best_epoch = epoch;
        }
        if (new_validation > best_score) {
            best_score = new_validation;
            // save best:
            if (!FLAGS_save_location.empty()) {
                model.save(FLAGS_save_location);
                best_file = FLAGS_save_location;
            }
        }
    }

    /*if (!FLAGS_test.empty()) {
        auto test_set = SST::convert_trees_to_indexed_minibatches(
            word_vocab,
            SST::load(FLAGS_test),
            FLAGS_minibatch
        );
        if (!FLAGS_save_location.empty() && !best_file.empty()) {
            std::cout << "loading from best validation parameters \"" << best_file << "\"" << std::endl;
            auto params = model.parameters();
            utils::load_matrices(params, best_file);
        }
        auto recall = SST::average_recall(test_set, pred_fun, FLAGS_j);

        std::cout << "Done training" << std::endl;
        std::cout << "Test recall "  << std::get<0>(recall) << "%, root => " << std::get<1>(recall)<< "%" << std::endl;
        if (!FLAGS_results_file.empty()) {
            ofstream fp;
            fp.open(FLAGS_results_file.c_str(), std::ios::out | std::ios::app);
            fp         << FLAGS_solver
               << "\t" << FLAGS_minibatch
               << "\t" << (FLAGS_fast_dropout ? "fast" : "std")
               << "\t" << FLAGS_dropout
               << "\t" << FLAGS_hidden
               << "\t" << std::get<0>(recall)
               << "\t" << std::get<1>(recall)
               << "\t" << best_epoch
               << "\t" << FLAGS_memory_penalty
               << "\t" << FLAGS_memory_penalty_curve;
            if ((FLAGS_solver == "adagrad" || FLAGS_solver == "sgd")) {
                fp << "\t" << FLAGS_learning_rate;
            } else {
                fp << "\t" << "N/A";
            }
            fp  << "\t" << FLAGS_reg << std::endl;
        }
    }*/
}
