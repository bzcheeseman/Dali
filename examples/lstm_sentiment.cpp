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
#include "dali/data_processing/SST.h"
#include "dali/data_processing/Glove.h"
#include "dali/models/StackedModel.h"

using std::atomic;
using std::chrono::seconds;
using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
using std::string;
using std::ofstream;
using std::min;
using SST::Databatch;
using utils::Vocab;
using utils::tokenized_uint_labeled_dataset;
using utils::ConfusionMatrix;
using utils::assert2;

static const int ADADELTA_TYPE = 0;
static const int ADAGRAD_TYPE  = 1;
static const int SGD_TYPE      = 2;
static const int ADAM_TYPE     = 3;
static const int RMSPROP_TYPE  = 4;

typedef double REAL_t;
typedef Mat<REAL_t> mat;

DEFINE_int32(minibatch,           100,        "What size should be used for the minibatches ?");
DEFINE_int32(patience,            5,          "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_double(dropout,            0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(reg,                0.0,        "What penalty to place on L2 norm of weights?");
DEFINE_bool(fast_dropout,         true,       "Use fast dropout?");
DEFINE_string(solver,             "adadelta", "What solver to use (adadelta, sgd, adam)");
DEFINE_string(test,               "",         "Where is the test set?");
DEFINE_double(root_weight,        1.0,        "By how much to weigh the roots in the objective function?");
DEFINE_string(pretrained_vectors, "",         "Load pretrained word vectors?");
DEFINE_double(learning_rate,      0.01,       "Learning rate for SGD and Adagrad.");
DEFINE_string(results_file,       "",         "Where to save test performance.");
DEFINE_string(save_location,      "",         "Where to save test performance.");
DEFINE_int32(validation_metric,   0,          "Use root (1) or overall (0) objective to choose best validation parameters?");
DEFINE_double(embedding_learning_rate, -1.0,  "A separate learning rate for embedding layer");
DEFINE_bool(svd_init,             true,       "Initialize weights using SVD?");
DEFINE_bool(average_gradient,     false,      "Error during minibatch should be average or sum of errors.");

ThreadPool* pool;

int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Sentiment Analysis using single LSTM\n"
        "------------------------------------\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date April 7th 2015"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
    auto epochs              = FLAGS_epochs;
    auto sentiment_treebank  = SST::load(FLAGS_train);
    auto embedding = Mat<REAL_t>(100, 0);
    auto word_vocab = Vocab();
    if (!FLAGS_pretrained_vectors.empty() && FLAGS_load.empty()) {
        glove::load(FLAGS_pretrained_vectors, embedding, word_vocab, 50000);
    } else {
        word_vocab = SST::get_word_vocab(sentiment_treebank, FLAGS_min_occurence);
    }
    auto vocab_size          = word_vocab.index2word.size();

    // Load Dataset of Trees:
    // Put trees into matrices:
    std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>> dataset;
    auto to_index_pair = [&word_vocab](std::pair<std::vector<std::string>, uint>&& pair, bool&& is_root) {
        return std::tuple<std::vector<uint>, uint, bool>(
            word_vocab.encode(pair.first),
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
    pool = new ThreadPool(FLAGS_j);
    // Create a model with an embedding, and several stacks:

    auto stack_size  = std::max(FLAGS_stack_size, 1);

    auto model = FLAGS_load.empty() ? StackedModel<REAL_t>(
        FLAGS_pretrained_vectors.empty() ? word_vocab.index2word.size() : 0,
        FLAGS_pretrained_vectors.empty() ? FLAGS_hidden : embedding.dims(1),
        FLAGS_hidden,
        stack_size,
        SST::label_names.size(),
        (FLAGS_shortcut && stack_size > 1) ? FLAGS_shortcut : false,
        FLAGS_memory_feeds_gates) : StackedModel<REAL_t>::load(FLAGS_load);

    if (FLAGS_shortcut && stack_size == 1) {
        std::cout << "shortcut flag ignored: Shortcut connections only take effect with stack size > 1" << std::endl;
    }

    // don't send the input vector to the
    // decoder:
    model.input_vector_to_decoder(false);

    std::cout << "model.input_vector_to_decoder() = " << model.input_vector_to_decoder() << std::endl;

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

    std::cout << " Unique Trees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << "        Dropout Prob : " << FLAGS_dropout << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "   First Hidden Size : " << model.hidden_sizes[0] << std::endl
              << "           LSTM type : " << (model.memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << model.hidden_sizes.size() << std::endl
              << " # training examples : " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl
              << "     validation obj. : " << (FLAGS_validation_metric == 0 ? "overall" : "root") << std::endl
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
    vector<StackedModel<REAL_t>> thread_models;
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
    auto svd_init = weights<REAL_t>::svd(weights<REAL_t>::gaussian(0.0, 1.0));

    if (FLAGS_svd_init) {
        for (auto& param : params) {
            svd_init(param);
        }
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

    std::tuple<REAL_t, REAL_t> best_validation_score(0.0, 0.0);
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

    auto pred_fun = [&model](vector<uint>& example) {
        graph::NoBackprop nb;
        auto final_states = model.get_final_activation(
            example, 0.0
        );
        // no softmax needed, simply get best guess
        return model.decode(
            model.embedding[example.back()],
            final_states
        ).argmax();
    };

    // if no training should occur then use the validation set
    // to see how good the loaded model is.
    if (epochs == 0) {
        best_validation_score = SST::average_recall(validation_set, pred_fun, FLAGS_j);
        std::cout << "   Root recall = " << std::get<1>(best_validation_score) << std::endl;
        std::cout << "Overall recall = " << std::get<0>(best_validation_score) << std::endl;
    }

    while (patience < FLAGS_patience && epoch < epochs) {

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
                    auto final_states = thread_model.get_final_activation(std::get<0>(example), FLAGS_dropout);
                    auto logprobs = thread_model.decode(
                        thread_model.embedding[std::get<0>(example).back()],
                        final_states,
                        FLAGS_dropout
                    );
                    auto error = MatOps<REAL_t>::softmax_cross_entropy(logprobs, std::get<1>(example));
                    if (FLAGS_average_gradient) {
                        error = error/ minibatch.size();
                    }
                    if (std::get<2>(example) && FLAGS_root_weight != 1.0) {
                        error = error * FLAGS_root_weight;
                    }
                    error.grad();
                    graph::backward(); // backpropagate
                }
                // One step of gradient descent
                solver->step(params);
                // no L2 penalty on embedding:
                embedding_solver->step(embedding_params);

                // show sentiment detection as it happens:
                if (visualizer != nullptr) {
                    // show sentiment detection as system learns:
                    visualizer->throttled_feed(seconds(10), [&word_vocab, &visualizer, &minibatch, &thread_model]() {
                            // pick example
                            auto& example = std::get<0>(minibatch[utils::randint(0, minibatch.size()-1)]);
                            // visualization does not backpropagate.
                            graph::NoBackprop nb;
                            auto final_states = thread_model.get_final_activation(example, 0.0);
                            // make prediction
                            auto probs = MatOps<REAL_t>::softmax(
                                thread_model.decode(
                                    thread_model.embedding[example.back()],
                                    final_states,
                                    0.0
                                )
                            );
                            return SST::json_classification(
                                word_vocab.decode(example),
                                probs
                            );
                        }
                    );
                }
                // report minibatch completion to progress bar
                journalist.tick(++batches_processed, std::get<0>(best_validation_score));
            });
        }
        pool->wait_until_idle();
        journalist.done();
        auto new_validation = SST::average_recall(validation_set, pred_fun, FLAGS_j);
        std::cout << "Root recall=" << std::get<1>(new_validation) << std::endl;
        if (solver_type == ADAGRAD_TYPE) {
            solver->reset_caches(params);
            embedding_solver->reset_caches(embedding_params);
        }
        if (
            (FLAGS_validation_metric == 0 && std::get<0>(new_validation) + 1e-6 < std::get<0>(best_validation_score)) ||
            (FLAGS_validation_metric == 1 && std::get<1>(new_validation) + 1e-6 < std::get<1>(best_validation_score))
            ) {
            // lose patience:
            patience += 1;
        } else {
            // recover some patience:
            patience = std::max(patience - 1, 0.0);
            best_validation_score = new_validation;
        }
        if (best_validation_score != new_validation) {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << std::get<0>(best_validation_score) << "% ("<< std::get<0>(new_validation) << "%), patience = " << patience << std::endl;
        } else {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << std::get<0>(best_validation_score) << "%, patience = " << patience << std::endl;
            best_epoch = epoch;
        }
        if ((FLAGS_validation_metric == 0 && std::get<0>(new_validation) > best_score) ||
            (FLAGS_validation_metric == 1 && std::get<1>(new_validation) > best_score)) {
            best_score = (FLAGS_validation_metric == 0) ?
                std::get<0>(new_validation) :
                std::get<1>(new_validation);
            // save best:
            if (!FLAGS_save_location.empty()) {
                // stringstream ss;
                // ss << FLAGS_save_location;
                // ss << "_" << epoch;
                model.save(FLAGS_save_location);
                best_file = FLAGS_save_location;
            }
        }
    }

    if (!FLAGS_test.empty()) {
        std::vector<std::vector<std::tuple<std::vector<uint>, uint, bool>>> test_set;
        {
            auto test_treebank = SST::load(FLAGS_test);
            add_to_dataset_in_minibatches(test_set, test_treebank);
        }
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
               << "\t" << best_epoch;
            if ((FLAGS_solver == "adagrad" || FLAGS_solver == "sgd")) {
                fp << "\t" << FLAGS_learning_rate;
            } else {
                fp << "\t" << "N/A";
            }
            fp  << "\t" << FLAGS_reg << std::endl;
        }
    }
}
