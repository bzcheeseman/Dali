#include <algorithm>
#include <atomic>
#include <fstream>
#include <ostream>
#include <fstream>
#include <iterator>
#include <chrono>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/stacked_model_builder.h"
#include "dali/utils/NlpUtils.h"
#include "dali/data_processing/NER.h"
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

typedef double REAL_t;

DEFINE_int32(minibatch,           5,          "What size should be used for the minibatches ?");
DEFINE_int32(patience,            5,          "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_double(dropout,            0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(reg,                0.0,        "What penalty to place on L2 norm of weights?");
DEFINE_bool(fast_dropout,         true,       "Use fast dropout?");
DEFINE_string(test,               "",         "Where is the test set?");
DEFINE_string(pretrained_vectors, "",         "Load pretrained word vectors?");
DEFINE_string(results_file,       "",         "Where to save the accuracy results.");
DEFINE_string(save_location,      "",         "Where to save test performance.");
DEFINE_double(embedding_learning_rate, -1.0,  "A separate learning rate for embedding layer");
DEFINE_bool(svd_init,             true,       "Initialize weights using SVD?");
DEFINE_bool(average_gradient,     false,      "Error during minibatch should be average or sum of errors.");
DEFINE_string(memory_penalty_curve, "flat",   "Type of annealing used on gate memory penalty (flat, linear, square)");

ThreadPool* pool;

int main (int argc,  char* argv[]) {
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\n"
        "Named Entity Recognition using single LSTM\n"
        "------------------------------------------\n"
        "\n"
        " @author Jonathan Raiman\n"
        " @date April 7th 2015"
    );
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    int memory_penalty_curve_type;
    if (FLAGS_memory_penalty_curve        == "flat") {
        memory_penalty_curve_type = 0;
    } else if (FLAGS_memory_penalty_curve == "linear") {
        memory_penalty_curve_type = 1;
    } else if (FLAGS_memory_penalty_curve == "square") {
        memory_penalty_curve_type = 2;
    } else {
        utils::assert2(false, "memory_penalty_curve can only be flat, linear, or square.");
    }

    auto epochs = FLAGS_epochs;
    int rampup_time = 10;

    auto ner_data       = NER::load(FLAGS_train);
    auto embedding      = Mat<REAL_t>(100, 0);
    auto word_vocab     = Vocab();
    if (!FLAGS_pretrained_vectors.empty()) {
        glove::load(FLAGS_pretrained_vectors, &embedding, &word_vocab, 50000);
    } else {
        word_vocab = Vocab(NER::get_vocabulary(ner_data, FLAGS_min_occurence), true);
    }
    auto label_vocab    = Vocab(NER::get_label_vocabulary(ner_data), false);
    auto vocab_size     = word_vocab.size();
    auto dataset        = NER::convert_to_indexed_minibatches(
        word_vocab,
        label_vocab,
        ner_data,
        FLAGS_minibatch
    );
    // No validation set yet...
    decltype(dataset) validation_set;
    {
        auto ner_valid_data = NER::load(FLAGS_validation);
        validation_set = NER::convert_to_indexed_minibatches(
            word_vocab,
            label_vocab,
            ner_valid_data,
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
        label_vocab.size(),
        (FLAGS_shortcut && stack_size > 1) ? FLAGS_shortcut : false,
        FLAGS_memory_feeds_gates,
        FLAGS_memory_penalty) : StackedGatedModel<REAL_t>::load(FLAGS_load);

    if (FLAGS_shortcut && stack_size == 1)
        std::cout << "shortcut flag ignored: Shortcut connections only take effect with stack size > 1" << std::endl;
    // don't send the input vector to the
    // decoder:
    model.input_vector_to_decoder(false);
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
    if (FLAGS_embedding_learning_rate > 0)
        std::cout << " Embedding step size : " << FLAGS_embedding_learning_rate << std::endl;

    if (!FLAGS_pretrained_vectors.empty()) {
        std::cout << "  Pretrained Vectors : " << FLAGS_pretrained_vectors << std::endl;
        model.embedding = embedding;
    }


    vector<vector<Mat<REAL_t>>> thread_params;
    vector<vector<Mat<REAL_t>>> thread_embedding_params;
    // what needs to be optimized:
    vector<StackedGatedModel<REAL_t>> thread_models;
    std::tie(thread_models, thread_embedding_params, thread_params) = utils::shallow_copy_multi_params(model, FLAGS_j, [&model](const Mat<REAL_t>& mat) {
        return &mat.w().memory() == &model.embedding.w().memory();
    });

    vector<Mat<REAL_t>> params = model.parameters();
    vector<Mat<REAL_t>> embedding_params(params.begin(), params.begin() + 1);
    params = vector<Mat<REAL_t>>(params.begin() + 1, params.end());
    auto solver           = Solver::construct(FLAGS_solver, params, (REAL_t) FLAGS_learning_rate, (REAL_t) FLAGS_reg);
    auto embedding_solver = Solver::construct(FLAGS_solver,
        embedding_params,
        (REAL_t) (FLAGS_embedding_learning_rate > 0 ? FLAGS_embedding_learning_rate : FLAGS_learning_rate),
        (REAL_t) FLAGS_reg);

    REAL_t best_validation_score = 0.0;
    int epoch = 0, best_epoch = 0;
    double patience = 0;
    string best_file = "";
    REAL_t best_score = 0.0;

    shared_ptr<Visualizer> visualizer;
    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer);

    auto pred_fun = [&model](vector<uint>& example) {
        graph::NoBackprop nb;
        vector<uint> predictions(example.size());
        auto state = model.initial_states();
        Mat<REAL_t> memory;
        Mat<REAL_t> probs;
        int ex_idx = 0;
        for (auto& el : example) {
            std::tie(state, probs, memory) = model.activate(state, el);
            predictions[ex_idx++] = probs.argmax();
        }
        return predictions;
    };

    // if no training should occur then use the validation set
    // to see how good the loaded model is.
    if (epochs == 0) {
        best_validation_score = NER::average_recall(validation_set, pred_fun, FLAGS_j);
        std::cout << "recall = " << best_validation_score << std::endl;
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
            pool->run([&word_vocab, &label_vocab, &visualizer, &thread_embedding_params, &thread_params, &thread_models, batch_id, &journalist, &solver, &embedding_solver, &dataset, &best_validation_score, &batches_processed]() {
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

                        error = error + MatOps<REAL_t>::cross_entropy_rowwise(
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
                    visualizer->throttled_feed(seconds(5), [&word_vocab, &label_vocab, &visualizer, &minibatch, &thread_model]() {
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

                        auto input_sentence = make_shared<visualizable::Sentence<REAL_t>>(word_vocab.decode(&example));
                        input_sentence->set_weights(MatOps<REAL_t>::hstack(memories));
                        auto decoded = label_vocab.decode(&prediction);
                        for (auto it_decoded = decoded.begin(); it_decoded < decoded.end(); it_decoded++) {
                            if (*it_decoded == label_vocab.index2word[0]) {
                                *it_decoded = " ";
                            }
                        }

                        auto psentence = visualizable::ParallelSentence<REAL_t>(
                            input_sentence,
                            make_shared<visualizable::Sentence<REAL_t>>(decoded)
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
        auto new_validation = NER::average_recall(validation_set, pred_fun, FLAGS_j);
        if (solver->method == Solver::METHOD_ADAGRAD) {
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
}
