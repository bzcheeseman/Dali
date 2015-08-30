#include <algorithm>
#include <atomic>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <chrono>
#include <fstream>
#include <ostream>

#include "dali/core.h"
#include "dali/utils.h"
#include "dali/utils/stacked_model_builder.h"
#include "dali/utils/NlpUtils.h"
#include "dali/data_processing/SST.h"
#include "dali/data_processing/Glove.h"
#include "dali/models/StackedModel.h"
#include "dali/models/StackedGatedModel.h"
#include "dali/visualizer/visualizer.h"
#include "dali/tensor/__MatMacros__.h"

using std::vector;
using std::make_shared;
using std::shared_ptr;
using std::ifstream;
using std::istringstream;
using std::stringstream;
using std::string;
using std::ofstream;
using std::min;
using utils::Vocab;
using utils::tokenized_uint_labeled_dataset;
using std::atomic;
using std::chrono::seconds;
using SST::SentimentBatch;
using utils::assert2;
using json11::Json;
using std::to_string;

typedef float REAL_t;
DEFINE_string(results_file,            "",         "Where to save test performance.");
DEFINE_string(save_location,           "",         "Where to save test performance.");
DEFINE_int32(minibatch,      100,        "What size should be used for the minibatches ?");
DEFINE_int32(patience,       5,          "How many unimproving epochs to wait through before witnessing progress ?");
DEFINE_double(dropout,       0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(reg,           0.0,        "What penalty to place on L2 norm of weights?");
DEFINE_bool(fast_dropout,    false,       "Use fast dropout?");
DEFINE_bool(bidirectional,   true,       "Read forward and backwards ?");
DEFINE_string(test,          "",         "Where is the test set?");
DEFINE_double(root_weight,   1.0,        "By how much to weigh the roots in the objective function?");
DEFINE_bool(recursive_gates, true,       "Make a prediction at every timestep?");
DEFINE_bool(surprise,        false,      "Use Surprise distance with target distribution?");
DEFINE_bool(convolution,     false,      "Perform a convolution before passing to LSTMs ?");
DEFINE_int32(filters,        50,         "Number of filters to use for Convolution");
DEFINE_string(pretrained_vectors, "",    "Load pretrained word vectors?");
DEFINE_string(memory_penalty_curve, "flat",   "Type of annealing used on gate memory penalty (flat, linear, square)");
DEFINE_int32(validation_metric,   0,          "Use root (1) or overall (0) objective to choose best validation parameters?");

const REAL_t EPS = 1e-6;

// // Error in this computation
// template<typename T>
// Mat<T> softmax_categorical_surprise(Mat<T> logprobs, int target) {
//     auto out = Mat<T>(1, 1, false);
//     auto probs = MatOps<T>::softmax_no_grad_colwise(logprobs);

//     out.w(0) = -(
//         std::log1p(-std::sqrt(1.0 - probs.w(target, 0))) -
//         std::log1p( std::sqrt(1.0 - probs.w(target, 0)))
//     );

//     if (graph::backprop_enabled()) {
//         if (!logprobs.constant) {
//             graph::emplace_back([logprobs, probs, target]() {
//                 auto root_coeff = std::sqrt(std::max(MatOps<T>::EPS, (T) (1.0 - probs.w(target, 0))));
//                 GET_GRAD(logprobs).noalias() += (
//                     (
//                         (0.5 * probs.w(target, 0) / (root_coeff * (root_coeff + 1.0)) + MatOps<T>::EPS) +
//                         (0.5 * probs.w(target, 0) / (root_coeff * (1.0 - root_coeff)) + MatOps<T>::EPS)

//                     ) * GET_MAT(probs)
//                 );
//                 GET_GRAD(logprobs)(target, 0) -= probs.w(target, 0) * (
//                     (0.5 / (root_coeff * (root_coeff + 1.0)) + MatOps<T>::EPS) +
//                     (0.5 / (root_coeff * (1.0 - root_coeff)) + MatOps<T>::EPS)
//                 );
//             });
//         }
//     }
//     return out;
// }

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

vector<int> repeat_val(int val, int times) {
    vector<int> vals(times);
    for (auto& t : vals) {
        t = val;
    }
    return vals;
}

// For convolutions:
template<typename T>
class Filters {
    public:
        std::vector<Mat<T>> filters;
        Filters(int n, int width, int height){
            for (int i = 0; i < n; i++) {
                filters.emplace_back(
                    width,
                    height,
                    weights<T>::uniform(1.0 / (height + width))
                );
            }
        }
        Filters(const Filters& model, bool copy_w, bool copy_dw) {
            for (auto & f : model.filters) {
                filters.emplace_back(f, copy_w, copy_dw);
            }
        }
        vector<Mat<T>> parameters() const {
            return filters;
        }

        Filters<T> shallow_copy() const {
            return Filters(*this, false, true);
        }
};

template<typename T>
class BidirectionalLSTM {
    public:
        Mat<T> embedding;
        StackedLSTM<T> stacked_lstm;
        StackedInputLayer<T> decoder;
        bool use_recursive_gates;
        int output_size;
        int hidden_size;
        int stack_size;
        bool convolution;
        bool bidirectional;
        T memory_penalty;
        std::shared_ptr<StackedInputLayer<T>> prediction_gate;
        std::shared_ptr<Filters<T>> filters;
        typedef std::map<std::string, std::vector<std::string>> config_t;

        BidirectionalLSTM(int vocabulary_size, int input_size, int hidden_size, int stack_size, int output_size, bool shortcut, bool memory_feeds_gates, bool use_recursive_gates, bool convolution, bool bidirectional)
            :
              bidirectional(bidirectional),
              convolution(convolution),
              embedding(vocabulary_size, input_size, weights<T>::uniform(-0.05, 0.05)),
              stacked_lstm(hidden_size, repeat_val(hidden_size, std::max(stack_size-1, 1)), shortcut, memory_feeds_gates),
            decoder({hidden_size, hidden_size}, output_size),
            use_recursive_gates(use_recursive_gates),
            stack_size(stack_size),
            hidden_size(hidden_size),
            output_size(output_size) {

            if (use_recursive_gates) {
                prediction_gate = std::make_shared< StackedInputLayer<T> >( std::vector<int>({hidden_size, hidden_size, output_size}), 1);
            } else {
                prediction_gate = nullptr;
            }

            if (convolution) {
                filters = make_shared<Filters<T>>(
                    hidden_size, // number of filters
                    input_size, // filter height
                    2); // filter width (temporal dimension)
            } else {
                if (input_size != hidden_size) {
                    throw std::runtime_error(
                        "Input size must equal hidden size "
                        "when no convolution is applied to change"
                        " the word dimensions.");
                }
            }

            if (stack_size > 1) {
                stacked_lstm.cells.insert(
                    stacked_lstm.cells.begin(),
                    LSTM<T>(hidden_size, hidden_size, memory_feeds_gates)
                );
            }
        }

        config_t configuration() const  {
            config_t config;
            config["output_size"].emplace_back(to_string(output_size));
            config["vocabulary_size"].emplace_back(to_string(embedding.dims(0)));
            config["input_size"].emplace_back(to_string(embedding.dims(1)));
            config["use_recursive_gates"].emplace_back(use_recursive_gates ? "true" : "false");
            config["convolution"].emplace_back(convolution ? "true" : "false");
            config["bidirectional"].emplace_back(bidirectional ? "true" : "false");
            for (auto& cell : stacked_lstm.cells)
                config["hidden_sizes"].emplace_back(to_string(cell.hidden_size));
            return config;
        }

        BidirectionalLSTM(const BidirectionalLSTM& model, bool copy_w, bool copy_dw)
            :
              embedding(model.embedding, copy_w, copy_dw),
              bidirectional(model.bidirectional),
              convolution(model.convolution),
              stacked_lstm(model.stacked_lstm, copy_w, copy_dw),
              decoder(model.decoder, copy_w, copy_dw),
              use_recursive_gates(model.use_recursive_gates),
              hidden_size(model.hidden_size),
              stack_size(model.stack_size),
              output_size(model.output_size) {
            if (use_recursive_gates) {
                prediction_gate = make_shared<StackedInputLayer<T>>(*model.prediction_gate, copy_w, copy_dw);
            } else {
                prediction_gate = nullptr;
            }
            if (convolution) {
                filters = make_shared<Filters<T>>(*model.filters, copy_w, copy_dw);
            } else {
                filters = nullptr;
            }
        }

        BidirectionalLSTM<T> shallow_copy() const {
            return BidirectionalLSTM(*this, false, true);
        }


        std::tuple<Mat<T>, Mat<T>> activate_sequence(Indexing::Index example, T drop_prob = 0.0) {
            size_t pass = 0;
            vector<Mat<T>> forwardX;
            vector<Mat<T>> backwardX;
            vector<Mat<T>> embeddingX;
            assert(example.size() > 0);
            std::tuple<Mat<T>, Mat<T>> prediction_tuple;

            if (convolution) {
                auto convolved = MatOps<T>::conv1d(
                    embedding[example],
                    filters->filters,
                    example.size() < 2
                ).tanh();
                for (size_t i = 0; i < convolved.dims(1); i++) {
                    embeddingX.emplace_back(convolved(NULL, i));
                    // null as first argument means "for all rows take this column"
                    forwardX.push_back(embeddingX.back());
                    backwardX.push_back(forwardX.back());
                }
            } else {
                for (size_t i = 0; i < example.size(); i++) {
                    embeddingX.emplace_back(embedding[example[i]]);
                    forwardX.push_back(embeddingX.back());
                    backwardX.push_back(forwardX.back());
                }
            }
            auto state = stacked_lstm.cells[0].initial_states();
            for (auto& cell : stacked_lstm.cells) {
                if (pass != 0) {
                    state = cell.initial_states();
                }
                if ((pass % 2 == 0) || !bidirectional) {
                    for (auto it_back = backwardX.begin(),
                              it_forward = forwardX.begin(),
                              it_embed = embeddingX.begin();
                        (it_back != backwardX.end() && it_forward != forwardX.end() && it_embed != embeddingX.end());
                        ++it_back, ++it_forward, ++it_embed) {
                        if (cell.input_sizes.size() == 2) {
                            // if the model used is bidirectional
                            // then the backward stream should be available
                            if (bidirectional) {
                                state = cell.activate_shortcut(
                                    MatOps<REAL_t>::dropout_normalized(*it_forward, drop_prob),
                                    MatOps<REAL_t>::dropout_normalized(*it_back, drop_prob),
                                    state);
                            } else {
                                // else the "shortcut" stream is the lowest level's embeddings
                                // simply.
                                state = cell.activate_shortcut(
                                    MatOps<REAL_t>::dropout_normalized(*it_forward, drop_prob),
                                    MatOps<REAL_t>::dropout_normalized(*it_embed, drop_prob),
                                    state);
                            }
                        } else {
                            state = cell.activate(
                                MatOps<REAL_t>::dropout_normalized(*it_forward, drop_prob),
                                state);
                        }
                        // prepare the observation sequence to be fed to the next
                        // level up:
                        *it_forward = state.hidden;
                    }
                } else {
                    for (auto it_back = backwardX.rbegin(),
                              it_forward = forwardX.rbegin();
                        (it_back != backwardX.rend() && it_forward != forwardX.rend());
                        ++it_back, ++it_forward) {

                        if (cell.input_sizes.size() == 2) {
                            state = cell.activate_shortcut(
                                MatOps<REAL_t>::dropout_normalized(*it_back, drop_prob),
                                MatOps<REAL_t>::dropout_normalized(*it_forward, drop_prob),
                                state);
                        } else {
                            state = cell.activate(
                                MatOps<REAL_t>::dropout_normalized(*it_back, drop_prob),
                                state);
                        }
                        // prepare the observation sequence to be fed to the next
                        // level up:
                        *it_back = state.hidden;
                    }
                }
                pass+=1;
            }
            // store memory here:
            std::get<0>(prediction_tuple) = Mat<T>(forwardX.size(), 1);
            if (use_recursive_gates) {
                // Create a flat distribution
                auto prediction = MatOps<T>::consider_constant(
                                    MatOps<T>::fill(
                                        Mat<T>(1, output_size),
                                        1.0 / output_size
                                    )
                                );
                auto total_memory = MatOps<T>::consider_constant(Mat<T>(1, 1));
                // With recursive gates the prediction happens all along
                // we rerun through the highest layer's predictions:
                int step = 0;
                for (auto it_back = backwardX.begin(), it_forward = forwardX.begin(), it_embed = embeddingX.begin();
                        (it_back != backwardX.end() && it_forward != forwardX.end() && it_embed != embeddingX.end());
                        ++it_back, ++it_forward, ++it_embed)                                      {
                    // get a value between 0 and 1 for how much we keep
                    // update the current prediction

                    auto new_memory = prediction_gate->activate({
                        *it_forward,
                        bidirectional ? *it_back : *it_embed,
                        prediction
                    }).sigmoid();

                    // add this memory to output:
                    std::get<0>(prediction_tuple).w(step) = new_memory.w(0);
                    step++;
                    // Make a new prediction:
                    auto new_prediction = MatOps<T>::softmax_rowwise(decoder.activate({
                        apply_dropout(*it_forward, drop_prob),
                        apply_dropout(bidirectional ? *it_back : *it_embed, drop_prob)
                    }));

                    // Update the prediction using the alpha value (tradeoff between old and new)
                    prediction = (
                        new_prediction.eltmul_broadcast_colwise(new_memory) +
                        prediction.eltmul_broadcast_colwise(1.0 - new_memory)
                    );
                    // penalize wavering memory (commit to something)
                    if (graph::backprop_enabled() && memory_penalty > 0) {
                        // penalize memory
                        new_memory = new_memory * memory_penalty;
                        new_memory.grad();
                    }
                }
                std::get<1>(prediction_tuple) = prediction;
            } else {
                if (!bidirectional) {
                    std::get<1>(prediction_tuple) = MatOps<T>::softmax_rowwise(decoder.activate(
                            apply_dropout(forwardX.back(), drop_prob)
                    ));
                } else if (pass % 2 == 0) {
                    // then we ended with backward pass
                    std::get<1>(prediction_tuple) = MatOps<T>::softmax_rowwise(decoder.activate({
                        apply_dropout(backwardX.front(), drop_prob),
                        apply_dropout(forwardX.back(), drop_prob)
                    }));
                } else {
                    // then we ended with forward pass
                    std::get<1>(prediction_tuple) = MatOps<T>::softmax_rowwise(decoder.activate({
                        apply_dropout(forwardX.back(), drop_prob),
                        apply_dropout(backwardX.front(), drop_prob)
                    }));
                }
            }
            return prediction_tuple;
        }

        vector<Mat<T>> parameters() const {
            auto params = stacked_lstm.parameters();
            auto decoder_params = decoder.parameters();
            params.insert(params.end(), decoder_params.begin(), decoder_params.end());
            params.push_back(embedding);
            if (use_recursive_gates) {
                auto gate_params = prediction_gate->parameters();
                params.insert(params.end(), gate_params.begin(), gate_params.end());
            }
            if (convolution) {
                auto filters_params = filters->parameters();
                params.insert(params.end(), filters_params.begin(), filters_params.end());
            }
            return params;
        }


        void save(std::string dirname) const {
            utils::ensure_directory(dirname);
            // Save the matrices:
            auto params = parameters();
            utils::save_matrices(params, dirname);
            dirname += "config.md";
            save_configuration(dirname);
        }

        void save_configuration(std::string fname) const {
            auto config = configuration();
            utils::map_to_file(config, fname);
        }

};

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

    if (FLAGS_patience > 40) FLAGS_patience = 40;

    int memory_penalty_curve_type;
    if (FLAGS_memory_penalty_curve == "flat") {
        memory_penalty_curve_type = 0;
    } else if (FLAGS_memory_penalty_curve == "linear") {
        memory_penalty_curve_type = 1;
    } else if (FLAGS_memory_penalty_curve == "square") {
        memory_penalty_curve_type = 2;
    } else {
        utils::assert2(false, "memory_penalty_curve can only be flat, linear, or square.");
    }

    auto epochs     = FLAGS_epochs;
    int rampup_time = 10;
    auto sentiment_treebank = SST::load(FLAGS_train);
    auto embedding          = Mat<REAL_t>(100, 0);
    auto word_vocab         = Vocab();
    if (!FLAGS_pretrained_vectors.empty()) {
        glove::load(FLAGS_pretrained_vectors, &embedding, &word_vocab, 50000);
    } else {
        word_vocab = SST::get_vocabulary(sentiment_treebank, FLAGS_min_occurence);
    }
    auto vocab_size     = word_vocab.size();
    auto dataset        = SST::convert_trees_to_indexed_minibatches(
        word_vocab,
        sentiment_treebank,
        FLAGS_minibatch
    );
    auto validation_set = SST::convert_trees_to_indexed_minibatches(
        word_vocab,
        SST::load(FLAGS_validation),
        FLAGS_minibatch
    );

    std::cout << " Unique Trees Loaded : " << sentiment_treebank.size() << std::endl
              << "        Example tree : " << *sentiment_treebank[sentiment_treebank.size()-1] << std::endl
              << "     Vocabulary size : " << vocab_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout type : " << (FLAGS_fast_dropout ? "fast" : "default") << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "           LSTM type : " << (FLAGS_memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << std::max(FLAGS_stack_size, 1) << std::endl
              << "       Loss function : " << (FLAGS_surprise ? "Categorical Surprise" : "KL divergence") << std::endl
              << " # training examples : " << dataset.size() * FLAGS_minibatch - (FLAGS_minibatch - dataset[dataset.size() - 1].size()) << std::endl;

    pool = new ThreadPool(FLAGS_j);

    /*
    Create a model with an embedding, and several stacks:
    */

    auto stack_size  = std::max(FLAGS_stack_size, 1);

    auto model = BidirectionalLSTM<REAL_t>(
         FLAGS_pretrained_vectors.empty() ? word_vocab.size() : 0,
         FLAGS_pretrained_vectors.empty() ? FLAGS_hidden : embedding.dims(1),
         FLAGS_hidden,
         stack_size,
         SST::label_names.size(),
         FLAGS_shortcut,
         FLAGS_memory_feeds_gates,
         FLAGS_recursive_gates,
         FLAGS_convolution,
         FLAGS_bidirectional);

    if (!FLAGS_pretrained_vectors.empty()) {
        model.embedding = embedding;
    }

    vector<vector<Mat<REAL_t>>> thread_params;
    vector<BidirectionalLSTM<REAL_t>> thread_models;
    std::tie(thread_models, thread_params) = utils::shallow_copy(model, FLAGS_j);
    auto params = model.parameters();
    auto pred_fun = [&model](vector<uint>& example){
        graph::NoBackprop nb;
        return std::get<1>(model.activate_sequence(
            &example, // see an example
            0.0      // activate without dropout
        )).argmax(); // no softmax needed, simply get best guess
    };
    auto solver = Solver::construct(FLAGS_solver, params, (REAL_t)FLAGS_learning_rate);

    std::tuple<REAL_t, REAL_t> best_validation_score(0.0, 0.0);
    int epoch = 0, best_epoch = 0;
    double patience = 0;
    string best_file = "";
    REAL_t best_score = 0.0;

    shared_ptr<Visualizer> visualizer;
    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer);

    while (patience < FLAGS_patience && epoch < epochs) {

        if (memory_penalty_curve_type == 1) { // linear
            model.memory_penalty = (REAL_t) std::min(
                FLAGS_memory_penalty,
                (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
            );
            for (auto& thread_model : thread_models) {
                thread_model.memory_penalty = (REAL_t) std::min(
                    FLAGS_memory_penalty,
                    (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
                );
            }
        } else if (memory_penalty_curve_type == 2) { // square
            model.memory_penalty = (REAL_t) std::min(
                FLAGS_memory_penalty,
                (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
            );
            for (auto& thread_model : thread_models) {
                thread_model.memory_penalty = (REAL_t) std::min(
                    FLAGS_memory_penalty,
                    (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
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
            pool->run([&word_vocab, &pred_fun, &visualizer, &thread_params, &thread_models, batch_id, &journalist, &solver, &dataset, &best_validation_score, &batches_processed]() {
                auto& thread_model  = thread_models[ThreadPool::get_thread_number()];
                auto& params        = thread_params[ThreadPool::get_thread_number()];
                auto& minibatch     = dataset[batch_id];
                // many forward steps here:
                // REAL_t err = 0.0;
                for (auto &example : minibatch) {
                    auto probs = std::get<1>(thread_model.activate_sequence(
                            &std::get<0>(example), FLAGS_dropout));
                    Mat<REAL_t> error;
                    if (FLAGS_surprise) {
                        throw std::runtime_error("Not Implemented!");
                        //error = softmax_categorical_surprise(probs, std::get<1>(example));
                    } else {
                        error = MatOps<REAL_t>::cross_entropy_rowwise(probs, std::get<1>(example));
                    }
                    // err += error.w()(0);
                    // auto error = MatOps<REAL_t>::softmax_cross_entropy_colwise(logprobs, std::get<1>(example));
                    if (std::get<2>(example) && FLAGS_root_weight != 1.0) {
                        error = error * FLAGS_root_weight;
                    }
                    error.grad();
                    graph::backward(); // backpropagate
                }
                // show sentiment detection as system learns:
                if (visualizer != nullptr) {
                    visualizer->throttled_feed(seconds(10), [&word_vocab, &visualizer, &minibatch, &thread_model]() {
                            // pick example
                            auto& example = std::get<0>(minibatch[utils::randint(0, minibatch.size()-1)]);
                            // visualization does not backpropagate.
                            graph::NoBackprop nb;
                            // make prediction
                            auto model_out = thread_model.activate_sequence(&example, 0.0);
                            return SST::json_classification(
                                word_vocab.decode(&example),
                                std::get<1>(model_out),
                                std::get<0>(model_out)
                            );
                        }
                    );
                }
                solver->step(params); // One step of gradient descent
                journalist.tick(++batches_processed, std::get<0>(best_validation_score));
            });
        }
        pool->wait_until_idle();
        journalist.done();
        auto new_validation = SST::average_recall(validation_set, pred_fun, FLAGS_j);
        std::cout << "Root recall=" << std::get<1>(new_validation) << std::endl;
        if (solver->method == Solver::METHOD_ADAGRAD) {
            solver->reset_caches(params);
            //embedding_solver->reset_caches(embedding_params);
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
                model.save(FLAGS_save_location);
                best_file = FLAGS_save_location;
            }
        }
    }

    if (!FLAGS_test.empty()) {
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
    }
}
