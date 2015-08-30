#include <algorithm>
#include <atomic>
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
#include "dali/visualizer/visualizer.h"
#include "dali/models/RecurrentEmbeddingModel.h"

using namespace std::placeholders;
using std::atomic;
using std::chrono::seconds;
using std::ifstream;
using std::make_shared;
using std::min;
using std::ofstream;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::tie;
using std::tuple;
using std::vector;
using utils::assert2;
using utils::vsum;
using utils::Vocab;
using utils::CharacterVocab;
using utils::MS;

typedef double REAL_t;
// training flags
DEFINE_int32(minibatch,                5,          "What size should be used for the minibatches ?");
DEFINE_int32(patience,                 5,          "How many unimproving epochs to wait through before witnessing progress ?");
// files
DEFINE_string(results_file,            "",         "Where to save test performance.");
DEFINE_string(save_location,           "",         "Where to save test performance.");
// solvers
DEFINE_double(reg,                     0.0,        "What penalty to place on L2 norm of weights?");
// model
DEFINE_int32(input_size,               100,        "Size of embeddings.");
DEFINE_int32(hidden,                   100,        "What hidden size to use.");
DEFINE_int32(stack_size,               2,          "How tall is the great LSTM tower.");
DEFINE_int32(gate_second_order,        50,         "How many second order terms to consider in gate");
DEFINE_bool(lstm_shortcut,             true,       "Should shortcut be used in LSTMs");
DEFINE_bool(lstm_memory_feeds_gates,   false,      "Should memory be fed to gates in LSTMs");
DEFINE_double(dropout,                 0.3,        "How much dropout noise to add to the problem ?");
DEFINE_double(memory_penalty,          0.1,        "Coefficient in front of memory penalty");
DEFINE_string(memory_penalty_curve,    "flat",     "Type of annealing used on gate memory penalty (flat, linear, square)");
// features
DEFINE_bool(svd_init,                  true,       "Initialize weights using SVD?");
DEFINE_bool(end_token,                 true,       "Whether to add a token indicating end of sentence.");
DEFINE_bool(use_characters,            true,       "Use word embeddings or character embeddings?");
DEFINE_int32(negative_samples,         30,         "Create random pairs of paraphrases and treat as negative samples.");
DEFINE_bool(show_similar,              true,       "Visualize neighboring representations?");
DEFINE_int32(number_of_comparisons,    20,         "How many sentences should be compared?");
ThreadPool* pool;


template<typename T>
class SparseStackedLSTM : public StackedLSTM<T> {
    public:
        typedef typename StackedLSTM<T>::state_t state_t;

        SecondOrderCombinator<T> gate_encoder;

        SparseStackedLSTM() : StackedLSTM<T>() {
        }

        SparseStackedLSTM(const SparseStackedLSTM<T>& other, bool copy_w, bool copy_dw) :
            StackedLSTM<T>(other, copy_w, copy_dw),
            gate_encoder(other.gate_encoder, copy_w, copy_dw) {
        }

        SparseStackedLSTM shallow_copy() const {
            return SparseStackedLSTM(*this, false, true);
        }

        vector<Mat<T>> parameters() const {
            auto params = StackedLSTM<T>::parameters();
            auto gate_params = gate_encoder.parameters();
            params.insert(params.end(), gate_params.begin(), gate_params.end());
            return params;
        }

        SparseStackedLSTM(int input_size,
                          vector<int> hidden_sizes,
                          int gate_second_order,
                          bool shortcut,
                          bool memory_feeds_gates) :
                StackedLSTM<T>(input_size, hidden_sizes, shortcut, memory_feeds_gates),
                gate_encoder(input_size, vsum(hidden_sizes), gate_second_order) {
        }

        Mat<T> activate_gate(Mat<T> input, Mat<T> hidden) const {
            return gate_encoder.activate(input, hidden).sum().sigmoid();
        }

        // returns next state and memory
        std::tuple<state_t, Mat<T>> activate(Mat<T> input, state_t prev_state, T dropout_probability) const {
            auto current_hiddens =  MatOps<T>::hstack(LSTM<T>::activation_t::hiddens(prev_state));
            auto gate_memory     =  activate_gate(input, current_hiddens);
            auto gated_input     =  input.eltmul_broadcast_colwise(gate_memory);
            auto next_state      =  StackedLSTM<T>::activate(prev_state, gated_input, dropout_probability);

            return std::make_tuple(next_state, gate_memory);
        }

        // returns last state and memory at every step
        std::tuple<state_t, vector<Mat<T>>> activate_sequence(vector<Mat<T>> inputs,
                                                              state_t state,
                                                              T dropout_probability) const {
            vector<Mat<T>> memories;
            for (auto& input: inputs) {
                Mat<T> memory;
                std::tie(state, memory) = activate(input, state, dropout_probability);
                memories.push_back(memory);
            }
            return make_tuple(state, memories);
        }
};

template<typename T>
class ParaphraseModel : public RecurrentEmbeddingModel<T> {
    typedef typename StackedLSTM<T>::state_t state_t;

    public:
        SparseStackedLSTM<T> sentence_encoder;
        Mat<T> end_of_sentence_token;

        vector<LSTMState<T>> initial_states() const {
            return sentence_encoder.initial_states();
        }

        ParaphraseModel(int _vocabulary_size,
                        int _input_size,
                        vector<int> _hidden_sizes,
                        int gate_second_order,
                        T dropout_probability) :
            RecurrentEmbeddingModel<T>(_vocabulary_size, _input_size, _hidden_sizes, 0),
                sentence_encoder(_input_size,
                                 _hidden_sizes,
                                 gate_second_order,
                                 FLAGS_lstm_shortcut,
                                 FLAGS_lstm_memory_feeds_gates),
                end_of_sentence_token(1, _input_size, weights<T>::uniform(1.0 / _input_size)) {}

        ParaphraseModel(const ParaphraseModel& other, bool copy_w, bool copy_dw) :
                RecurrentEmbeddingModel<T>(other, copy_w, copy_dw),
                sentence_encoder(other.sentence_encoder, copy_w, copy_dw),
                end_of_sentence_token(other.end_of_sentence_token, copy_w, copy_dw) {}

        ParaphraseModel<T> shallow_copy() const {
            return ParaphraseModel<T>(*this, false, true);
        }

        vector<Mat<T>> parameters() const {
            auto params = RecurrentEmbeddingModel<T>::parameters();
            auto sentence_encoder_params = sentence_encoder.parameters();
            params.insert(params.end(), sentence_encoder_params.begin(), sentence_encoder_params.end());
            params.emplace_back(end_of_sentence_token);
            return params;
        }

        // returns sentence and vector of memories
        std::tuple<Mat<T>, vector<Mat<T>>> encode_sentence(vector<uint> sentence, T drop_prob) const {
            vector<Mat<T>> embeddings;
            for (auto& word_idx: sentence) {
                embeddings.emplace_back(this->embedding[word_idx]);
            }
            if (FLAGS_end_token)
                embeddings.push_back(end_of_sentence_token);

            auto state = sentence_encoder.initial_states();
            vector<Mat<T>> memories;
            std::tie(state, memories) = sentence_encoder.activate_sequence(
                embeddings,
                state,
                drop_prob);
            auto sentence_hidden = MatOps<T>::hstack(LSTM<T>::activation_t::hiddens(state));
            return std::make_tuple(sentence_hidden, memories);
        }

        Mat<T> cosine_distance(Mat<T> s1, Mat<T> s2) const {
            return (s1 * s2).sum() / (s1.L2_norm() * s2.L2_norm());
        }

        std::tuple<Mat<T>, vector<Mat<T>>, vector<Mat<T>>> similarity(vector<uint> sentence1,
                                                                      vector<uint> sentence2,
                                                                      T drop_prob) const {
            Mat<T> sentence1_hidden, sentence2_hidden;
            vector<Mat<T>> sentence1_memories, sentence2_memories;
            std::tie(sentence1_hidden, sentence1_memories) =
                    encode_sentence(sentence1, drop_prob);
            std::tie(sentence2_hidden, sentence2_memories) =
                    encode_sentence(sentence2, drop_prob);

            auto similarity_score = cosine_distance(sentence1_hidden, sentence2_hidden );
            return std::make_tuple(similarity_score, sentence1_memories, sentence2_memories);
        }

        tuple<Mat<T>, vector<Mat<T>>, vector<Mat<T>>> error(
                const vector<uint>& sentence1,
                const vector<uint>& sentence2,
                T drop_prob,
                double correct_score) const {
            Mat<T> similarity_score;
            vector<Mat<T>> memory1, memory2;
            std::tie(similarity_score, memory1, memory2) = similarity(sentence1, sentence2, drop_prob);

            auto error = (similarity_score - correct_score)^2;
            return std::make_tuple(error, memory1, memory2);
        }

        tuple<double, vector<double>, vector<double>> predict_with_memories(vector<uint> sentence1, vector<uint> sentence2) const {
            graph::NoBackprop nb;
            vector<Mat<T>> memory1_mat, memory2_mat;
            Mat<T> similarity_mat;
            std::tie(similarity_mat, memory1_mat, memory2_mat) =
                    similarity(sentence1, sentence2, 0.0);

            auto extract_double  = [](Mat<T> m) { return m.w(0,0); };
            vector<double> memory1, memory2;

            std::transform(memory1_mat.begin(), memory1_mat.end(), std::back_inserter(memory1), extract_double);
            std::transform(memory2_mat.begin(), memory2_mat.end(), std::back_inserter(memory2), extract_double);

            auto similarity_score = extract_double(similarity_mat);
            return make_tuple(similarity_score, memory1, memory2);
        }

        double predict(vector<uint> sentence1, vector<uint> sentence2) const {
            double score;
            vector<double> ignored1, ignored2;
            std::tie(score, ignored1, ignored2) = predict_with_memories(sentence1, sentence2);
            return score;
        }

};

typedef ParaphraseModel<REAL_t> model_t;

std::tuple<Vocab, CharacterVocab, typename paraphrase::paraphrase_minibatch_dataset, typename paraphrase::paraphrase_minibatch_dataset, typename paraphrase::paraphrase_minibatch_dataset> load_data(
        double data_split,
        int batch_size,
        bool use_characters,
        int min_word_occurence,
        int max_training_examples) {
    using namespace paraphrase;
    // combine two datasets by mixing their generators:
    auto train      = STS_2014::generate_train() + wikianswers::generate();
    auto word_vocab = Vocab(get_vocabulary(train, min_word_occurence, FLAGS_use_characters ? 0 : 300000), true);
    auto char_vocab = CharacterVocab(32, 255);

    std::cout << "loaded vocabulary" << std::endl;

    auto dataset       = use_characters ? convert_to_indexed_minibatches(char_vocab, train, batch_size)
                                        : convert_to_indexed_minibatches(word_vocab, train, batch_size);

    auto test = STS_2014::load_test();
    auto test_dataset  = use_characters ? convert_to_indexed_minibatches(char_vocab, test, batch_size)
                                        : convert_to_indexed_minibatches(word_vocab, test, batch_size);


    std::cout << "got test set" << std::endl;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0, 1);
    std::random_device rd;
    generator.seed(rd());

    decltype(test_dataset) validation_data;
    decltype(test_dataset) training_data;

    int seen = 0;
    for (auto example : dataset) {
        seen+=example.size();
        if (max_training_examples > -1 && seen > max_training_examples) break;

        if (distribution(generator) > data_split)
            validation_data.emplace_back(example);
        else
            training_data.emplace_back(example);
    }

    std::cout << "Loaded training data" << std::endl;
    // implement validation set online.
    return make_tuple(word_vocab, char_vocab, training_data, validation_data, test_dataset);
}

void backprop_example(
        const model_t& thread_model,
        const vector<uint>& sentence1,
        const vector<uint>& sentence2,
        const double& label,
        const double& memory_penalty,
        const int& minibatch_size,
        double& minibatch_error) {
    Mat<REAL_t> partial_error;
    vector<Mat<REAL_t>> memory1, memory2;
    std::tie(partial_error, memory1, memory2) =
            thread_model.error(sentence1, sentence2, FLAGS_dropout, label);

    if (memory_penalty > 0) {
        auto memory = MatOps<REAL_t>::add(memory1) + MatOps<REAL_t>::add(memory2);
        partial_error = partial_error + memory_penalty * memory;
    }

    partial_error = partial_error / minibatch_size;
    minibatch_error += partial_error.w(0,0);

    partial_error.grad();
    graph::backward(); // backpropagate
}

void backprop_random_example(
        const model_t& thread_model,
        const typename paraphrase::paraphrase_minibatch_dataset& dataset,
        const double& memory_penalty,
        const int& minibatch_size,
        double& minibatch_error) {
    auto batch_id_1 = utils::randint(0, dataset.size() - 1);
    auto ex_id_1    = utils::randint(0, dataset[batch_id_1].size() - 1);
    auto ex_id_1_b  = utils::randint(0, 1);

    auto batch_id_2 = utils::randint(0, dataset.size() - 1);
    int  ex_id_2    = utils::randint(0, dataset[batch_id_2].size() - 1);
    auto ex_id_2_b  = utils::randint(0, 1);

    // make sure they are different
    if (batch_id_1 == batch_id_2 && ex_id_1 == ex_id_2) {
        if (dataset[batch_id_2].size() > 2) {
            while (ex_id_2 == ex_id_1) {
                ex_id_2 = utils::randint(0, dataset[batch_id_2].size() - 1);
            }
        } else {
            while (batch_id_1 == batch_id_2) {
                batch_id_2 = utils::randint(0, dataset.size() - 1);
            }
            ex_id_2 = utils::randint(0, dataset[batch_id_2].size() - 1);
        }
    }

    if (ex_id_1_b == 1) {
        if (ex_id_2_b == 1) {
            backprop_example(
                thread_model,
                std::get<1>(dataset[batch_id_1][ex_id_1]),
                std::get<1>(dataset[batch_id_2][ex_id_2]),
                0.0,
                memory_penalty,
                minibatch_size,
                minibatch_error);
        } else {
            backprop_example(
                thread_model,
                std::get<1>(dataset[batch_id_1][ex_id_1]),
                std::get<0>(dataset[batch_id_2][ex_id_2]),
                0.0,
                memory_penalty,
                minibatch_size,
                minibatch_error);
        }
    } else {
        if (ex_id_2_b == 1) {
            backprop_example(
                thread_model,
                std::get<0>(dataset[batch_id_1][ex_id_1]),
                std::get<1>(dataset[batch_id_2][ex_id_2]),
                0.0,
                memory_penalty,
                minibatch_size,
                minibatch_error);
        } else {
            backprop_example(
                thread_model,
                std::get<0>(dataset[batch_id_1][ex_id_1]),
                std::get<0>(dataset[batch_id_2][ex_id_2]),
                0.0,
                memory_penalty,
                minibatch_size,
                minibatch_error);
        }
    }
}

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

    auto epochs = FLAGS_epochs;
    int rampup_time = 10;

    auto dataset = load_data(
        0.9,
        FLAGS_minibatch,       // minibatch size
        FLAGS_use_characters,  // use characters or words
        FLAGS_min_occurence,   // min word appearance to be in vocab
        200000
    );

    auto& word_vocab      = std::get<0>(dataset);
    auto& char_vocab      = std::get<1>(dataset);
    auto& training_set    = std::get<2>(dataset);
    auto& validation_set  = std::get<3>(dataset);
    auto& test_set        = std::get<4>(dataset);

    pool = new ThreadPool(FLAGS_j);



    auto model = model_t(FLAGS_use_characters ? char_vocab.size() : word_vocab.size(),
                         FLAGS_input_size,
                         vector<int>(FLAGS_stack_size, FLAGS_hidden),
                         FLAGS_gate_second_order,
                         FLAGS_dropout);

    if (FLAGS_lstm_shortcut && FLAGS_stack_size == 1)
        std::cout << "shortcut flag ignored: Shortcut connections only take effect with stack size > 1" << std::endl;

    int total_examples = 0;
    for (auto minibatch: training_set)
        total_examples += (minibatch.size() + FLAGS_negative_samples);

    std::cout << "     Vocabulary size : " << model.vocabulary_size << std::endl
              << "      minibatch size : " << FLAGS_minibatch << std::endl
              << "   training examples : " << total_examples << std::endl
              << "   number of threads : " << FLAGS_j << std::endl
              << "        Dropout Prob : " << FLAGS_dropout << std::endl
              << " Max training epochs : " << FLAGS_epochs << std::endl
              << "   First Hidden Size : " << model.hidden_sizes[0] << std::endl
              << "           LSTM type : " << (FLAGS_lstm_memory_feeds_gates ? "Graves 2013" : "Zaremba 2014") << std::endl
              << "          Stack size : " << model.hidden_sizes.size() << std::endl
              << "              Solver : " << FLAGS_solver << std::endl;

    vector<vector<Mat<REAL_t>>> thread_params;
    vector<model_t> thread_models;
    std::tie(thread_models, thread_params) = utils::shallow_copy(model, FLAGS_j);
    auto params = model.parameters();
    auto solver = Solver::construct<REAL_t>(FLAGS_solver, params, FLAGS_learning_rate, (REAL_t) FLAGS_reg);

    REAL_t best_validation_score = 0.0;
    int best_epoch = 0, epoch = 0;
    double patience = 0;
    string best_file = "";
    REAL_t best_score = 0.0;

    shared_ptr<Visualizer> visualizer;
    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer);

    // if no training should occur then use the validation set
    // to see how good the loaded model is.
    if (epochs == 0) {
        best_validation_score = paraphrase::pearson_correlation(
                validation_set,
                std::bind(&model_t::predict, &model, _1, _2),
                FLAGS_j);
        std::cout << "correlation = " << best_validation_score << std::endl;
    }

    while (patience < FLAGS_patience && epoch < epochs) {

        double memory_penalty = 0.0;
        if (memory_penalty_curve_type == 1) { // linear
            memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch) / (double)(rampup_time))))
            );
        } else if (memory_penalty_curve_type == 2) { // square
            memory_penalty = std::min(
                FLAGS_memory_penalty,
                (REAL_t) (FLAGS_memory_penalty * std::min(1.0, ((double)(epoch * epoch) / (double)(rampup_time * rampup_time))))
            );
        }

        atomic<int> examples_processed(0);
        ReportProgress<double> journalist(
            utils::MS() << "Epoch " << ++epoch, // what to say first
            total_examples // how many steps to expect before being done with epoch
        );

        for (auto minibatch : training_set) {
            pool->run([&word_vocab, &char_vocab, &visualizer, &thread_params, &thread_models,
                       minibatch, &journalist, &solver, &training_set,
                       &best_validation_score, &examples_processed,
                       &memory_penalty, &total_examples]() {
                auto& thread_model     = thread_models[ThreadPool::get_thread_number()];
                auto& params           = thread_params[ThreadPool::get_thread_number()];
                // many forward steps here:
                REAL_t minibatch_error = 0.0;
                for (auto& example : minibatch) {
                    backprop_example(
                        thread_model,
                        std::get<0>(example),
                        std::get<1>(example),
                        std::get<2>(example),
                        memory_penalty,
                        minibatch.size(),
                        minibatch_error);
                    examples_processed++;
                }
                // One step of gradient descent
                solver->step(params);

                if (total_examples >= 2 && FLAGS_negative_samples > 0) {
                    for (int artificial_idx = 0; artificial_idx < FLAGS_negative_samples; artificial_idx++) {
                        backprop_random_example(
                            thread_model,
                            training_set,
                            memory_penalty,
                            FLAGS_negative_samples,
                            minibatch_error);
                        examples_processed++;
                    }
                    // One step of gradient descent
                    solver->step(params);
                }

                journalist.tick(examples_processed, minibatch_error);

                if (visualizer != nullptr) {
                    visualizer->throttled_feed(seconds(5), [&total_examples, &word_vocab, &training_set, &char_vocab, &visualizer, minibatch, &thread_model]() {
                        graph::NoBackprop nb;
                        // pick example
                        vector<uint> sentence1, sentence2;
                        double true_score;

                        auto example_id = utils::randint(0, minibatch.size()-1);
                        std::tie(sentence1, sentence2, true_score) = minibatch[example_id];

                        if (FLAGS_show_similar) {
                            auto& sampled_sentence = utils::randint(0, 1) > 0 ? sentence2 : sentence1;
                            auto cosine_distance = [&thread_model](const std::vector<uint>& s1, const std::vector<uint>& s2) {
                                    auto original_encoded = thread_model.encode_sentence(s1, 0.0);
                                    auto other_encoded    = thread_model.encode_sentence(s2, 0.0);
                                    return (double) thread_model.cosine_distance(
                                        std::get<0>(original_encoded),
                                        std::get<0>(other_encoded)
                                    ).w(0,0);
                            };
                            if (FLAGS_use_characters) {
                                return paraphrase::nearest_neighbors(
                                    char_vocab, sampled_sentence, training_set, cosine_distance, FLAGS_number_of_comparisons
                                );
                            } else {
                                return paraphrase::nearest_neighbors(
                                    word_vocab, sampled_sentence, training_set, cosine_distance, FLAGS_number_of_comparisons
                                );
                            }
                        } else {
                            double predicted_score;
                            vector<double> memory1, memory2;
                            std::tie(predicted_score, memory1, memory2) =
                                    thread_model.predict_with_memories(sentence1, sentence2);

                            auto vs1  = make_shared<visualizable::Sentence<REAL_t>>(
                                FLAGS_use_characters ? char_vocab.decode_characters(&sentence1) :
                                                       word_vocab.decode(&sentence1));
                            vs1->set_weights(memory1);
                            vs1->spaces = !FLAGS_use_characters;

                            auto vs2  = make_shared<visualizable::Sentence<REAL_t>>(
                                FLAGS_use_characters ? char_vocab.decode_characters(&sentence2) :
                                                       word_vocab.decode(&sentence2));
                            vs2->set_weights(memory2);
                            vs2->spaces = !FLAGS_use_characters;

                            auto msg1 = make_shared<visualizable::Message>(MS() << "Predicted similarity: " << predicted_score);
                            auto msg2 = make_shared<visualizable::Message>(MS() << "True similarity: " << true_score);

                            auto grid = make_shared<visualizable::GridLayout>();

                            grid->add_in_column(0, vs1);
                            grid->add_in_column(0, vs2);
                            grid->add_in_column(1, msg1);
                            grid->add_in_column(1, msg2);

                            return grid->to_json();
                        }
                    });
                }
            });
        }
        pool->wait_until_idle();
        journalist.done();
        auto new_validation = paraphrase::pearson_correlation(
            validation_set,
            std::bind(&model_t::predict, &model, _1, _2),
            FLAGS_j);

        if (solver->method == Solver::METHOD_ADAGRAD) solver->reset_caches(params);

        if (new_validation + 1e-6 < best_validation_score) {
            // lose patience:
            patience += 1;
        } else {
            // recover some patience:
            patience = 0.0;
            best_validation_score = new_validation;
        }
        if (best_validation_score != new_validation) {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << best_validation_score << " ("<< new_validation << "), patience = " << patience << std::endl;
        } else {
            std::cout << "Epoch (" << epoch << ") Best validation score = " << best_validation_score << ", patience = " << patience << std::endl;
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

    if (!FLAGS_save_location.empty() && !best_file.empty()) {
        std::cout << "loading from best validation parameters \"" << best_file << "\"" << std::endl;
        auto params = model.parameters();
        utils::load_matrices(params, best_file);
    }

    // write test code and reporting here.
    auto test_score = paraphrase::pearson_correlation(test_set, std::bind(&model_t::predict, &model, _1, _2), FLAGS_j);
    auto acc        = paraphrase::binary_accuracy(test_set, std::bind(&model_t::predict, &model, _1, _2), FLAGS_j);

    std::cout << "Done training" << std::endl;
    std::cout << "Test correlation "    << test_score
                     << ", recall "     << acc.recall() * 100.0
                     << "%, precision " << acc.precision() << std::endl;
    if (!FLAGS_results_file.empty()) {
        ofstream fp;
        fp.open(FLAGS_results_file.c_str(), std::ios::out | std::ios::app);
        fp         << FLAGS_solver
           << "\t" << FLAGS_minibatch
           << "\t" << "std"
           << "\t" << FLAGS_dropout
           << "\t" << FLAGS_hidden
           << "\t" << test_score
           << "\t" << acc.recall()
           << "\t" << acc.precision()
           << "\t" << acc.F1()
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
    // Write test accuracy here.
}
