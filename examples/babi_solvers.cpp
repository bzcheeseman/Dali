#include <algorithm>
#include <chrono>
#include <iostream>
#include <set>
#include <vector>
#include <json11.hpp>
#include <gflags/gflags.h>
#include "dali/data_processing/babi.h"
#include "dali/core.h"
#include "dali/utils.h"
#include "dali/mat/model.h"
#include "dali/visualizer/visualizer.h"

using namespace std::placeholders;
using babi::Fact;
using babi::Item;
using babi::QA;
using babi::Story;
using std::get;
using std::make_shared;
using std::make_tuple;
using std::set;
using std::shared_ptr;
using std::string;
using std::tie;
using std::tuple;
using std::vector;
using utils::in_vector;
using utils::reversed;
using utils::Timer;
using utils::Vocab;
using Eigen::MatrixXd;
using std::chrono::seconds;
using utils::assert2;
using utils::MS;
using std::to_string;

DEFINE_int32(j,                        1,      "Number of threads");
DEFINE_bool(solver_mutex,              false,  "Synchronous execution of solver step.");
DEFINE_int32(minibatch,                100,    "How many stories to put in a single batch.");
DEFINE_int32(max_epochs,               2000,   "maximum number of epochs for training");
DEFINE_int32(patience,                 200,   "after <patience> epochs of performance decrease we stop.");
// generic lstm properties
DEFINE_bool(lstm_shortcut,             true,   "Should shortcut be used in LSTMs");
DEFINE_bool(lstm_feed_mry,             true,   "Should memory be fed to gates in LSTMs");
// high level
DEFINE_int32(hl_hidden,                20,     "What hidden size to use for high level.");
DEFINE_int32(hl_stack,                 4,      "How high is the LSTM for high level.");
DEFINE_double(hl_dropout,              0.7,    "How much dropout to use at high level.");
// question gate model
DEFINE_int32(gate_input,               20,     "Embedding size for gates.");
DEFINE_int32(gate_second_order,        10,     "How many second order terms to consider for a gate.");
// text repr model
DEFINE_int32(text_repr_input,          40,     "Embedding size for text_representation models.");
DEFINE_int32(text_repr_hidden,         40,     "What hidden size to use for question representation.");
DEFINE_int32(text_repr_stack,          2,      "How high is the LSTM for text representation");
DEFINE_double(text_repr_dropout,       0.3,    "How much dropout to use for text representation.");
// error
DEFINE_bool(margin_loss,               true,  "Use margin loss instead of cross entropy");
DEFINE_double(margin,                  0.1,    "Margin for margine loss (must use --margin_loss).");
DEFINE_double(unsupporting_ratio,      0.1,    "How much to penalize unsporring facts (for supporting it's 1.0).");
DEFINE_double(fact_selection_lambda,   0.1,    "How much fact selection matters in error.");
DEFINE_double(word_selection_sparsity, 0.0001, "How much does word selection sparsity matters");
DEFINE_string(babi_problem,            "",     "Which problem to benchmark");

static bool dummy3 = GFLAGS_NAMESPACE::RegisterFlagValidator(
            &FLAGS_babi_problem, &utils::validate_flag_nonempty);

typedef float REAL_t;
// Visualizer
std::shared_ptr<Visualizer> visualizer;

template<typename T>
struct StoryActivation {
    Mat<T> log_probs;
    vector<Mat<T>> fact_gate_memory;
    vector<vector<Mat<T>>> word_gate_memory;

    StoryActivation(Mat<T> log_probs,
                    vector<Mat<T>> fact_gate_memory,
                    vector<vector<Mat<T>>> word_gate_memory) :
            log_probs(log_probs),
            fact_gate_memory(fact_gate_memory),
            word_gate_memory(word_gate_memory) {
    }

    Mat<T> fact_memory_sum() {
        Mat<T> res(1,1);
        for(auto& fact_mem: fact_gate_memory) {
            res = res + fact_mem;
        }
        return res;
    }

    Mat<T> word_memory_sum() {
        Mat<T> res(1,1);
        for(auto& fact_words: word_gate_memory) {
            for(auto& word_mem: fact_words) {
                res = res + word_mem;
            }
        }
        return res;
    }
};

template<typename T>
class LstmBabiModel {
    StackedLSTM<T> fact_model;
    Mat<T> fact_embeddings;

    StackedLSTM<T> question_model;
    Mat<T> question_embeddings;

    SecondOrderCombinator<T> word_gate;
    Mat<T> word_gate_embeddings;

    SecondOrderCombinator<T> fact_gate;
    Mat<T> fact_gate_embeddings;

    StackedLSTM<T> hl_model;

    Mat<T> please_start_prediction;

    Layer<T>            decoder;

    public:
        shared_ptr<Vocab> vocab;

        vector<Mat<T>> parameters() {
            vector<Mat<T>> res;
            for (auto model: std::vector<AbstractLayer<T>*>({
                                &fact_model,
                                &question_model,
                                &word_gate,
                                &fact_gate,
                                &hl_model,
                                &decoder })) {
                auto params = model->parameters();
                res.insert(res.end(), params.begin(), params.end());
            }
            for (auto& matrix: { fact_embeddings,
                                 question_embeddings,
                                 word_gate_embeddings,
                                 fact_gate_embeddings,
                                 please_start_prediction
                                 }) {
                res.emplace_back(matrix);
            }

            return res;
        }

        LstmBabiModel<T> shallow_copy() {
            return LstmBabiModel<T>(*this, false, true);
        }

        LstmBabiModel(const LstmBabiModel& model, bool copy_w, bool copy_dw) :
                fact_model(model.fact_model, copy_w, copy_dw),
                question_model(model.question_model, copy_w, copy_dw),
                word_gate(model.word_gate, copy_w, copy_dw),
                fact_gate(model.fact_gate, copy_w, copy_dw),
                hl_model(model.hl_model, copy_w, copy_dw),
                decoder(model.decoder, copy_w, copy_dw),
                // parameters begin here
                fact_embeddings(model.fact_embeddings, copy_w, copy_dw),
                question_embeddings(model.question_embeddings, copy_w, copy_dw),
                word_gate_embeddings(model.word_gate_embeddings, copy_w, copy_dw),
                fact_gate_embeddings(model.fact_gate_embeddings, copy_w, copy_dw),
                please_start_prediction(model.please_start_prediction, copy_w, copy_dw) {
            vocab = model.vocab;
        }

        LstmBabiModel(shared_ptr<Vocab> vocabulary) :
                // first true - shortcut, second true - feed memory to gates
                fact_model(FLAGS_text_repr_input,
                           vector<int>(FLAGS_text_repr_stack, FLAGS_text_repr_hidden),
                           FLAGS_lstm_shortcut,
                           FLAGS_lstm_feed_mry),
                question_model(FLAGS_text_repr_input,
                               vector<int>(FLAGS_text_repr_stack, FLAGS_text_repr_hidden),
                               FLAGS_lstm_shortcut,
                               FLAGS_lstm_feed_mry),
                word_gate(FLAGS_gate_input, FLAGS_gate_input, FLAGS_gate_second_order),
                fact_gate(FLAGS_gate_input,
                          utils::vsum(vector<int>(FLAGS_text_repr_stack, FLAGS_text_repr_hidden)),
                          FLAGS_gate_second_order),
                hl_model(utils::vsum(vector<int>(FLAGS_text_repr_stack, FLAGS_text_repr_hidden)),
                         vector<int>(FLAGS_hl_stack, FLAGS_hl_hidden),
                         FLAGS_lstm_shortcut,
                         FLAGS_lstm_feed_mry),
                decoder(utils::vsum(vector<int>(FLAGS_hl_stack, FLAGS_hl_hidden)),
                        vocabulary->word2index.size()) {

            vocab = vocabulary;
            size_t n_words = vocab->size();

            word_gate_embeddings =
                    Mat<T>(n_words, FLAGS_gate_input,
                           weights<T>::uniform(1.0/FLAGS_gate_input));
            fact_gate_embeddings =
                    Mat<T>(n_words, FLAGS_gate_input,
                           weights<T>::uniform(1.0/FLAGS_gate_input));
            fact_embeddings =
                    Mat<T>(n_words, FLAGS_text_repr_input,
                           weights<T>::uniform(1.0/FLAGS_text_repr_input));
            question_embeddings =
                    Mat<T>(n_words, FLAGS_text_repr_input,
                           weights<T>::uniform(1.0/FLAGS_text_repr_input));
            please_start_prediction =
                    Mat<T>(utils::vsum(vector<int>(FLAGS_text_repr_stack, FLAGS_text_repr_hidden)), 1,
                           weights<T>::uniform(1.0));
        }

        vector<Mat<T>> get_embeddings(const vector<string>& words,
                                      Mat<T> embedding_matrix) const {
            vector<Mat<T>> seq;
            for (auto& word: words) {
                auto question_idx = vocab->word2index.at(word);
                auto embedding = embedding_matrix[question_idx];
                seq.push_back(embedding);
                // We don't need explicitly start prediction token because
                // each question is guaranteed to end with "?" token.
            }
            return seq;
        }

        vector<Mat<T>> apply_gate(vector<Mat<T>> memory,
                                  vector<Mat<T>> seq) const {
            assert(memory.size() == seq.size());
            vector<Mat<T>> gated_seq;
            for(int i=0; i < memory.size(); ++i) {
                gated_seq.push_back(seq[i].eltmul_broadcast_rowwise(memory[i]));
            }
            return gated_seq;
        }

        Mat<T> lstm_final_activation(const vector<Mat<T>>& embeddings,
                                     const StackedLSTM<T>& model,
                                     T dropout_value) const {
            auto out_states = model.activate_sequence(model.initial_states(),
                                                     embeddings,
                                                     dropout_value);
            // out_state.second corresponds to LSTM hidden (as opposed to memory).
            return MatOps<T>::vstack(LSTM<T>::State::hiddens(out_states));
        }

        StoryActivation<T> activate_story(const vector<vector<string>>& facts,
                                          const vector<string>& question,
                                          bool use_dropout) const {
            auto word_gate_embeddings_question = get_embeddings(question, word_gate_embeddings);
            auto word_gate_hidden_question = MatOps<T>::add(word_gate_embeddings_question) /
                                             (T)word_gate_embeddings_question.size();

            vector<Mat<T>> fact_representations;
            vector<vector<Mat<T>>> word_gate_memories;

            for (auto& fact: facts) {
                auto gate_embeddings_fact = get_embeddings(reversed(fact), word_gate_embeddings);
                vector<Mat<T>> word_gate_memory;
                for (auto& gate_hidden_fact_word: gate_embeddings_fact) {
                    word_gate_memory.push_back(
                        word_gate.activate(word_gate_hidden_question, gate_hidden_fact_word).sum().sigmoid()
                    );
                }
                word_gate_memories.push_back(word_gate_memory);

                auto gated_embeddings = apply_gate(
                    word_gate_memory,
                    get_embeddings(reversed(fact), fact_embeddings)
                );

                auto fact_repr = lstm_final_activation(
                        gated_embeddings,
                        fact_model,
                        use_dropout ? FLAGS_text_repr_dropout : 0.0);

                fact_representations.push_back(fact_repr);
            }

            auto fact_gate_embeddings_question = get_embeddings(question, fact_gate_embeddings);
            auto fact_gate_hidden_question = MatOps<T>::add(fact_gate_embeddings_question) /
                                             (T)fact_gate_embeddings_question.size();

            vector<Mat<T>> fact_gate_memory;
            for (auto& fact_representation: fact_representations) {
                fact_gate_memory.push_back(
                    fact_gate.activate(fact_gate_hidden_question, fact_representation).sum().sigmoid()
                );
            }
            auto gated_facts = apply_gate(fact_gate_memory, fact_representations);
            auto question_hidden = lstm_final_activation(
                    get_embeddings(reversed(question), question_embeddings),
                    question_model,
                    use_dropout ? FLAGS_text_repr_dropout : 0.0);

            // There is probably a better way
            vector<Mat<T>> hl_input;
            hl_input.push_back(question_hidden);
            hl_input.insert(hl_input.end(), gated_facts.begin(), gated_facts.end());
            hl_input.push_back(please_start_prediction);

            auto hl_hidden = lstm_final_activation(
                    hl_input, hl_model, use_dropout ? FLAGS_hl_dropout : 0.0);

            auto log_probs = decoder.activate(hl_hidden);

            return StoryActivation<T>(log_probs,
                                      fact_gate_memory,
                                      word_gate_memories);
        }

        void visualize_example(const vector<vector<string>>& facts,
                               const vector<string>& question,
                               const vector<string>& correct_answer) const {
            graph::NoBackprop nb;
            auto activation = activate_story(facts, question, false);

            shared_ptr<visualizable::FiniteDistribution<T>> vdistribution;
            if (FLAGS_margin_loss) {
                auto scores = activation.log_probs;
                std::vector<REAL_t> scores_as_vec;
                for (int i=0; i < scores.dims(0); ++i) {
                    scores_as_vec.push_back(scores.w()(i,0));
                }
                auto distribution_as_vec = utils::normalize_weights(scores_as_vec);
                vdistribution = make_shared<visualizable::FiniteDistribution<T>>(
                        distribution_as_vec, scores_as_vec, vocab->index2word, 5);
            } else {
                auto distribution = MatOps<T>::softmax(activation.log_probs);
                std::vector<REAL_t> distribution_as_vec;
                for (int i=0; i < distribution.dims(0); ++i) {
                    distribution_as_vec.push_back(distribution.w()(i,0));
                }
                vdistribution = make_shared<visualizable::FiniteDistribution<T>>(
                        distribution_as_vec, vocab->index2word, 5);
            }

            vector<REAL_t> facts_weights;
            vector<std::shared_ptr<visualizable::Sentence<T>>> facts_sentences;
            for (int fidx=0; fidx < facts.size(); ++fidx) {
                auto vfact = make_shared<visualizable::Sentence<T>>(facts[fidx]);

                vector<REAL_t> words_weights;

                for (Mat<T> weight: activation.word_gate_memory[fidx]) {
                    words_weights.push_back(weight.w()(0,0));
                }
                vfact->set_weights(words_weights);
                facts_sentences.push_back(vfact);
                facts_weights.push_back(activation.fact_gate_memory[fidx].w()(0,0));
            }

            auto vcontext = make_shared<visualizable::Sentences<T>>(facts_sentences);
            vcontext->set_weights(facts_weights);
            auto vquestion = make_shared<visualizable::Sentence<T>>(question);
            auto vanswer = make_shared<visualizable::Sentence<T>>(correct_answer);

            auto vqa = make_shared<visualizable::QA<T>>(vcontext, vquestion, vanswer);

            auto vgrid = make_shared<visualizable::GridLayout>();
            vgrid->add_in_column(0, vqa);
            vgrid->add_in_column(1, vdistribution);

            visualizer->feed(vgrid->to_json());
        }

        Mat<T> error(const vector<vector<string>>& facts,
                     const vector<string>& question,
                     uint answer_idx,
                     vector<int> supporting_facts) const {
            auto activation = activate_story(facts, question, true);
            Mat<REAL_t> prediction_error;
            if (FLAGS_margin_loss) {
                // We estimated eprically that margin loss scales as about 6.0
                // cross entropy. We want to put the roughly in the same bucket, so
                // so that improtance of gate errors and sparsity has more or less
                // the same effect for both types of errors.
                prediction_error = 6.0 * MatOps<REAL_t>::margin_loss(activation.log_probs, answer_idx, FLAGS_margin);
            } else {
                prediction_error = MatOps<REAL_t>::softmax_cross_entropy(activation.log_probs,
                                                                              answer_idx);
            }
            Mat<REAL_t> fact_selection_error(1,1);

            for (int i=0; i<activation.fact_gate_memory.size(); ++i) {
                bool supporting = in_vector(supporting_facts, i);
                auto partial_error = MatOps<REAL_t>::binary_cross_entropy(
                                            activation.fact_gate_memory[i],
                                            supporting ? 1.0 : 0.0);
                float coeff = supporting ? 1.0 : FLAGS_unsupporting_ratio;

                fact_selection_error = fact_selection_error + partial_error * coeff;
            }

            Mat<REAL_t> total_error;

            total_error = prediction_error
                        + fact_selection_error * FLAGS_fact_selection_lambda
                        + activation.word_memory_sum() * FLAGS_word_selection_sparsity;

            return total_error;
        }

        vector<string> predict(const vector<vector<string>>& facts,
                       const vector<string>& question) const {
            graph::NoBackprop nb;

            // Don't use dropout for validation.
            int word_idx = activate_story(facts, question, false).log_probs.argmax();

            return {vocab->index2word[word_idx]};
        }
};

/* PARAMETERS FOR TRAINING */
typedef LstmBabiModel<REAL_t> model_t;

/* TRAINING FUNCTIONS */
shared_ptr<model_t> model;

shared_ptr<model_t> best_model;
int best_model_epoch = 0;

vector<model_t> thread_models;
std::mutex solver_mutex;

// returns the errors;
double run_epoch(const vector<babi::Story>& dataset,
                 Solver::AbstractSolver<REAL_t>* solver) {
    const int NUM_THREADS = FLAGS_j;
    ThreadPool pool(NUM_THREADS);

    if (thread_models.size() == 0) {
        for (int i = 0; i < NUM_THREADS; ++i) {
            thread_models.push_back(model->shallow_copy());
        }
    }
    std::atomic<int> num_questions(0);

    vector<REAL_t> thread_error(NUM_THREADS, 0.0);

    auto random_order = utils::random_arange(dataset.size());
    vector<vector<int>> batches;
    for (int i=0; i<dataset.size(); i+=FLAGS_minibatch) {
        vector<int> batch;
        for (int j=i; j<std::min(i+FLAGS_minibatch, (int)dataset.size()); ++j) {
            batch.push_back(random_order[j]);
        }
        batches.push_back(batch);
    }

    for (auto& batch : batches) {
        pool.run([batch, &dataset, &num_questions, &thread_error, &solver]() {
            model_t& thread_model = thread_models[ThreadPool::get_thread_number()];
            auto params = thread_model.parameters();
            for (auto story_id: batch) {
                auto story = dataset[story_id];

                babi::StoryParser parser(&story);
                vector<vector<string>> facts_so_far;
                QA* qa;
                while (!parser.done()) {
                    std::tie(facts_so_far, qa) = parser.next();

                    uint answer_idx = thread_model.vocab->word2index.at(qa->answer[0]);

                    auto error      = thread_model.error(facts_so_far,
                                                          qa->question,
                                                          answer_idx,
                                                          qa->supporting_facts);
                    (error / FLAGS_minibatch).grad();

                    thread_error[ThreadPool::get_thread_number()] += error.w()(0,0);

                    num_questions += 1;

                    graph::backward();
                }
            }
            if (FLAGS_solver_mutex) {
                std::lock_guard<decltype(solver_mutex)> guard(solver_mutex);
                solver->step(params);
            } else {
                solver->step(params);
            }
        });
    }

    pool.wait_until_idle();

    return utils::vsum(thread_error) / num_questions;
}

void visualize_examples(const vector<babi::Story>& data, int num_examples) {
    while(num_examples--) {
        int example = rand()%data.size();
        int question_no = utils::randint(0, 5);
        babi::StoryParser parser(&data[example]);
        vector<vector<string>> facts_so_far;
        QA* qa;
        bool example_sent = false;
        while (!parser.done()) {
            std::tie(facts_so_far, qa) = parser.next();
            if (question_no == 0) {
                model->visualize_example(facts_so_far, qa->question, qa->answer);
                example_sent = true;
                break;
            }
            question_no--;
        }
        if (!example_sent) ++num_examples;
    }
}

void train(const vector<babi::Story>& data, float training_fraction = 0.8) {
    for (auto param: model->parameters()) {
        weights<REAL_t>::svd(weights<REAL_t>::gaussian(1.0))(param);
    }

    int training_size = (int)(training_fraction * data.size());
    std::vector<babi::Story> train(data.begin(), data.begin() + training_size);
    std::vector<babi::Story> validate(data.begin() + training_size, data.end());

    auto params = model->parameters();
    bool solver_reset_cache = false;

    Solver::AdaDelta<REAL_t> solver(params, 0.95, 1e-9, 5.0);
    //Solver::Adam<REAL_t> solver(params);
    solver.regc = 1e-5;

    double best_validation_accuracy = 0.0;
    best_model = std::make_shared<model_t>(*model, true, true);
    best_model_epoch = 0;

    int epoch = 0;
    int patience = 0;

    Throttled example_visualization;

    while (epoch < FLAGS_max_epochs && patience < FLAGS_patience) {
        auto training_error = run_epoch(train, &solver);

        if (!FLAGS_visualizer.empty()) {
            example_visualization.maybe_run(seconds(5), [&validate]() {
                visualize_examples(validate, 1);
            });
        }
        if (solver_reset_cache)
            solver.reset_caches(params);

        double validation_accuracy = babi::accuracy(
            validate,
            std::bind(&model_t::predict, model.get(), _1, _2),
            FLAGS_j
        );

        if (best_validation_accuracy < validation_accuracy) {
            std::cout << "NEW WORLD RECORD!" << std::endl;
            best_validation_accuracy = validation_accuracy;
            best_model = std::make_shared<model_t>(*model, true, true);
            best_model_epoch = epoch;
        }

        if (best_validation_accuracy > validation_accuracy) {
            patience += 1;
        } else {
            patience = 0;
        }

        std::cout << "Epoch: " << ++epoch << ", "
                  << "Training error: " << utils::bold << training_error << utils::reset_color << ", "
                  << "Validation accuracy: " << utils::bold << 100.0 * validation_accuracy << "%" << utils::reset_color
                  << " (patience: " << patience << "/" << FLAGS_patience << ")." << std::endl;
    }
}

void reset(const vector<babi::Story>& data) {
    thread_models.clear();
    auto vocab_vector = babi::vocab_from_data(data);
    model.reset();
    best_model.reset();
    model = std::make_shared<model_t>(make_shared<Vocab> (vocab_vector));
}

double benchmark_task(const std::string task) {
    std::cout << "Solving the most important problem in the world: " << task << std::endl;
    auto data = babi::Parser::training_data(task);
    reset(data);

    train(data, 0.8);
    std::cout << "[RESULTS] Best model was achieved at epoch " << best_model_epoch << " ." << std::endl;
    double accuracy = babi::task_accuracy(
        task,
        std::bind(&model_t::predict, model.get(), _1, _2),
        FLAGS_j);
    std::cout << "[RESULTS] Accuracy on " << task << " is " << 100.0 * accuracy << " ." << std::endl;
    return accuracy;
}

int main(int argc, char** argv) {
    // sane_crashes::activate();
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\nBabi!"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Eigen::setNbThreads(0);
    Eigen::initParallel();

    int increment = 0;
    if (!FLAGS_visualizer.empty()) {
        visualizer = make_shared<Visualizer>(FLAGS_visualizer);
    }

    std::cout << "Number of threads: " << FLAGS_j << (FLAGS_solver_mutex ? "(with solver mutex)" : "") << std::endl;
    std::cout << "Using " << (FLAGS_margin_loss ? "margin loss" : "cross entropy") << std::endl;

    benchmark_task(FLAGS_babi_problem);
}
