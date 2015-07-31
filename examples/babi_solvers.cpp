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
#include "dali/execution/BeamSearch.h"
#include "dali/visualizer/visualizer.h"

using namespace std::placeholders;

using beam_search::BeamSearchResult;
using std::chrono::seconds;
using std::get;
using std::make_shared;
using std::make_tuple;
using std::set;
using std::shared_ptr;
using std::string;
using std::tie;
using std::to_string;
using std::tuple;
using std::vector;
using utils::assert2;
using utils::in_vector;
using utils::MS;
using utils::reversed;
using utils::Timer;
using utils::Vocab;

DEFINE_int32(j,                        1,      "Number of threads");
DEFINE_int32(minibatch,                25,     "How many stories to put in a single batch.");
DEFINE_int32(max_epochs,               2000,   "maximum number of epochs for training");
DEFINE_int32(patience,                 200,    "after <patience> epochs of performance decrease we stop.");
DEFINE_int32(beam_width,               5,      "width of the beam for prediction.");
DEFINE_bool(svd_init,                  false,  "Initialize weights using SVD?");
// generic lstm properties
DEFINE_bool(lstm_shortcut,             true,   "Should shortcut be used in LSTMs");
DEFINE_bool(lstm_feed_mry,             true,   "Should memory be fed to gates in LSTMs");
// high level
DEFINE_int32(hl_hidden,                20,     "What hidden size to use for high level.");
DEFINE_int32(hl_stack,                 4,      "How high is the LSTM for high level.");
DEFINE_double(hl_dropout,              0.7,    "How much dropout to use at high level.");
// answer model
DEFINE_int32(answer_input,             10,     "Embeddings for answer");
DEFINE_double(answer_dropout,          0.3,    "How much dropout to use at answer generation model.");
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
Vocab vocab;

template<typename T>
struct StoryActivation {
    typedef typename StackedLSTM<T>::state_t lstm_state_t;
    lstm_state_t lstm_state;
    vector<Mat<T>> fact_gate_memory;
    vector<vector<Mat<T>>> word_gate_memory;

    StoryActivation(lstm_state_t lstm_state,
                    vector<Mat<T>> fact_gate_memory,
                    vector<vector<Mat<T>>> word_gate_memory) :
            lstm_state(lstm_state),
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
    const uint MAX_ANSWER_LENGTH = 5;
    typedef typename StackedLSTM<T>::state_t lstm_state_t;

    StackedLSTM<T> fact_model;
    Mat<T> fact_embeddings;

    StackedLSTM<T> question_model;
    Mat<T> question_embeddings;

    SecondOrderCombinator<T> word_gate;
    Mat<T> word_gate_embeddings;

    SecondOrderCombinator<T> fact_gate;
    Mat<T> fact_gate_embeddings;

    StackedLSTM<T> hl_model;

    StackedLSTM<T> answer_model;
    Mat<T> answer_embeddings;

    Mat<T> please_start_prediction;

    Layer<T>            decoder;

    public:
        vector<Mat<T>> parameters() {
            vector<Mat<T>> res;
            for (auto model: std::vector<AbstractLayer<T>*>({
                                &fact_model,
                                &question_model,
                                &word_gate,
                                &fact_gate,
                                &hl_model,
                                &answer_model,
                                &decoder })) {
                auto params = model->parameters();
                res.insert(res.end(), params.begin(), params.end());
            }
            for (auto& matrix: { fact_embeddings,
                                 question_embeddings,
                                 word_gate_embeddings,
                                 fact_gate_embeddings,
                                 answer_embeddings,
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
                answer_model(model.answer_model, copy_w, copy_dw),
                decoder(model.decoder, copy_w, copy_dw),
                // parameters begin here
                fact_embeddings(model.fact_embeddings, copy_w, copy_dw),
                question_embeddings(model.question_embeddings, copy_w, copy_dw),
                word_gate_embeddings(model.word_gate_embeddings, copy_w, copy_dw),
                fact_gate_embeddings(model.fact_gate_embeddings, copy_w, copy_dw),
                answer_embeddings(model.answer_embeddings, copy_w, copy_dw),
                please_start_prediction(model.please_start_prediction, copy_w, copy_dw) {
        }

        LstmBabiModel() :
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
                answer_model(FLAGS_answer_input,
                         { FLAGS_hl_stack * FLAGS_hl_hidden },
                         FLAGS_lstm_shortcut,
                         FLAGS_lstm_feed_mry),
                decoder(utils::vsum(vector<int>(FLAGS_hl_stack, FLAGS_hl_hidden)),
                        vocab.size()) {

            size_t n_words = vocab.size();
            size_t hl_model_input = utils::vsum(vector<int>(FLAGS_text_repr_stack, FLAGS_text_repr_hidden));

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
            answer_embeddings =
                    Mat<T>(n_words, FLAGS_answer_input,
                           weights<T>::uniform(1.0/hl_model_input));
            please_start_prediction =
                    Mat<T>(1, FLAGS_answer_input, weights<T>::uniform(1.0));
        }

        vector<Mat<T>> get_embeddings(const vector<string>& words,
                                      Mat<T> embedding_matrix) const {
            vector<Mat<T>> seq;
            for (auto& word: words) {
                auto question_idx = vocab.word2index.at(word);
                auto embedding = embedding_matrix[question_idx];
                seq.push_back(embedding);
                // We don't need explicitly start prediction token because
                // each question is guaranteed to end with "?" token.
            }
            return seq;
        }

        static vector<Mat<T>> apply_gate(vector<Mat<T>> memory,
                                  vector<Mat<T>> seq) {
            assert(memory.size() == seq.size());
            vector<Mat<T>> gated_seq;
            for(int i=0; i < memory.size(); ++i) {
                gated_seq.push_back(seq[i].eltmul_broadcast_colwise(memory[i]));

            }
            return gated_seq;
        }

        static Mat<T> state_to_hidden(lstm_state_t state) {
            return MatOps<T>::hstack(LSTM<T>::activation_t::hiddens(state));
        }

        static lstm_state_t flatten_state(lstm_state_t state) {
            auto hidden = MatOps<T>::hstack(LSTM<T>::activation_t::hiddens(state));
            auto memory = MatOps<T>::hstack(LSTM<T>::activation_t::memories(state));
            return {typename LSTM<T>::activation_t(memory, hidden)};
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
                        word_gate.activate(word_gate_hidden_question, gate_hidden_fact_word).sigmoid().sum().sigmoid()
                    );
                }


                word_gate_memories.push_back(word_gate_memory);

                auto gated_embeddings = apply_gate(
                    word_gate_memory,
                    get_embeddings(reversed(fact), fact_embeddings)
                );

                auto fact_repr = state_to_hidden(fact_model.activate_sequence(
                        fact_model.initial_states(),
                        gated_embeddings,
                        use_dropout ? FLAGS_text_repr_dropout : 0.0));

                fact_representations.push_back(fact_repr);
            }

            auto fact_gate_embeddings_question = get_embeddings(question, fact_gate_embeddings);
            auto fact_gate_hidden_question = MatOps<T>::add(fact_gate_embeddings_question) /
                                             (T)fact_gate_embeddings_question.size();

            vector<Mat<T>> fact_gate_memory;
            for (auto& fact_representation: fact_representations) {
                fact_gate_memory.push_back(
                    fact_gate.activate(fact_gate_hidden_question, fact_representation).sigmoid().sum().sigmoid()
                );
            }
            auto gated_facts = apply_gate(fact_gate_memory, fact_representations);
            auto question_hidden = state_to_hidden(question_model.activate_sequence(
                    question_model.initial_states(),
                    get_embeddings(reversed(question), question_embeddings),
                    use_dropout ? FLAGS_text_repr_dropout : 0.0));

            // There is probably a better way
            vector<Mat<T>> hl_input;
            hl_input.push_back(question_hidden);
            hl_input.insert(hl_input.end(), gated_facts.begin(), gated_facts.end());

            auto hl_final_state = hl_model.activate_sequence(
                    hl_model.initial_states(),
                    hl_input,
                    use_dropout ? FLAGS_hl_dropout : 0.0);

            return StoryActivation<T>(hl_final_state,
                                      fact_gate_memory,
                                      word_gate_memories);
        }

        Mat<T> error(const vector<vector<string>>& facts,
                     const vector<string>& question,
                     const vector<string>& answer,
                     vector<uint> supporting_facts) const {
            auto activation = activate_story(facts, question, true);

            auto current_state = answer_model.activate(
                        flatten_state(activation.lstm_state), please_start_prediction, FLAGS_answer_dropout);

            Mat<REAL_t> prediction_error(1,1);

            vector<Mat<T>> prediction_errors;

            auto answer_idxes = vocab.encode(answer, true);


            for (auto& word_idx: answer_idxes) { // for word idxes with end token
                Mat<T> partial_error;
                if (FLAGS_margin_loss) {
                    // We estimated eprically that margin loss scales as about 3.0
                    // cross entropy. We want to put the roughly in the same bucket, so
                    // so that improtance of gate errors and sparsity has more or less
                    // the same effect for both types of errors.

                    auto scores = decoder.activate(state_to_hidden(current_state));
                    partial_error =
                            MatOps<REAL_t>::margin_loss_rowwise(scores, word_idx, FLAGS_margin);
                } else {
                    auto log_probs = decoder.activate(state_to_hidden(current_state));
                    partial_error =
                            MatOps<REAL_t>::softmax_cross_entropy_rowwise(log_probs, word_idx);

                }

                prediction_errors.push_back(partial_error);
                current_state = answer_model.activate(
                        current_state, answer_embeddings[word_idx], FLAGS_answer_dropout);
            }

            Mat<REAL_t> fact_selection_error(1,1);

            for (uint i=0; i < activation.fact_gate_memory.size(); ++i) {
                bool supporting = in_vector(supporting_facts, i);
                auto partial_error = MatOps<REAL_t>::binary_cross_entropy(
                                            activation.fact_gate_memory[i],
                                            supporting ? 1.0 : 0.0);
                float coeff = supporting ? 1.0 : FLAGS_unsupporting_ratio;

                fact_selection_error += partial_error * coeff;
            }

            Mat<REAL_t> total_error;

            total_error = MatOps<T>::add(prediction_errors)
                        + fact_selection_error * FLAGS_fact_selection_lambda
                        + activation.word_memory_sum() * FLAGS_word_selection_sparsity;

            return total_error;
        }

        vector<BeamSearchResult<T, lstm_state_t>> my_beam_search(StoryActivation<T>& activation) const {
            // candidate_t = int, state type = lstm_state_t,
             auto initial_state = answer_model.activate(
                        flatten_state(activation.lstm_state), please_start_prediction, FLAGS_answer_dropout);

            auto candidate_scores = [this](lstm_state_t state) -> Mat<T> {
                auto scores = decoder.activate(LstmBabiModel<T>::state_to_hidden(state));
                return MatOps<T>::softmax_rowwise(scores).log();
            };
            auto make_choice = [this](lstm_state_t state, uint candidate) -> lstm_state_t {
                return answer_model.activate(state, answer_embeddings[candidate]);
            };

            auto beam_search_results = beam_search::beam_search<T, lstm_state_t>(
                    initial_state,
                    FLAGS_beam_width,
                    candidate_scores,
                    make_choice,
                    vocab.word2index.at(utils::end_symbol),
                    MAX_ANSWER_LENGTH);

            return beam_search_results;
        }

        vector<string> predict(const vector<vector<string>>& facts,
                       const vector<string>& question) const {
            graph::NoBackprop nb;

            auto activation = activate_story(facts, question, false);
            auto result_as_idxes = my_beam_search(activation)[0].solution;
            auto result = vocab.decode(&result_as_idxes, true);

            return result;
        }

        void visualize_example(const vector<vector<string>>& facts,
                               const vector<string>& question,
                               const vector<string>& correct_answer) const {
            graph::NoBackprop nb;
            auto activation = activate_story(facts, question, false);
            auto beam_search_results = my_beam_search(activation);

            shared_ptr<visualizable::FiniteDistribution<T>> vdistribution;
            std::vector<REAL_t> scores_as_vec;
            std::vector<string> beam_search_results_solutions;
            for (auto& result: beam_search_results) {
                scores_as_vec.push_back(
                        FLAGS_margin_loss ? result.score : std::exp(result.score));
                auto answer_str = vocab.decode(&result.solution, true);
                beam_search_results_solutions.push_back(utils::join(answer_str, " "));
            }
            auto distribution_as_vec =
                    FLAGS_margin_loss ? utils::normalize_weights(scores_as_vec) : scores_as_vec;
            vdistribution = make_shared<visualizable::FiniteDistribution<T>>(
                    distribution_as_vec, scores_as_vec, beam_search_results_solutions, 5);

            vector<REAL_t> facts_weights;
            vector<std::shared_ptr<visualizable::Sentence<T>>> facts_sentences;
            for (int fidx=0; fidx < facts.size(); ++fidx) {
                auto vfact = make_shared<visualizable::Sentence<T>>(facts[fidx]);

                vector<REAL_t> words_weights;

                for (Mat<T> weight: activation.word_gate_memory[fidx]) {
                    words_weights.push_back(weight.w(0,0));
                }
                vfact->set_weights(words_weights);
                facts_sentences.push_back(vfact);
                facts_weights.push_back(activation.fact_gate_memory[fidx].w(0,0));
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
};

/* PARAMETERS FOR TRAINING */
typedef LstmBabiModel<REAL_t> model_t;

/* TRAINING FUNCTIONS */
shared_ptr<model_t> model;

shared_ptr<model_t> best_model;
int best_model_epoch = 0;

// returns the errors;
double run_epoch(const vector<babi::Story<string>>& dataset,
                 Solver::AbstractSolver<REAL_t>* solver) {
    const int NUM_THREADS = FLAGS_j;
    ThreadPool pool(NUM_THREADS);

    vector<model_t> thread_models;
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_models.push_back(model->shallow_copy());
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
        pool.run([batch, &dataset, &num_questions, &thread_error, &solver, &thread_models]() {
            model_t& thread_model = thread_models[ThreadPool::get_thread_number()];
            auto params = thread_model.parameters();
            for (auto story_id: batch) {
                auto& story = dataset[story_id];

                for (int qa_idx = 0; qa_idx < story.size(); ++qa_idx) {
                    auto qa = story.get(qa_idx);
                    auto error = thread_model.error(qa.facts,
                                                    qa.question,
                                                    qa.answer,
                                                    qa.supporting_facts);
                    error.grad();
                    thread_error[ThreadPool::get_thread_number()] += error.w(0);
                    num_questions += 1;
                    graph::backward();
                }
            }
            solver->step(params);
        });
    }

    pool.wait_until_idle();

    return utils::vsum(thread_error) / num_questions;
}

double accuracy(const vector<babi::Story<string>> dataset,
                std::shared_ptr<model_t> model) {
    graph::NoBackprop nb;
    int num_correct   = 0;
    int num_questions = 0;
    for (auto& story: dataset) {
        for (int qa_idx = 0; qa_idx < story.size(); ++ qa_idx) {
            auto qa = story.get(qa_idx);

            auto predicted_answer = model->predict(qa.facts, qa.question);
            if (predicted_answer == qa.answer)
                ++num_correct;
            ++num_questions;
        }
    }
    return (double)num_correct/num_questions;
}


void visualize_examples(const vector<babi::Story<string>>& data, int num_examples) {
    while(num_examples--) {
        int example_idx = rand()%data.size();
        auto& story = data[example_idx];
        int question_no = utils::randint(0, story.size() - 1);
        auto qa = story.get(question_no);
        model->visualize_example(qa.facts, qa.question, qa.answer);
    }
}

void train(const vector<babi::Story<string>>& train,
           const vector<babi::Story<string>>& validate) {
    if (FLAGS_svd_init) {
        for (auto param: model->parameters()) {
            weights<REAL_t>::svd(weights<REAL_t>::gaussian(1.0))(param.w());
        }
    }

    auto params = model->parameters();

    Solver::AdaDelta<REAL_t> solver(params); // , 0.1, 0.0001);

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

        double validation_accuracy = accuracy(validate, model);

        bool best_so_far = best_validation_accuracy < validation_accuracy;

        if (best_so_far) {
            best_validation_accuracy = validation_accuracy;
            best_model = std::make_shared<model_t>(*model, true, true);
            best_model_epoch = epoch;
        }

        if (epoch % 2 == 0) {
            solver.reset_caches(params);
        }

        if (best_validation_accuracy > validation_accuracy) {
            patience += 1;
        } else {
            patience = 0;
        }

        std::cout << (best_so_far ? "*" : "")
                  << "Epoch: " << ++epoch << ", "
                  << "Training error: " << utils::bold << training_error << utils::reset_color << ", "
                  << "Validation accuracy: " << utils::bold << 100.0 * validation_accuracy << "%" << utils::reset_color
                  << " (patience: " << patience << "/" << FLAGS_patience << ")." << std::endl;
    }
}


double benchmark_task(const std::string task) {
    std::cout << "Solving the most important problem in the world: " << task << std::endl;
    auto str_training_data = babi::dataset(FLAGS_babi_problem, "train");
    auto str_testing_data = babi::dataset(FLAGS_babi_problem, "test");
    /*std::cout << "EXAMPLE STORY" << std::endl;
    for(auto& babi_item: training_data[0]) {
        std::cout << *babi_item << std::endl;
    }*/

    vector<babi::Story<uint>> encoded_data;
    encoded_data = encode_dataset(str_training_data, &vocab);

    model.reset();
    best_model.reset();
    model = std::make_shared<model_t>();

    const float validation_fraction = 0.2;

    std::random_shuffle(str_training_data.begin(), str_training_data.end());
    int training_size = (int)((1.0 - validation_fraction) * str_training_data.size());
    vector<babi::Story<string>> train_data(str_training_data.begin(), str_training_data.begin() + training_size);
    vector<babi::Story<string>> validate_data(str_training_data.begin() + training_size, str_training_data.end());

    train(train_data, validate_data);
    std::cout << "[RESULTS] Best model was achieved at epoch " << best_model_epoch << " ." << std::endl;
    model = best_model;
    double test_accuracy = accuracy(str_testing_data, best_model);
    std::cout << "[RESULTS] Accuracy on " << task << " is " << 100.0 * test_accuracy << " ." << std::endl;
    return test_accuracy;
}

int main(int argc, char** argv) {
    GFLAGS_NAMESPACE::SetUsageMessage("\nBabi!");
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    int increment = 0;
    if (!FLAGS_visualizer.empty())
        visualizer = make_shared<Visualizer>(FLAGS_visualizer);

    std::cout << "Number of threads: " << FLAGS_j << std::endl;
    std::cout << "Using " << (FLAGS_margin_loss ? "margin loss" : "cross entropy") << std::endl;

    if (FLAGS_babi_problem == "all") {
        std::cout << "Running a benchmark for all babi problems" << std::endl;
        for (auto& task: babi::tasks())
            benchmark_task(task);
    } else {
        benchmark_task(FLAGS_babi_problem);
    }
}
