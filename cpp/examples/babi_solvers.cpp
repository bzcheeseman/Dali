#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
#include <gflags/gflags.h>

#include "dali/data_processing/babi.h"
#include "dali/core.h"
#include "dali/utils.h"

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


DEFINE_int32(j, 9, "Number of threads");

class MostCommonAnswer: public babi::Model {
    vector<string> most_common_answer;

    public:

        void train(const vector<babi::Story>& data) {
            std::unordered_map<string, int> freq;
            for (auto& story : data) {
                for(auto& item_ptr : story) {
                    if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                        // ignore
                    } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                        // When we are training we want to use dropout
                        freq[utils::join(qa->answer, "|")] += 1;
                    }
                }
            }
            int max_freq = 0;
            for (auto& kv: freq) {
                if (kv.second > max_freq) {
                    max_freq = kv.second;
                    most_common_answer = utils::split(kv.first,'|');
                }
            }
        }

        void new_story() {
        }

        void fact(const vector<string>& fact) {
        }
        vector<string> question(const vector<string>& question) {
            return most_common_answer;
        }
};

class RandomAnswer: public babi::Model {
    set<string> tokens;
    public:

        void train(const vector<babi::Story>& data) {

        }

        void new_story() {
            tokens.clear();
        }

        void fact(const vector<string>& fact) {
            for(const string& token: fact) {
                if (token.compare(".") == 0 || token.compare("?") == 0)
                    continue;
                tokens.insert(token);
            }
        }
        vector<string> question(const vector<string>& question) {
            string ans;
            int ans_idx = rand()%tokens.size();
            int current_idx = 0;
            for(auto& el: tokens) {
                if (current_idx == ans_idx)
                    ans = el;
                ++current_idx;
            }
            return {ans};
        }
};

template<typename T>
struct StoryActivation {
    Mat<T> word_fact_gate_memory_sum;
    Seq<Mat<T>> fact_gate_memory;
    Mat<T> log_probs;

    StoryActivation(Mat<T> word_fact_gate_memory_sum,
                    Seq<Mat<T>> fact_gate_memory,
                    Mat<T> log_probs) :
            word_fact_gate_memory_sum(word_fact_gate_memory_sum),
            fact_gate_memory(fact_gate_memory),
            log_probs(log_probs) {
    }
};

template<typename T>
class RecurrentGate : public AbstractLayer<T> {
    const int input_size;
    const int hidden_size;

    public:
        RNN<T> recurrent;
        Layer<T> gate_classifier;

        RecurrentGate(int input_size, int hidden_size) :
                input_size(input_size),
                hidden_size(hidden_size),
                recurrent(input_size, hidden_size),
                gate_classifier(hidden_size, 1) {
        }

        RecurrentGate (const RecurrentGate<T>& other, bool copy_w, bool copy_dw) :
            input_size(other.input_size),
            hidden_size(other.hidden_size),
            recurrent(other.recurrent, copy_w, copy_dw),
            gate_classifier(other.gate_classifier, copy_w, copy_dw) {
        }

        Mat<T> initial_states() const {
            return Mat<T>(hidden_size, 1, true);
        }

        tuple<Mat<T>,Mat<T>> activate(Mat<T> input, Mat<T> prev_hidden) const {
            auto next_hidden = recurrent.activate(input, prev_hidden).tanh();
            auto output = gate_classifier.activate(next_hidden).sigmoid();
            return make_tuple(output, next_hidden);
        }

        virtual vector<Mat<T>> parameters() const {
            std::vector<Mat<T>> ret;

            auto rnn_params = recurrent.parameters();
            auto gc_params = gate_classifier.parameters();
            ret.insert(ret.end(), rnn_params.begin(), rnn_params.end());
            ret.insert(ret.end(), gc_params.begin(), gc_params.end());
            return ret;
        }
};


template<typename T>
class LstmBabiModel {
    // MODEL PARAMS
    const vector<int>   TEXT_REPR_STACKS           =      {30, 30};
    const int           TEXT_REPR_EMBEDDINGS       =      30;
    const T             TEXT_REPR_DROPOUT          =      0.3;

    StackedShortcutLSTM<T> fact_model;
    Mat<T> fact_embeddings;

    StackedShortcutLSTM<T> question_model;
    Mat<T> question_representation_embeddings;


    const vector<int>   QUESTION_GATE_STACKS              =      {20, 20};
    const int           QUESTION_GATE_EMBEDDINGS          =      15;

    const T             QUESTION_GATE_DROPOUT             =      0.3;

    // input here is fact word embedding and question_fact_word_gate_model final hidden.
    const int           QUESTION_GATE_IN_FACT_WORDS       =
            TEXT_REPR_EMBEDDINGS + utils::vsum(QUESTION_GATE_STACKS);
    // input here is fact_model final hidden and question_fact_word_gate_model final hidden.
    const int           QUESTION_GATE_IN_FACTS            =
            utils::vsum(TEXT_REPR_STACKS) + utils::vsum(QUESTION_GATE_STACKS);

    const int           QUESTION_GATE_HIDDEN              =      50;

    StackedShortcutLSTM<T> question_fact_gate_model;
    Mat<T> question_fact_gate_embeddings;

    StackedShortcutLSTM<T> question_fact_word_gate_model;
    Mat<T> question_fact_word_gate_embeddings;

    RecurrentGate<T> fact_gate;
    RecurrentGate<T> fact_word_gate;


    const vector<int>   HL_STACKS                  =      {20,20,20,20}; //,20,20};
    const int           HL_INPUT_SIZE              =      utils::vsum(TEXT_REPR_STACKS);
    const T             HL_DROPOUT                 =      0.5;

    StackedShortcutLSTM<T> hl_model;

    Mat<T> please_start_prediction;

    const int           DECODER_INPUT              =      utils::vsum(HL_STACKS);
    const int           DECODER_OUTPUT; // gets initialized to vocabulary size in constructor
    Layer<T>            decoder;

    // TODO:
    // -> we are mostly concerned with gates being on for positive facts.
    //    some false positives are acceptable.
    // -> second order (between question and fact) relation for fact word gating
    // -> consider quadratic form]
    // -> add multiple answers

    shared_ptr<Vocab> vocab;

    public:
        vector<Mat<T>> parameters() {
            vector<Mat<T>> res;
            for (auto model: std::vector<AbstractLayer<T>*>({
                                &question_fact_gate_model,
                                &question_fact_word_gate_model,
                                &fact_gate,
                                &fact_word_gate,
                                &question_model,
                                &fact_model,
                                &hl_model,
                                &decoder })) {
                auto params = model->parameters();
                res.insert(res.end(), params.begin(), params.end());
            }
            for (auto& matrix: { question_fact_gate_embeddings,
                                 question_fact_word_gate_embeddings,
                                 fact_embeddings,
                                 question_representation_embeddings,
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
                question_fact_gate_model(model.question_fact_gate_model, copy_w, copy_dw),
                question_fact_word_gate_model(
                        model.question_fact_word_gate_model, copy_w, copy_dw),
                fact_gate(model.fact_gate, copy_w, copy_dw),
                fact_word_gate(model.fact_word_gate, copy_w, copy_dw),
                question_model(model.question_model, copy_w, copy_dw),
                fact_model(model.fact_model, copy_w, copy_dw),
                hl_model(model.hl_model, copy_w, copy_dw),
                DECODER_OUTPUT(model.DECODER_OUTPUT),
                decoder(model.decoder, copy_w, copy_dw),
                // paramters begin here
                question_fact_gate_embeddings(
                        model.question_fact_gate_embeddings, copy_w, copy_dw),
                question_fact_word_gate_embeddings(
                        model.question_fact_word_gate_embeddings, copy_w, copy_dw),
                fact_embeddings(model.fact_embeddings, copy_w, copy_dw),
                question_representation_embeddings(
                        model.question_representation_embeddings, copy_w, copy_dw),
                please_start_prediction(model.please_start_prediction, copy_w, copy_dw) {
            vocab = model.vocab;
        }

        LstmBabiModel(shared_ptr<Vocab> vocabulary) :
                question_fact_gate_model(QUESTION_GATE_EMBEDDINGS, QUESTION_GATE_STACKS),
                question_fact_word_gate_model(QUESTION_GATE_EMBEDDINGS, QUESTION_GATE_STACKS),
                fact_gate(QUESTION_GATE_IN_FACTS, QUESTION_GATE_HIDDEN),
                fact_word_gate(QUESTION_GATE_IN_FACT_WORDS, QUESTION_GATE_HIDDEN),
                question_model(TEXT_REPR_EMBEDDINGS, TEXT_REPR_STACKS),
                fact_model(TEXT_REPR_EMBEDDINGS, TEXT_REPR_STACKS),
                hl_model(HL_INPUT_SIZE, HL_STACKS),
                DECODER_OUTPUT(vocabulary->word2index.size()),
                decoder(DECODER_INPUT, vocabulary->word2index.size()) {

            vocab = vocabulary;
            size_t n_words = vocab->index2word.size();

            question_fact_gate_embeddings = Mat<T>(n_words, QUESTION_GATE_EMBEDDINGS);
            question_fact_word_gate_embeddings = Mat<T>(n_words, QUESTION_GATE_EMBEDDINGS);
            fact_embeddings = Mat<T>(n_words, TEXT_REPR_EMBEDDINGS);
            question_representation_embeddings = Mat<T>(n_words, TEXT_REPR_EMBEDDINGS);
            please_start_prediction = Mat<T>(HL_INPUT_SIZE, 1);

        }


        Seq<Mat<T>> get_embeddings(const vector<string>& words,
                                   Mat<T> embeddings) {
            Seq<Mat<T>> seq;
            for (auto& word: words) {
                auto question_idx = vocab->word2index.at(word);
                auto embedding = embeddings.row_pluck(question_idx);
                seq.push_back(embedding);
                // We don't need explicitly start prediction token because
                // each question is guaranteed to end with "?" token.
            }
            return seq;
        }

        Seq<Mat<T>> gate_memory(Seq<Mat<T>> seq,
                                const RecurrentGate<T>& gate,
                                Mat<T> gate_input) {
            Seq<Mat<T>> memory_seq;
            // By default initialized to zeros.
            auto prev_hidden = gate.initial_states();
            Mat<T> gate_activation;
            for (auto& embedding : seq) {
                // out_state: next_hidden, output
                std:tie(gate_activation, prev_hidden) =
                        gate.activate(MatOps<T>::vstack(embedding, gate_input), prev_hidden);
                // memory - gate activation - how much of that embedding do we keep.
                memory_seq.push_back(gate_activation);
            }
            return memory_seq;
        }

        Seq<Mat<T>> apply_gate(Seq<Mat<T>> memory,
                               Seq<Mat<T>> seq) {
            assert(memory.size() == seq.size());
            Seq<Mat<T>> gated_seq;
            for(int i=0; i < memory.size(); ++i) {
                gated_seq.push_back(seq[i].eltmul_broadcast_rowwise(memory[i]));
            }
            return gated_seq;
        }


        Mat<T> lstm_final_activation(const Seq<Mat<T>>& embeddings,
                                     const StackedShortcutLSTM<T>& model,
                                     T dropout_value) {
            vector<Mat<T>> memory, hidden;
            std::tie(memory, hidden) = model.activate_sequence(model.initial_states(),
                                                     embeddings,
                                                     dropout_value);
            // out_state.second corresponds to LSTM hidden (as opposed to memory).
            return MatOps<T>::vstack(hidden);
        }

        StoryActivation<T> activate_story(const vector<vector<string>>& facts,
                                          const vector<string>& question,
                                          bool use_dropout) {
            auto fact_word_gate_hidden = lstm_final_activation(
                    get_embeddings(reversed(question), question_fact_word_gate_embeddings),
                    question_fact_word_gate_model,
                    use_dropout ? QUESTION_GATE_DROPOUT : 0.0);

            auto fact_gate_hidden = lstm_final_activation(
                    get_embeddings(reversed(question), question_fact_gate_embeddings),
                    question_fact_gate_model,
                    use_dropout ? QUESTION_GATE_DROPOUT : 0.0);

            auto question_representation = lstm_final_activation(
                    get_embeddings(reversed(question), question_representation_embeddings),
                    question_model,
                    use_dropout ? TEXT_REPR_DROPOUT : 0.0);

            Seq<Mat<T>> fact_representations;
            Mat<T> word_fact_gate_memory_sum(1,1);

            for (auto& fact: facts) {
                auto this_fact_embeddings = get_embeddings(reversed(fact), fact_embeddings);
                auto fact_word_gate_memory = gate_memory(this_fact_embeddings,
                                                         fact_word_gate,
                                                         fact_word_gate_hidden);
                for (auto mem: fact_word_gate_memory) {
                    word_fact_gate_memory_sum = word_fact_gate_memory_sum + mem;
                }

                auto gated_embeddings = apply_gate(fact_word_gate_memory, this_fact_embeddings);

                auto fact_repr = lstm_final_activation(gated_embeddings,
                                                       fact_model,
                                                       use_dropout ? TEXT_REPR_DROPOUT : 0.0);
                fact_representations.push_back(fact_repr);
            }

            auto fact_gate_memory = gate_memory(fact_representations, fact_gate, fact_gate_hidden);

            auto gated_facts = apply_gate(fact_gate_memory, fact_representations);
            // There is probably a better way
            Seq<Mat<T>> hl_input;
            hl_input.push_back(question_representation);
            hl_input.insert(hl_input.end(), gated_facts.rbegin(), gated_facts.rend());
            hl_input.push_back(please_start_prediction);

            auto hl_hidden = lstm_final_activation(hl_input,
                                                   hl_model,
                                                   use_dropout ? HL_DROPOUT : 0.0);

            auto log_probs = decoder.activate(hl_hidden);

            return StoryActivation<T>(word_fact_gate_memory_sum,
                                      fact_gate_memory,
                                      log_probs);
        }

        vector<Mat<T>> errors(const vector<Fact*>& facts,
                              QA* qa,
                              bool use_dropout) {
            vector<vector<string>> facts_as_strings;

            for (auto fact_ptr: facts)
                facts_as_strings.push_back(fact_ptr->fact);

            auto activation = activate_story(facts_as_strings, qa->question, use_dropout);

            uint answer_idx = vocab->word2index.at(qa->answer[0]);

            auto prediction_error = MatOps<T>::softmax_cross_entropy(activation.log_probs,
                                                                     answer_idx);

            Mat<T> fact_selection_error(1,1);

            assert(facts.size() == activation.fact_gate_memory.size());
            for (int i=0; i<facts.size(); ++i) {
                auto partial_error = MatOps<T>::binary_cross_entropy(
                                            activation.fact_gate_memory[i],
                                            in_vector(qa->supporting_facts, i) ? 1.0 : 0.0);
                fact_selection_error = fact_selection_error + partial_error;
            }
            int num_words_in_facts = 0;
            for (Fact* fact: facts) {
                num_words_in_facts += fact->fact.size();
            }

            return { prediction_error,
                     fact_selection_error,
                     activation.word_fact_gate_memory_sum };
        }
};

template<typename T>
class LstmBabiModelRunner: public babi::Model {
    // TRAINING_PROCEDURE_PARAMS
    const float TRAINING_FRAC = 0.8;
    const float MINIMUM_IMPROVEMENT = 0.0001; // good one: 0.003
    const double LONG_TERM_VALIDATION = 0.02;
    const double SHORT_TERM_VALIDATION = 0.1;

    // gates overfit easily
    const int GATES_PATIENCE = 10;
    // prediction haz dropout.
    const int PREDICTION_PATIENCE = 100;

    const T FACT_SELECTION_LAMBDA_MAX = 0.2;
    const T FACT_WORD_SELECTION_LAMBDA_MAX = 0.0001;

    const int BAKING_EPOCHS = 1000;

    vector<string> data_to_vocab(const vector<babi::Story>& data) {
        set<string> vocab_set;
        for (auto& story : data) {
            for(auto& item_ptr : story) {
                if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                    vocab_set.insert(f->fact.begin(), f->fact.end());
                } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                    vocab_set.insert(qa->question.begin(), qa->question.end());
                    vocab_set.insert(qa->answer.begin(), qa->answer.end());
                }
            }
        }
        vector<string> vocab_as_vector;
        vocab_as_vector.insert(vocab_as_vector.end(), vocab_set.begin(), vocab_set.end());
        return vocab_as_vector;
    }

    shared_ptr<Vocab> vocab;
    shared_ptr<LstmBabiModel<T>> model;

    vector<LstmBabiModel<T>> thread_models;

    int epoch;

    enum training_mode_t {
        GATES = 1,
        PREDICTION = 2
    };


    training_mode_t training_mode;

    Mat<T> error_function(vector<Mat<T>> errors) {
        if (training_mode == GATES) {
            return errors[1];
        } else if (training_mode == PREDICTION) {
            return errors[0] + errors[1] * 0.1 + errors[2] * 0.00001;
        }
    }


    public:
        vector<double> compute_errors(const vector<babi::Story>& dataset,
                                      Solver::Adam<T>* solver,
                                      bool training) {
            const int NUM_THREADS = FLAGS_j;
            ThreadPool pool(NUM_THREADS);

            if (thread_models.size() == 0) {
                for (int i = 0; i < NUM_THREADS; ++i) {
                    thread_models.push_back(model->shallow_copy());
                }
            }
            std::atomic<int> num_questions(0);

            vector<vector<double>> thread_error;
            for (int thread=0; thread<NUM_THREADS; ++thread) {
                thread_error.emplace_back(vector<double>(3, 0.0));
            }


            const int BATCH_SIZE = std::max( (int)dataset.size() / (2*NUM_THREADS), 2);

            auto random_order = utils::random_arange(dataset.size());
            vector<vector<int>> batches;
            for (int i=0; i<dataset.size(); i+=BATCH_SIZE) {
                vector<int> batch;
                for (int j=i; j<std::min(i+BATCH_SIZE, (int)dataset.size()); ++j) {
                    batch.push_back(random_order[j]);
                }
                batches.push_back(batch);
            }

            for (auto& batch : batches) {
                pool.run([this, batch, &dataset, training, &num_questions, &thread_error, &solver]() {
                    LstmBabiModel<T>& thread_model = thread_models[ThreadPool::get_thread_number()];
                    auto params = thread_model.parameters();
                    for (auto story_id: batch) {
                        auto story = dataset[story_id];

                        babi::StoryParser parser(&story);
                        vector<Fact*> facts_so_far;
                        QA* qa;
                        while (!parser.done()) {
                            std::tie(facts_so_far, qa) = parser.next();
                            // When we are training we want to do backprop
                            graph::NoBackprop nb(!training);
                            // When we are training we want to use dropout
                            auto errors = thread_model.errors(facts_so_far, qa, training);
                            for (int i=0; i<3; ++i)
                                thread_error[ThreadPool::get_thread_number()][i] += errors[i].w()(0,0);

                            auto total_error = error_function(errors);
                            total_error.grad();

                            num_questions += 1;
                            if (training)
                                graph::backward();
                         }
                    }
                    if (training)
                        solver->step(params, 0.001);
                });
            }

            pool.wait_until_idle();

            vector<double> total_error(3, 0.0);
            for (int err=0; err<3; ++err) {
                for (int i=0; i<NUM_THREADS; ++i)
                    total_error[err] += thread_error[i][err];
                total_error[err] /= num_questions;
            }
            return total_error;
        }

        void train(const vector<babi::Story>& data) {
            auto vocab_vector = data_to_vocab(data);
            vocab = make_shared<Vocab> (vocab_vector);

            model = std::make_shared<LstmBabiModel<T>>(vocab);

            int training_size = (int)(TRAINING_FRAC * data.size());
            std::vector<babi::Story> train(data.begin(), data.begin() + training_size);
            std::vector<babi::Story> validation(data.begin() + training_size, data.end());

            epoch = 0;

            auto params = model->parameters();

            for (training_mode_t cur_training_mode : { GATES, PREDICTION }) {
                training_mode = cur_training_mode;

                // Solver::AdaDelta<T> solver(params, 0.95, 1e-9, 5.0);
                Solver::Adam<T> solver(params);
                LSTV training(SHORT_TERM_VALIDATION, LONG_TERM_VALIDATION,
                        (cur_training_mode == GATES ? GATES_PATIENCE : PREDICTION_PATIENCE));

                while (true) {
                    auto training_errors = compute_errors(train, &solver, true);
                    auto validation_errors = compute_errors(validation, &solver, false);

                    std::cout << "Epoch " << ++epoch << std::endl;
                    std::cout << (cur_training_mode == GATES ? "Optimizing gates" : "Optimizing prediction") << std::endl;
                    std::cout << "Errors(prob_answer, fact_select, words_sparsity): " << std::endl
                              << "VALIDATION: " << validation_errors[0] << " "
                                                << validation_errors[1] << " "
                                                << validation_errors[2] << std::endl
                              << "TRAINING: " << training_errors[0] << " "
                                              << training_errors[1] << " "
                                              << training_errors[2] << std::endl;
                    if (cur_training_mode == GATES &&
                            training.update(validation_errors[1])) {
                        break;
                    } else if (cur_training_mode == PREDICTION  &&
                            training.update(validation_errors[0])) {
                        break;
                    }
                    training.report();
                }
            }
        }

        vector<vector<string>> story_so_far;

        void new_story() {
            story_so_far.clear();
        }

        void fact(const vector<string>& fact) {
            story_so_far.push_back(fact);
        }

        vector<string> question(const vector<string>& question) {
            graph::NoBackprop nb;
            // Don't use dropout for validation.
            int word_idx = argmax(model->activate_story(story_so_far, question, false).log_probs);

            return {vocab->index2word[word_idx]};
        }
};


int main(int argc, char** argv) {
    sane_crashes::activate();
    GFLAGS_NAMESPACE::SetUsageMessage(
        "\nBabi!"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Eigen::initParallel();

    // babi::benchmark_task<LstmBabiModelRunner<double>>("qa1_single-supporting-fact");
    // babi::benchmark_task<LstmBabiModelRunner<double>>("qa4_two-arg-relations");
    // babi::benchmark_task<LstmBabiModelRunner<double>>("qa2_two-supporting-facts");
    // babi::benchmark_task<LstmBabiModelRunner<double>>("qa3_three-supporting-facts");
    babi::benchmark<LstmBabiModelRunner<double>>();
}
