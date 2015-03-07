#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
#include <gperftools/profiler.h>
#include <gflags/gflags.h>

#include "core/babi.h"
#include "core/CrossEntropy.h"
#include "core/Reporting.h"
#include "core/SaneCrashes.h"
#include "core/Seq.h"
#include "core/Solver.h"
#include "core/StackedModel.h"

using babi::Fact;
using babi::Item;
using babi::QA;
using babi::Story;
using std::make_shared;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;
using utils::Timer;
using utils::Vocab;
using utils::reversed;

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
class LstmBabiModel {
    typedef Mat<T> mat;
    typedef shared_ptr<mat> shared_mat;
    typedef Graph<T> graph_t;
    typedef StackedModel<T> model_t;

    // MODEL PARAMS
    const vector<int>   TEXT_REPR_STACKS           =      {30, 30};
    const int           TEXT_REPR_EMBEDDINGS       =      30;
    const T             TEXT_REPR_DROPOUT          =      0.3;

    StackedShortcutLSTM<T> fact_model;
    shared_mat fact_embeddings;

    StackedShortcutLSTM<T> question_model;
    shared_mat question_representation_embeddings;


    const vector<int>   QUESTION_GATE_STACKS              =      {20, 20};
    const int           QUESTION_GATE_EMBEDDINGS          =      15;

    const T             QUESTION_GATE_DROPOUT             =      0.3;

    // input here is fact word embedding and question_fact_word_gate_model final hidden.
    const int           QUESTION_GATE_IN_FACT_WORDS       =      TEXT_REPR_EMBEDDINGS + utils::vsum(QUESTION_GATE_STACKS);
    // input here is fact_model final hidden and question_fact_word_gate_model final hidden.
    const int           QUESTION_GATE_IN_FACTS            =      utils::vsum(TEXT_REPR_STACKS) + utils::vsum(QUESTION_GATE_STACKS);

    const int           QUESTION_GATE_HIDDEN              =      30;
    const int           QUESTION_GATE_OUT                 =      1;

    StackedShortcutLSTM<T> question_fact_gate_model;
    shared_mat question_fact_gate_embeddings;

    StackedShortcutLSTM<T> question_fact_word_gate_model;
    shared_mat question_fact_word_gate_embeddings;

    DelayedRNN<T> fact_gate;
    DelayedRNN<T> fact_word_gate;


    const vector<int>   HL_STACKS                  =      {15, 15};
    const int           HL_INPUT_SIZE              =      utils::vsum(TEXT_REPR_STACKS);
    const T             HL_DROPOUT                 =      0.5;

    StackedShortcutLSTM<T> hl_model;

    shared_mat please_start_prediction;

    const int           DECODER_INPUT              =      utils::vsum(HL_STACKS);
    const int           DECODER_OUTPUT; // gets initialized to vocabulary size in constructor
    Layer<T>            decoder;

    // TODO:
    // -> sparsity on words gates
    // -> explicit error on fact hidden gates
    // -> second order (between question and fact) relation for fact word gating
    // -> consider quadratic form]
    // -> add multiple answers

    shared_ptr<Vocab> vocab;

    public:
        vector<shared_mat> parameters() {
            vector<shared_mat> res;
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
                question_fact_word_gate_model(model.question_fact_word_gate_model, copy_w, copy_dw),
                fact_gate(model.fact_gate, copy_w, copy_dw),
                fact_word_gate(model.fact_word_gate, copy_w, copy_dw),
                question_model(model.question_model, copy_w, copy_dw),
                fact_model(model.fact_model, copy_w, copy_dw),
                hl_model(model.hl_model, copy_w, copy_dw),
                DECODER_OUTPUT(model.DECODER_OUTPUT),
                decoder(model.decoder, copy_w, copy_dw) {
            vocab = model.vocab;
            question_fact_gate_embeddings =
                    make_shared<mat>(*model.question_fact_gate_embeddings, copy_w, copy_dw);
            question_fact_word_gate_embeddings =
                    make_shared<mat>(*model.question_fact_word_gate_embeddings, copy_w, copy_dw);
            fact_embeddings =
                    make_shared<mat>(*model.fact_embeddings, copy_w, copy_dw);
            question_representation_embeddings =
                    make_shared<mat>(*model.question_representation_embeddings, copy_w, copy_dw);

            please_start_prediction =
                    make_shared<mat>(*model.please_start_prediction, copy_w, copy_dw);
        }

        LstmBabiModel(shared_ptr<Vocab> vocabulary) :
                question_fact_gate_model(QUESTION_GATE_EMBEDDINGS, QUESTION_GATE_STACKS),
                question_fact_word_gate_model(QUESTION_GATE_EMBEDDINGS, QUESTION_GATE_STACKS),
                fact_gate(QUESTION_GATE_IN_FACTS, QUESTION_GATE_HIDDEN, QUESTION_GATE_OUT),
                fact_word_gate(QUESTION_GATE_IN_FACT_WORDS, QUESTION_GATE_HIDDEN, QUESTION_GATE_OUT),
                question_model(TEXT_REPR_EMBEDDINGS, TEXT_REPR_STACKS),
                fact_model(TEXT_REPR_EMBEDDINGS, TEXT_REPR_STACKS),
                hl_model(HL_INPUT_SIZE, HL_STACKS),
                DECODER_OUTPUT(vocabulary->word2index.size()),
                decoder(DECODER_INPUT, vocabulary->word2index.size()) {

            vocab = vocabulary;

            size_t n_words = vocab->index2word.size();
            question_fact_gate_embeddings =
                    make_shared<mat>(n_words, QUESTION_GATE_EMBEDDINGS);
            question_fact_word_gate_embeddings =
                    make_shared<mat>(n_words, QUESTION_GATE_EMBEDDINGS);
            fact_embeddings =
                    make_shared<mat>(n_words, TEXT_REPR_EMBEDDINGS);
            question_representation_embeddings =
                    make_shared<mat>(n_words, TEXT_REPR_EMBEDDINGS);

            please_start_prediction =
                    make_shared<mat>(HL_INPUT_SIZE, 1);

        }


        Seq<shared_mat> get_embeddings(graph_t& G,
                                       const vector<string>& words,
                                       shared_mat embeddings) {
            Seq<shared_mat> seq;
            for (auto& word: words) {
                auto question_idx = vocab->word2index.at(word);
                auto embedding = G.row_pluck(embeddings, question_idx);
                seq.push_back(embedding);
                // We don't need explicitly start prediction token because
                // each question is guaranteed to end with "?" token.
            }
            return seq;
        }

        Seq<shared_mat> gate_memory(graph_t& G,
                                   Seq<shared_mat> seq,
                                   const DelayedRNN<T>& gate,
                                   shared_mat gate_input) {
            Seq<shared_mat> memory_seq;
            // By default initialized to zeros.
            auto prev_hidden = gate.initial_states();
            for (auto& embedding : seq) {
                // out_state: next_hidden, output
                auto out_state = gate.activate(G,
                                               G.vstack(embedding, gate_input),
                                               prev_hidden);
                prev_hidden = out_state.first;
                // memory - gate activation - how much of that embedding do we keep.
                auto memory = G.sigmoid(out_state.second);
                memory_seq.push_back(memory);
            }
            return memory_seq;
        }

        Seq<shared_mat> apply_gate(graph_t& G,
                                   Seq<shared_mat> memory,
                                   Seq<shared_mat> seq) {
            assert(memory.size() == seq.size());
            Seq<shared_mat> gated_seq;
            for(int i=0; i< memory.size(); ++i) {
                gated_seq.push_back(G.eltmul_broadcast_rowwise(seq[i], memory[i]));
            }
            return gated_seq;
        }


        shared_mat lstm_final_activation(graph_t& G,
                                         const Seq<shared_mat>& embeddings,
                                         const StackedShortcutLSTM<T>& model,
                                         T dropout_value) {
            auto out_state = model.activate_sequence(
                    G, model.initial_states(), embeddings, dropout_value);
            // out_state.second corresponds to LSTM hidden (as opposed to memory).
            return G.vstack(out_state.second);
        }

        shared_mat predict_answer_distribution(graph_t& G,
                                               const vector<vector<string>>& facts,
                                               const vector<string>& question,
                                               bool use_dropout) {
            auto fact_word_gate_hidden = lstm_final_activation(G,
                                                               get_embeddings(G, reversed(question), question_fact_word_gate_embeddings),
                                                               question_fact_word_gate_model,
                                                               use_dropout ? QUESTION_GATE_DROPOUT : 0.0);

            auto fact_gate_hidden = lstm_final_activation(G,
                                                          get_embeddings(G, reversed(question), question_fact_gate_embeddings),
                                                          question_fact_gate_model,
                                                          use_dropout ? QUESTION_GATE_DROPOUT : 0.0);

            auto question_representation = lstm_final_activation(G,
                                                                 get_embeddings(G, reversed(question), question_representation_embeddings),
                                                                 question_model,
                                                                 use_dropout ? TEXT_REPR_DROPOUT : 0.0);

            Seq<shared_mat> fact_representations;

            for (auto& fact: facts) {
                auto this_fact_embeddings = get_embeddings(G, reversed(fact), fact_embeddings);
                auto fact_word_gate_memory = gate_memory(G,
                                                        this_fact_embeddings,
                                                        fact_word_gate,
                                                        fact_word_gate_hidden
                                                        );

                auto gated_embeddings = apply_gate(G, fact_word_gate_memory, this_fact_embeddings);

                auto fact_repr = lstm_final_activation(G,
                                                       gated_embeddings,
                                                       fact_model,
                                                       use_dropout ? TEXT_REPR_DROPOUT : 0.0);
                fact_representations.push_back(fact_repr);
            }

            auto fact_gate_memory = gate_memory(G, fact_representations, fact_gate, fact_gate_hidden);

            auto gated_facts = apply_gate(G, fact_gate_memory, fact_representations);
            // There is probably a better way
            Seq<shared_mat> hl_input;
            hl_input.push_back(question_representation);
            hl_input.insert(hl_input.end(), gated_facts.rbegin(), gated_facts.rend());
            hl_input.push_back(please_start_prediction);

            auto hl_hidden = lstm_final_activation(G,
                                                   hl_input,
                                                   hl_model,
                                                   use_dropout ? HL_DROPOUT : 0.0);

            auto log_probs = decoder.activate(G, hl_hidden);

            return log_probs;
        }

        T error(graph_t& G,
                     const vector<Fact*>& facts,
                     QA* qa,
                     bool use_dropout) {
            vector<vector<string>> facts_as_strings;

            for (auto fact_ptr: facts)
                facts_as_strings.push_back(fact_ptr->fact);

            shared_mat log_probs = predict_answer_distribution(G,
                                                               facts_as_strings,
                                                               qa->question,
                                                               use_dropout);

            uint answer_idx = vocab->word2index.at(qa->answer[0]);

            return cross_entropy(log_probs, answer_idx);
        }
};

template<typename T>
class LstmBabiModelRunner: public babi::Model {

    // TRAINING_PROCEDURE_PARAMS

    const float TRAINING_FRAC = 0.8;
    const float MINIMUM_IMPROVEMENT = 0.0001; // good one: 0.003
    const double VALIDATION_FORGETTING = 0.06;
    const int PATIENCE = 5;

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

    public:
        T compute_error(const vector<babi::Story>& dataset, Solver::AdaDelta<T>& solver, bool training) {
            T total_error = 0.0;

            const int NUM_THREADS = FLAGS_j;
            ThreadPool pool(NUM_THREADS);

            if (thread_models.size() == 0) {
                for (int i = 0; i < NUM_THREADS; ++i) {
                    thread_models.push_back(model->shallow_copy());
                }
            }
            std::atomic<int> num_questions(0);

            vector<double> thread_error(NUM_THREADS, 0.0);

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
                        auto& story = dataset[story_id];
                        vector<Fact*> facts_so_far;

                        for(auto& item_ptr : story) {
                            if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                                facts_so_far.push_back(f);
                            } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                                // When we are training we want to do backprop
                                Graph<T> G(training);
                                // When we are training we want to use dropout
                                thread_error[ThreadPool::get_thread_number()] +=
                                        thread_model.error(G, facts_so_far, qa, training);
                                num_questions += 1;
                                if (training)
                                    G.backward();
                            }
                        }
                        facts_so_far.clear();
                        // Only update weights during training.
                    }
                    if (training)
                        solver.step(params, 0.00);
                });
            }

            pool.wait_until_idle();

            for (int i=0; i<NUM_THREADS; ++i)
                total_error += thread_error[i];

            return total_error/num_questions;
        }


        void train(const vector<babi::Story>& data) {
            auto vocab_vector = data_to_vocab(data);
            vocab = make_shared<Vocab> (vocab_vector);

            model = std::make_shared<LstmBabiModel<T>>(vocab);

            int training_size = (int)(TRAINING_FRAC * data.size());
            std::vector<babi::Story> train(data.begin(), data.begin() + training_size);
            std::vector<babi::Story> validation(data.begin() + training_size, data.end());

            double training_error = 0.0;
            double validation_error = 0.0;
            double last_validation_error = std::numeric_limits<double>::infinity();

            int epoch = 0;
            int epochs_validation_increasing = 0;

            auto params = model->parameters();
            Solver::AdaDelta<T> solver(params, 0.95, 1e-9, 5.0);

            while (epochs_validation_increasing <= PATIENCE) {
                double training_error = compute_error(train, solver, true);
                double validation_error = compute_error(validation, solver, false);
                std::stringstream ss;
                ss << "Epoch " << ++epoch
                   << " validation: " << validation_error
                   << " training: " << training_error
                   << " last validation: " << last_validation_error;
                ThreadPool::print_safely(ss.str());

                if (validation_error < last_validation_error - MINIMUM_IMPROVEMENT) {
                    epochs_validation_increasing = 0;
                } else {
                    epochs_validation_increasing += 1;
                }
                double scaled_forgetting = VALIDATION_FORGETTING / (double)FLAGS_j;
                if (last_validation_error == std::numeric_limits<double>::infinity()) {
                    last_validation_error = validation_error;
                } else {
                    last_validation_error = scaled_forgetting * validation_error +
                                            (1.0 - scaled_forgetting)*last_validation_error;
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
            Graph<T> G(false);

            // Don't use dropout for validation.
            int word_idx = argmax(model->predict_answer_distribution(G, story_so_far, question, false));

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

    babi::benchmark_task<LstmBabiModelRunner<double>>("qa4_two-arg-relations");
    //babi::benchmark<LstmBabiModelRunner<double>>();
}
