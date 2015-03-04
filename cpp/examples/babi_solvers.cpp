#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
#include <gperftools/profiler.h>

#include "core/babi.h"
#include "core/Solver.h"
#include "core/CrossEntropy.h"
#include "core/Reporting.h"
#include "core/StackedModel.h"
#include "core/Seq.h"

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

// LSTM Model todo:
// -> add cross validation
// -> increase performance
// -> supporting fact gate - add second order term between question and answer.
template<typename REAL_t>
class LstmBabiModel {
    typedef Mat<REAL_t> mat;
    typedef shared_ptr<mat> shared_mat;
    typedef Graph<REAL_t> graph_t;
    typedef StackedModel<REAL_t> model_t;

    // MODEL PARAMS

    const int TEXT_STACK_SIZE =      2;
    const int TEXT_HIDDEN_SIZE =    20;
    const REAL_t TEXT_DROPOUT = 0.1;

    const int HL_STACK_SIZE =      4;
    const int HL_HIDDEN_SIZE =    15;
    const REAL_t HL_DROPOUT = 0.3;

    const int EMBEDDING_SIZE = 30;


    shared_ptr<model_t> question_model;
    shared_ptr<model_t> fact_model;
    shared_ptr<model_t> story_model;

    shared_mat please_start_prediction;
    shared_mat you_will_see_question_soon;

    void vocab_from_training(const vector<babi::Story>& data) {
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
        vector<string> vocab_vector(vocab_set.begin(), vocab_set.end());



        vocab = make_shared<Vocab> (vocab_vector);
    }

    void construct_model(int vocab_size) {
        question_model = make_shared<model_t>(
                vocab_size,
                EMBEDDING_SIZE,
                TEXT_STACK_SIZE,
                TEXT_HIDDEN_SIZE,
                1,
                true); // unused output
        fact_model = make_shared<model_t>(
                vocab_size,
                EMBEDDING_SIZE,
                TEXT_STACK_SIZE,
                TEXT_HIDDEN_SIZE,
                1,
                true); // unused output
        story_model = make_shared<model_t>(
                0,
                TEXT_STACK_SIZE*TEXT_HIDDEN_SIZE,
                HL_HIDDEN_SIZE,
                HL_STACK_SIZE,
                vocab_size,
                true);
        please_start_prediction = make_shared<mat>(TEXT_HIDDEN_SIZE*TEXT_STACK_SIZE, 1);
        you_will_see_question_soon = make_shared<mat>(TEXT_HIDDEN_SIZE*TEXT_STACK_SIZE, 1);
    }

    public:
        shared_ptr<Vocab> vocab;


        vector<shared_mat> parameters() {
            vector<shared_mat> res;
            for(auto& m: {question_model, fact_model, story_model}) {
                auto params = m->parameters();
                res.insert(res.end(), params.begin(), params.end());
            }
            res.emplace_back(please_start_prediction);
            res.emplace_back(you_will_see_question_soon);
            return res;
        }

        LstmBabiModel<REAL_t> shallow_copy() {
            return LstmBabiModel<REAL_t>(*this, false, true);
        }

        LstmBabiModel(const LstmBabiModel& model, bool copy_w, bool copy_dw) {
            vocab = model.vocab;
            question_model = make_shared<model_t>(*model.question_model, copy_w, copy_dw);
            fact_model = make_shared<model_t>(*model.fact_model, copy_w, copy_dw);
            story_model = make_shared<model_t>(*model.story_model, copy_w, copy_dw);
            please_start_prediction = make_shared<mat>(*model.please_start_prediction, copy_w, copy_dw);
            you_will_see_question_soon = make_shared<mat>(*model.you_will_see_question_soon, copy_w, copy_dw);
        }

        LstmBabiModel(const vector<babi::Story>& data) {
            vocab_from_training(data);
            construct_model(vocab->index2word.size());
        }



        // ACTIVATES A sequence of words (fact or question) on an LSTM, returns the output
        // hidden for that sequence of words.
        shared_mat activate_words(graph_t& G, model_t& model, const VS& words, bool use_dropout) {
            auto ex = vocab->transform(words);
            auto out_state = model.get_final_activation(G, ex, use_dropout ? TEXT_DROPOUT : 0.0);
            // TODO(szymon): Implement G.join method, so that we can join all the hidden
            // from different levels of stacks.
            // return G.join(out_state.second);
            return G.vstack(out_state.second);
        }

        // Activates a story - takes hiddens for facts and question
        // and feeds them to an LSTM, to get final activation for that story.
        shared_mat activate_story(graph_t& G, const vector<shared_mat>& facts,
                                  shared_mat question,
                                  bool use_dropout) {
            auto state = story_model->initial_states();
            utils::Timer a_timer("Forward");

            Seq<shared_mat> sequence;
            sequence.insert(sequence.end(), facts.begin(), facts.end());
            // sequence.push_back(you_will_see_question_soon);
            sequence.push_back(question);
            // sequence.push_back(please_start_prediction);

            state = story_model->stacked_lstm->activate_sequence(G,
                state,
                sequence,
                use_dropout ? HL_DROPOUT : 0.0);

            auto log_probs = story_model->decoder->activate(G,
                                                           please_start_prediction,
                                                           state.second);

            return log_probs;
        }

        shared_mat predict_answer_distribution(graph_t& G,
                                               const vector<vector<string>>& facts,
                                               const vector<string>& question,
                                               bool use_dropout) {
            vector<shared_mat> fact_hiddens;

            vector<string> tokens;
            for (auto& fact: facts) {
                tokens.clear();
                // tokens.insert(tokens.end(), question.begin(), question.end());
                // don't need fact coming token because question ends with question mark
                tokens.insert(tokens.end(), fact.begin(), fact.end());
                // instead of using end of stream we rely on the fact that
                // each fact ends with a dot
                fact_hiddens.emplace_back(activate_words(G, *fact_model, tokens, use_dropout));
            }

            // similarly to above each question ends with a question mark.
            shared_mat question_hidden = activate_words(G, *question_model, question, use_dropout);

            shared_mat story_activation = activate_story(G, fact_hiddens, question_hidden, use_dropout);

            return story_activation;
        }

        REAL_t error(graph_t& G,
                     const vector<Fact*>& facts,
                     QA* qa,
                     bool use_dropout) {
            vector<vector<string>> facts_as_strings;

            for (auto fact_ptr: facts)
                facts_as_strings.push_back(fact_ptr->fact);

            shared_mat log_probs = predict_answer_distribution(G, facts_as_strings, qa->question, use_dropout);

            uint answer_idx = vocab->word2index.at(qa->answer[0]);

            return cross_entropy(log_probs, answer_idx);
        }
};

template<typename REAL_t>
class LstmBabiModelRunner: public babi::Model {

    // TRAINING_PROCEDURE_PARAMS

    const float TRAINING_FRAC = 0.8;
    const float MINIMUM_IMPROVEMENT = 0.0001; // good one: 0.003
    const int PATIENCE = 20;


    shared_ptr<LstmBabiModel<REAL_t>> model;
    public:

        REAL_t compute_error(const vector<babi::Story>& dataset, Solver::AdaDelta<REAL_t>& solver, bool training) {
            REAL_t total_error = 0.0;

            const int NUM_THREADS = 9;
            ThreadPool pool(NUM_THREADS);

            vector<LstmBabiModel<REAL_t>> thread_models;
            for (int i=0; i<NUM_THREADS; ++i) {
                thread_models.push_back(model->shallow_copy());
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

                pool.run([batch, &dataset, &thread_models, training, &num_questions, &thread_error, &solver]() {
                    LstmBabiModel<REAL_t>& thread_model = thread_models[ThreadPool::get_thread_number()];
                    auto params = thread_model.parameters();
                    for (auto story_id: batch) {
                        auto& story = dataset[story_id];
                        vector<Fact*> facts_so_far;

                        for(auto& item_ptr : story) {
                            if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                                facts_so_far.push_back(f);
                            } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                                // When we are training we want to do backprop
                                Graph<REAL_t> G(training);
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
            model = std::make_shared<LstmBabiModel<REAL_t>>(data);

            int training_size = (int)(TRAINING_FRAC * data.size());
            std::vector<babi::Story> train(data.begin(), data.begin() + training_size);
            std::vector<babi::Story> validation(data.begin() + training_size, data.end());

            double training_error = 0.0;
            double validation_error = 0.0;
            double last_validation_error = std::numeric_limits<double>::infinity();

            int epoch = 0;
            int epochs_validation_increasing = 0;

            auto params = model->parameters();
            Solver::AdaDelta<REAL_t> solver(params, 0.95, 1e-9, 5.0);

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
                    if (std::numeric_limits<double>::infinity() == last_validation_error)
                          last_validation_error = validation_error;
                    else
                          last_validation_error = 0.2*validation_error + 0.8*last_validation_error;
                } else {
                    epochs_validation_increasing += 1;
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
            Graph<REAL_t> G(false);

            // Don't use dropout for validation.
            int word_idx = argmax(model->predict_answer_distribution(G, story_so_far, question, false));

            return {model->vocab->index2word[word_idx]};
        }
};


int main() {
    //babi::benchmark_task<LstmBabiModelRunner<double>>("qa11_basic-coreference");
    babi::benchmark<LstmBabiModelRunner<double>>();
}
