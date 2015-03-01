#include <iostream>
#include <set>
#include <vector>

#include "core/babi.h"
#include "core/StackedShortcutModel.h"
#include "core/CrossEntropy.h"

using std::vector;
using std::string;
using std::set;
using utils::Vocab;
using std::shared_ptr;
using std::make_shared;
using babi::Item;
using babi::QA;
using babi::Fact;
using babi::Story;

class DumbModel: public babi::Model {
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


template<typename REAL_t>
class MarginallyLessDumbModel: public babi::Model {
    typedef Mat<REAL_t> mat;
    typedef shared_ptr<mat> shared_mat;
    typedef Graph<REAL_t> graph_t;
    typedef StackedShortcutModel<REAL_t> model_t;

    const int TEXT_STACK_SIZE =      1;
    const int TEXT_HIDDEN_SIZE =    20;
    const int HL_STACK_SIZE =      1;
    const int HL_HIDDEN_SIZE =    20;
    const int EMBEDDING_SIZE = 20;

    shared_ptr<Vocab> vocab;

    shared_ptr<model_t> question_model;
    shared_ptr<model_t> fact_model;
    shared_ptr<model_t> story_model;

    shared_mat start_prediction;

    vector<shared_mat> parameters() const {
        vector<shared_mat> res;
        for(auto& m: {question_model, fact_model, story_model}) {
            auto params = m->parameters();
            res.insert(res.end(), params.begin(), params.end());
        }
        res.emplace_back(start_prediction);
        return res;
    }


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
                1); // unused output
        fact_model = make_shared<model_t>(
                vocab_size,
                EMBEDDING_SIZE,
                TEXT_STACK_SIZE,
                TEXT_HIDDEN_SIZE,
                1); // unused output
        story_model = make_shared<model_t>(
                0,
                TEXT_STACK_SIZE*TEXT_HIDDEN_SIZE,
                HL_HIDDEN_SIZE,
                HL_STACK_SIZE,
                vocab_size);
        start_prediction = make_shared<mat>(TEXT_HIDDEN_SIZE*TEXT_STACK_SIZE, 1);

    }

    public:
        shared_mat activate_words(graph_t& G, model_t& model, const VS& words) {
            auto ex = vocab->transform(words);
            auto out_state = model.get_final_activation(G, ex);
            // TODO(szymon): Implement G.join method, so that we can join all the hidden
            // from different levels of stacks.
            // return G.join(out_state.second);
            return G.hstack(out_state.second);
        }

        shared_mat activate_story(graph_t& G, const vector<shared_mat>& facts, shared_mat question) {
            auto state = story_model->initial_states();

            for (auto& fact: facts) {
                state = forward_LSTMs(G,
                                      fact,
                                      state,
                                      story_model->base_cell,
                                      story_model->cells);
            }

            state = forward_LSTMs(G,
                                  question,
                                  state,
                                  story_model->base_cell,
                                  story_model->cells);

            state = forward_LSTMs(G,
                                  start_prediction,
                                  state,
                                  story_model->base_cell,
                                  story_model->cells);

            auto log_probs = story_model->decoder.activate(G,
                                                           start_prediction,
                                                           state.second);

            return log_probs;
        }

        shared_mat predict_answer_distribution(graph_t& G,
                                               const vector<Fact*>& facts,
                                               QA* qa) {
            vector<shared_mat> fact_hiddens;

            for (auto& fact: facts) {
                fact_hiddens.emplace_back(activate_words(G, *fact_model, fact->fact));
            }

            shared_mat question_hidden = activate_words(G, *question_model, qa->question);

            shared_mat story_activation = activate_story(G, fact_hiddens, question_hidden);

            return story_activation;
        }

        REAL_t error(graph_t& G,
                     const vector<Fact*>& facts,
                     QA* qa) {
            shared_mat log_probs = predict_answer_distribution(G, facts, qa);

            uint answer_idx = vocab->word2index.at(qa->answer[0]);
            return cross_entropy(log_probs, answer_idx);
        }

        void train(const vector<babi::Story>& data) {
            vocab_from_training(data);
            construct_model(vocab->index2word.size());

            auto params = parameters();

            vector<Fact*> facts_so_far;
            Solver::AdaDelta<REAL_t> solver(params, 0.95, 1e-9, 5.0);
            for (auto& story : data) {
                for(auto& item_ptr : story) {
                    if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                        facts_so_far.push_back(f);
                    } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                        Graph<REAL_t> G;
                        error(G, facts_so_far, qa);
                        G.backward();
                    }
                }
                solver.step(params, 0.0);
            }
        }

        void new_story() {

        }

        void fact(const vector<string>& fact) {

        }
        vector<string> question(const vector<string>& question) {
            return {};
        }


};


int main() {
    babi::benchmark<MarginallyLessDumbModel<double>>(true);
}
