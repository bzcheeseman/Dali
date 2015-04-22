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

DEFINE_int32(j, 9, "Number of threads");
DEFINE_bool(solver_mutex, false, "Synchronous execution of solver step.");
DEFINE_bool(margin_loss, false, "Use margin loss instead of cross entropy");
DEFINE_int32(minibatch, 100, "How many stories to put in a single batch.");
DEFINE_double(margin, 0.1, "Margin for margine loss (must use --margin_loss).");
// Visualizer
std::shared_ptr<Visualizer> visualizer;

template<typename T>
struct StoryActivation {
    Mat<T> log_probs;
    vector<Mat<T>> fact_gate_memory;
    vector<vector<Mat<T>>> fact_word_gate_memory;

    StoryActivation(Mat<T> log_probs,
                    vector<Mat<T>> fact_gate_memory,
                    vector<vector<Mat<T>>> fact_word_gate_memory) :
            log_probs(log_probs),
            fact_gate_memory(fact_gate_memory),
            fact_word_gate_memory(fact_word_gate_memory) {
    }

    Mat<T> fact_sum() {
        Mat<T> res(1,1);
        for(auto& fact_mem: fact_gate_memory) {
            res = res + fact_mem;
        }
        return res;
    }

    Mat<T> fact_word_sum() {
        Mat<T> res(1,1);
        for(auto& fact_words: fact_word_gate_memory) {
            for(auto& word_mem: fact_words) {
                res = res + word_mem;
            }
        }
        return res;
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

template <typename T>
class LolGate : public AbstractLayer<T> {

    public:
    const int input1;
    const int input2;
    const int second_order_terms;
    const int hidden;

    RecurrentGate<T> gate;
    SecondOrderCombinator<T> combinator;

    LolGate(int input1, int input2, int second_order_terms, int hidden) :
            input1(input1),
            input2(input2),
            second_order_terms(second_order_terms),
            hidden(hidden),
            gate(second_order_terms, hidden),
            combinator(input1, input2, second_order_terms) {
    }

    LolGate(const LolGate<T>& other, bool copy_w, bool copy_dw) :
            input1(other.input1),
            input2(other.input2),
            second_order_terms(other.second_order_terms),
            hidden(other.hidden),
            gate(other.gate, copy_w, copy_dw),
            combinator(other.combinator, copy_w, copy_dw) {
    }

    Mat<T> initial_states() const {
        return gate.initial_states();
    }

    tuple<Mat<T>,Mat<T>> activate(Mat<T> input1, Mat<T> input2, Mat<T> prev_hidden) const {
        auto gate_input = combinator.activate(input1, input2);
        return gate.activate(gate_input, prev_hidden);
    }

    virtual vector<Mat<T>> parameters() const {
        std::vector<Mat<T>> ret;
        auto gate_params = gate.parameters();
        auto combinator_params = combinator.parameters();
        ret.insert(ret.end(), gate_params.begin(), gate_params.end());
        ret.insert(ret.end(), combinator_params.begin(), combinator_params.end());
        return ret;
    }
};


template<typename T>
class LstmBabiModel : public Model {
    // MODEL PARAMS

    StackedLSTM<T> fact_model;
    Mat<T> fact_embeddings;

    StackedLSTM<T> question_model;
    Mat<T> question_representation_embeddings;


    // const vector<int>   QUESTION_GATE_STACKS              =      {50};


    // input here is fact word embedding and question_fact_word_gate_model final hidden.
    // const int           QG_FACTS_INPUT1                   = utils::vsum(TEXT_REPR_STACKS);
    // const int           QG_FACT_WORDS_INPUT1              = TEXT_REPR_EMBEDDINGS;

    // const int           QG_INPUT2                         = utils::vsum(QUESTION_GATE_STACKS);
    // const int           QG_SECOND_ORDER                   = 40;
    // const int           QG_HIDDEN                         = 40;


    StackedLSTM<T> question_fact_gate_model;
    Mat<T> question_fact_gate_embeddings;

    StackedLSTM<T> question_fact_word_gate_model;
    Mat<T> question_fact_word_gate_embeddings;

    LolGate<T> fact_gate;
    LolGate<T> fact_word_gate;


    // const int           HL_INPUT_SIZE              =      utils::vsum(TEXT_REPR_STACKS);

    StackedLSTM<T> hl_model;

    Mat<T> please_start_prediction;

    // const int           DECODER_INPUT              =      utils::vsum(HL_STACKS);
    const int           DECODER_OUTPUT; // gets initialized to vocabulary size in constructor
    Layer<T>            decoder;

    // TODO:
    // -> we are mostly concerned with gates being on for positive facts.
    //    some false positives are acceptable.
    // -> second order (between question and fact) relation for fact word gating
    // -> consider quadratic form]
    // -> add multiple answers

    public:
        shared_ptr<Vocab> vocab;

        static Conf default_conf() {
            Conf conf = Model::default_conf();
            conf.def_bool("lstm_shortcut", true);
            conf.def_bool("lstm_feed_mry", true);
            conf.def_float("TEXT_REPR_DROPOUT", 0.0, 1.0, 0.3);
            conf.def_float("QUESTION_GATE_DROPOUT", 0.0, 1.0, 0.3);
            conf.def_float("HL_DROPOUT", 0.0, 1.0, 0.7);
            conf.def_int("QUESTION_GATE_EMBEDDINGS", 5, 50, 40);
            conf.def_int("TEXT_REPR_EMBEDDINGS", 5, 50, 40);
            conf.def_stacks("HL_STACKS", 2,7,4,10,100, 50, 20);
            conf.def_stacks("TEXT_REPR_STACKS", 1, 4, 2, 5, 50, 40, 30);
            conf.def_stacks("QUESTION_GATE_STACKS", 1, 4, 2, 5, 50, 40, 30);
            conf.def_int("QG_SECOND_ORDER", 3, 30, 10);
            conf.def_int("QG_HIDDEN", 3, 30, 10);
            return conf;
        }

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
                Model(model.c()),
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

        LstmBabiModel(shared_ptr<Vocab> vocabulary, Conf configuration) :
                Model(configuration),
                // first true - shortcut, second true - feed memory to gates
                question_fact_gate_model(c().i("QUESTION_GATE_EMBEDDINGS"),
                                         c().stacks("QUESTION_GATE_STACKS"),
                                         c().b("lstm_shortcut"),
                                         c().b("lstm_feed_mry")),
                question_fact_word_gate_model(c().i("QUESTION_GATE_EMBEDDINGS"),
                                              c().stacks("QUESTION_GATE_STACKS"),
                                              c().b("lstm_shortcut"),
                                              c().b("lstm_feed_mry")),
                fact_gate(utils::vsum(c().stacks("TEXT_REPR_STACKS")),
                          utils::vsum(c().stacks("QUESTION_GATE_STACKS")),
                          c().i("QG_SECOND_ORDER"),
                          c().i("QG_HIDDEN")),
                fact_word_gate(c().i("TEXT_REPR_EMBEDDINGS"),
                               utils::vsum(c().stacks("QUESTION_GATE_STACKS")),
                               c().i("QG_SECOND_ORDER"),
                               c().i("QG_HIDDEN")),
                question_model(c().i("TEXT_REPR_EMBEDDINGS"),
                               c().stacks("TEXT_REPR_STACKS"),
                               c().b("lstm_shortcut"),
                               c().b("lstm_feed_mry")),
                fact_model(c().i("TEXT_REPR_EMBEDDINGS"),
                           c().stacks("TEXT_REPR_STACKS"),
                           c().b("lstm_shortcut"),
                           c().b("lstm_feed_mry")),
                hl_model(utils::vsum(c().stacks("TEXT_REPR_STACKS")),
                         c().stacks("HL_STACKS"),
                         c().b("lstm_shortcut"),
                         c().b("lstm_feed_mry")),
                DECODER_OUTPUT(vocabulary->word2index.size()),
                decoder(utils::vsum(c().stacks("HL_STACKS")), vocabulary->word2index.size()) {
            vocab = vocabulary;
            size_t n_words = vocab->index2word.size();

            question_fact_gate_embeddings =
                    Mat<T>(n_words, c().i("QUESTION_GATE_EMBEDDINGS"),
                           weights<T>::uniform(1.0/c().i("QUESTION_GATE_EMBEDDINGS")));
            question_fact_word_gate_embeddings =
                    Mat<T>(n_words, c().i("QUESTION_GATE_EMBEDDINGS"),
                           weights<T>::uniform(1.0/c().i("QUESTION_GATE_EMBEDDINGS")));
            fact_embeddings =
                    Mat<T>(n_words, c().i("TEXT_REPR_EMBEDDINGS"),
                           weights<T>::uniform(1.0/c().i("TEXT_REPR_EMBEDDINGS")));
            question_representation_embeddings =
                    Mat<T>(n_words, c().i("TEXT_REPR_EMBEDDINGS"),
                           weights<T>::uniform(1.0/c().i("TEXT_REPR_EMBEDDINGS")));
            please_start_prediction =
                    Mat<T>(utils::vsum(c().stacks("TEXT_REPR_STACKS")), 1,
                           weights<T>::uniform(1.0));
        }


        vector<Mat<T>> get_embeddings(const vector<string>& words,
                                   Mat<T> embeddings) {
            vector<Mat<T>> seq;
            for (auto& word: words) {
                auto question_idx = vocab->word2index.at(word);
                auto embedding = embeddings[question_idx];
                seq.push_back(embedding);
                // We don't need explicitly start prediction token because
                // each question is guaranteed to end with "?" token.
            }
            return seq;
        }

        vector<Mat<T>> gate_memory(vector<Mat<T>> seq,
                                const LolGate<T>& gate,
                                Mat<T> gate_input) {
            vector<Mat<T>> memory_seq;
            // By default initialized to zeros.
            auto prev_hidden = gate.initial_states();
            Mat<T> gate_activation;
            for (auto& embedding : seq) {
                // out_state: next_hidden, output
                std:tie(gate_activation, prev_hidden) =
                        gate.activate(embedding, gate_input, prev_hidden);
                // memory - gate activation - how much of that embedding do we keep.
                memory_seq.push_back(gate_activation.sigmoid());
            }
            return memory_seq;
        }

        vector<Mat<T>> apply_gate(vector<Mat<T>> memory,
                               vector<Mat<T>> seq) {
            assert(memory.size() == seq.size());
            vector<Mat<T>> gated_seq;
            for(int i=0; i < memory.size(); ++i) {
                gated_seq.push_back(seq[i].eltmul_broadcast_rowwise(memory[i]));
            }
            return gated_seq;
        }

        Mat<T> lstm_final_activation(const vector<Mat<T>>& embeddings,
                                     const StackedLSTM<T>& model,
                                     T dropout_value) {
            auto out_states = model.activate_sequence(model.initial_states(),
                                                     embeddings,
                                                     dropout_value);
            // out_state.second corresponds to LSTM hidden (as opposed to memory).
            return MatOps<T>::vstack(LSTM<T>::State::hiddens(out_states));
        }

        StoryActivation<T> activate_story(const vector<vector<string>>& facts,
                                          const vector<string>& question,
                                          bool use_dropout) {
            auto fact_word_gate_hidden = lstm_final_activation(
                    get_embeddings(reversed(question), question_fact_word_gate_embeddings),
                    question_fact_word_gate_model,
                    use_dropout ? c().f("QUESTION_GATE_DROPOUT") : 0.0);

            auto fact_gate_hidden = lstm_final_activation(
                    get_embeddings(reversed(question), question_fact_gate_embeddings),
                    question_fact_gate_model,
                    use_dropout ? c().f("QUESTION_GATE_DROPOUT") : 0.0);

            auto question_representation = lstm_final_activation(
                    get_embeddings(reversed(question), question_representation_embeddings),
                    question_model,
                    use_dropout ? c().f("TEXT_REPR_DROPOUT") : 0.0);

            vector<Mat<T>> fact_representations;
            vector<vector<Mat<T>>> fact_words_gate_memory;

            for (auto& fact: facts) {
                auto this_fact_embeddings = get_embeddings(reversed(fact), fact_embeddings);
                auto this_fact_word_gate_memory = gate_memory(this_fact_embeddings,
                                                              fact_word_gate,
                                                              fact_word_gate_hidden);

                fact_words_gate_memory.push_back(this_fact_word_gate_memory);
                auto gated_embeddings = apply_gate(this_fact_word_gate_memory, this_fact_embeddings);

                auto fact_repr = lstm_final_activation(
                        this_fact_embeddings,//gated_embeddings,
                        fact_model,
                        use_dropout ? c().f("TEXT_REPR_DROPOUT") : 0.0);
                fact_representations.push_back(fact_repr);
            }

            auto fact_gate_memory = gate_memory(fact_representations, fact_gate, fact_gate_hidden);

            auto gated_facts = apply_gate(fact_gate_memory, fact_representations);
            // There is probably a better way
            vector<Mat<T>> hl_input;
            hl_input.push_back(question_representation);
            hl_input.insert(hl_input.end(), gated_facts.begin(), gated_facts.end());
            hl_input.push_back(please_start_prediction);

            auto hl_hidden = lstm_final_activation(
                    hl_input, hl_model, use_dropout ? c().f("HL_DROPOUT") : 0.0);

            auto log_probs = decoder.activate(hl_hidden);

            return StoryActivation<T>(log_probs,
                                      fact_gate_memory,
                                      fact_words_gate_memory);
        }

        void visualize_example(const vector<vector<string>>& facts,
                                     const vector<string>& question,
                                     const vector<string>& correct_answer) {
            graph::NoBackprop nb;
            auto activation = activate_story(facts, question, false);

            shared_ptr<visualizable::FiniteDistribution<T>> vdistribution;
            if (FLAGS_margin_loss) {
                auto scores = activation.log_probs;
                std::vector<double> scores_as_vec;
                for (int i=0; i < scores.dims(0); ++i) {
                    scores_as_vec.push_back(scores.w()(i,0));
                }
                auto distribution_as_vec = utils::normalize_weights(scores_as_vec);
                vdistribution = make_shared<visualizable::FiniteDistribution<T>>(
                        distribution_as_vec, scores_as_vec, vocab->index2word, 5);
            } else {
                auto distribution = MatOps<T>::softmax(activation.log_probs);
                std::vector<double> distribution_as_vec;
                for (int i=0; i < distribution.dims(0); ++i) {
                    distribution_as_vec.push_back(distribution.w()(i,0));
                }
                vdistribution = make_shared<visualizable::FiniteDistribution<T>>(
                        distribution_as_vec, vocab->index2word, 5);
            }


            vector<double> facts_weights;
            vector<std::shared_ptr<visualizable::Sentence<T>>> facts_sentences;
            for (int fidx=0; fidx < facts.size(); ++fidx) {
                auto vfact = make_shared<visualizable::Sentence<T>>(facts[fidx]);

                vector<double> words_weights;

                for (Mat<T> weight: activation.fact_word_gate_memory[fidx]) {
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

            auto vexample = make_shared<visualizable::ClassifierExample>(vqa, vdistribution);

            // visualizer->feed(vqa->to_json());
            visualizer->feed(vexample->to_json());
        }

};





/* PARAMETERS FOR TRAINING */
typedef double REAL_t;
typedef LstmBabiModel<REAL_t> BabiModel;



// TRAINING_PROCEDURE_PARAMS
const float TRAINING_FRAC = 0.8;
const float MINIMUM_IMPROVEMENT = 0.0001;
const double LONG_TERM_VALIDATION = 0.02;
const double SHORT_TERM_VALIDATION = 0.1;

// prediction haz dropout.
const int PREDICTION_PATIENCE = 200;

const double FACT_SELECTION_LAMBDA_MAX = 3.0;
const double FACT_WORD_SELECTION_LAMBDA_MAX = 0.0005;



/* TRAINING FUNCTIONS */
shared_ptr<BabiModel> model;
shared_ptr<BabiModel> best_model;

vector<BabiModel> thread_models;



MatrixXd errors(StoryActivation<REAL_t> activation,
                uint answer_idx,
                vector<int> supporting_facts) {
    Mat<REAL_t> prediction_error;
    if (FLAGS_margin_loss) {
        prediction_error = MatOps<REAL_t>::margin_loss(activation.log_probs, answer_idx, FLAGS_margin);
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
        float coeff = supporting ? 1.0 : 0.1;

        fact_selection_error = fact_selection_error + partial_error * coeff;
    }

    Mat<REAL_t> total_error;

    total_error = prediction_error
                + fact_selection_error * FACT_SELECTION_LAMBDA_MAX;
                //+ activation.fact_word_sum() * FACT_WORD_SELECTION_LAMBDA_MAX;

    total_error = total_error / FLAGS_minibatch;

    total_error.grad();

    MatrixXd reported_errors(3,1);
    reported_errors(0) = prediction_error.w()(0,0);
    reported_errors(1) = fact_selection_error.w()(0,0);
    reported_errors(2) = activation.fact_word_sum().w()(0,0);


    return reported_errors;
}

std::mutex solver_mutex;

// returns the errors;
MatrixXd run_epoch(const vector<babi::Story>& dataset,
                              Solver::AbstractSolver<REAL_t>* solver,
                              bool training) {


    const int NUM_THREADS = FLAGS_j;
    ThreadPool pool(NUM_THREADS);

    if (thread_models.size() == 0) {
        for (int i = 0; i < NUM_THREADS; ++i) {
            thread_models.push_back(model->shallow_copy());
        }
    }
    std::atomic<int> num_questions(0);

    vector<MatrixXd> thread_error;
    for (int thread=0; thread<NUM_THREADS; ++thread) {
        thread_error.emplace_back(3,1);
        thread_error.back().fill(0);
    }


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
        pool.run([batch, &dataset, training, &num_questions, &thread_error, &solver]() {
            BabiModel& thread_model = thread_models[ThreadPool::get_thread_number()];
            auto params = thread_model.parameters();
            for (auto story_id: batch) {
                auto story = dataset[story_id];

                babi::StoryParser parser(&story);
                vector<vector<string>> facts_so_far;
                QA* qa;
                while (!parser.done()) {
                    std::tie(facts_so_far, qa) = parser.next();
                    // When we are training we want to do backprop
                    graph::NoBackprop nb(!training);
                    // When we are training we want to use dropout
                    auto activation = thread_model.activate_story(
                            facts_so_far, qa->question, training);
                    uint answer_idx = model->vocab->word2index.at(qa->answer[0]);

                    thread_error[ThreadPool::get_thread_number()] +=
                            errors(activation, answer_idx, qa->supporting_facts);

                    num_questions += 1;
                    if (training)
                        graph::backward();
                 }
            }
            if (FLAGS_solver_mutex) {
                    std::lock_guard<decltype(solver_mutex)> guard(solver_mutex);
                    if (training)
                        solver->step(params);
            } else {
                if (training)
                    solver->step(params);
            }
        });
    }

    pool.wait_until_idle();
    MatrixXd total_error(3,1);
    total_error << 0, 0, 0;
    for (int i=0; i<NUM_THREADS; ++i)
        total_error += thread_error[i];
    total_error /= num_questions;

    return total_error;
}

void visualize_examples(const vector<babi::Story>& data, int num_examples) {
    while(num_examples--) {
        int example = rand()%data.size();
        int question_no = rand()%6;
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

vector<string> predict(const vector<vector<string>>& facts,
                       const vector<string>& question);

void train(const vector<babi::Story>& data, shared_ptr<Training> training_method) {
    for (auto param: model->parameters()) {
        weights<REAL_t>::svd(weights<REAL_t>::gaussian(1.0))(param);
    }

    int training_size = (int)(TRAINING_FRAC * data.size());
    std::vector<babi::Story> train(data.begin(), data.begin() + training_size);
    std::vector<babi::Story> validation(data.begin() + training_size, data.end());

    int epoch = 0;

    auto params = model->parameters();
    bool solver_reset_cache = false;

    //Solver::AdaDelta<REAL_t> solver(params, 0.95, 1e-9, 5.0);

    // Solver::Adam<REAL_t> solver(params);

    Solver::AdaGrad<REAL_t> solver(params, 1e-9, 100.0, 1e-8); solver_reset_cache = true;
    //AdaGrad:
    solver.step_size = 0.2 / FLAGS_minibatch;


    training_method->reset();

    double best_validation = run_epoch(validation, &solver, false)(0);
    best_model = std::make_shared<BabiModel>(*model, true, true);

    Throttled example_visualization;

    while (true) {
        auto training_errors = run_epoch(train, &solver, true);
        auto validation_errors = run_epoch(validation, &solver, false);
        if (!FLAGS_visualizer.empty()) {
            example_visualization.maybe_run(seconds(5), [&validation]() {
                visualize_examples(validation, 1);
            });
        }
        if (solver_reset_cache)
            solver.reset_caches(params);
        std::cout << "Epoch " << ++epoch << std::endl;
        std::cout << "Errors(prob_answer, fact_select, words_sparsity): " << std::endl
                  << "VALIDATION: " << validation_errors(0) << " "
                                    << validation_errors(1) << " "
                                    << validation_errors(2) << std::endl
                  << "TRAINING: "   << training_errors(0) << " "
                                    << training_errors(1) << " "
                                    << training_errors(2) << std::endl
                  << "VALIDATION ACCURACY: " << 100.0 * babi::accuracy(validation, predict) << "%" << std::endl;
        if (training_method->should_stop(validation_errors(0))) break;
        training_method->report();

        if (best_validation > training_method->validation_error()) {
            std::cout << "NEW WORLD RECORD!" << std::endl;
            best_validation = training_method->validation_error();
            best_model = std::make_shared<BabiModel>(*model, true, true);
        }
    }
}

void train(const vector<babi::Story>& data) {
    shared_ptr<LSTV> default_method = make_shared<LSTV>(SHORT_TERM_VALIDATION,
                                                        LONG_TERM_VALIDATION,
                                                        PREDICTION_PATIENCE);

    train(data, default_method);
}

vector<string> predict(const vector<vector<string>>& facts,
                       const vector<string>& question) {
    graph::NoBackprop nb;

    // Don't use dropout for validation.
    int word_idx = best_model->activate_story(facts, question, false).log_probs.argmax();

    return {best_model->vocab->index2word[word_idx]};
}

void reset(const vector<babi::Story>& data, Conf babi_model_conf=BabiModel::default_conf()) {
    thread_models.clear();
    auto vocab_vector = babi::vocab_from_data(data);
    model.reset();
    best_model.reset();
    model = std::make_shared<BabiModel>(make_shared<Vocab> (vocab_vector), babi_model_conf);
}

double benchmark_task(const std::string task) {
    std::cout << "Solving the most important problem in the world: " << task << std::endl;
    auto data = babi::Parser::training_data(task);
    reset(data);

    //shared_ptr<MaxEpochs> training_method = make_shared<MaxEpochs>(5);
    //train(data, training_method);
    train(data);
    double accuracy = babi::task_accuracy(task, predict);
    std::cout << "Accuracy on " << task << " is " << 100 * accuracy << "." << std::endl;
    return accuracy;
}

void benchmark_all() {
    vector<double> our_results;
    for (auto task : babi::tasks()) {
        our_results.push_back(100.0 * benchmark_task(task));
    }
    babi::compare_results(our_results);
}


void grid_search() {
    std::string task = "qa2_two-supporting-facts";
    auto data = babi::Parser::training_data(task);
    Conf babi_model_conf = BabiModel::default_conf();
    int iters = 1;
    double best_accuracy = 0.0;

    perturb_for(seconds(60*15), babi_model_conf, [&babi_model_conf, &task, &data, &iters, &best_accuracy]() {
        reset(data, babi_model_conf);
        std::cout << "Grid search iteration " << iters++ << std::endl;
        std::cout << "HL_STACK => "
                  << babi_model_conf.stacks("HL_STACKS") << std::endl;
        std::cout << "TEXT_REPR_STACKS => "
                  << babi_model_conf.stacks("TEXT_REPR_STACKS") << std::endl;
        std::cout << "QUESTION_GATE_STACKS => "
                  << babi_model_conf.stacks("QUESTION_GATE_STACKS") << std::endl;


        shared_ptr<TimeLimited> training_method = make_shared<TimeLimited>(seconds(30));
        train(data, training_method);
        double accuracy = babi::task_accuracy(task, predict);
        best_accuracy = std::max(accuracy, best_accuracy);
        std::cout << "Achieved " << 100.0 * accuracy << "% accuracy on task " << task
                  << " (best so far is " << 100.0 * best_accuracy
                  << "%). Configuration was: " << std::endl
                  << std::to_string(babi_model_conf, true) << std::endl;
        return -accuracy;
    });
    std::cout << "Best accuracy found by grid search (" << 100.0 * best_accuracy
              << "%) is achieved by the following configuration: " << std::endl
              << std::to_string(babi_model_conf, true) << std::endl;
}

int main(int argc, char** argv) {
    sane_crashes::activate();

    GFLAGS_NAMESPACE::SetUsageMessage(
        "\nBabi!"
    );

    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    Eigen::setNbThreads(0);
    Eigen::initParallel();

    int increment = 0;
    if (!FLAGS_visualizer.empty()) {
        visualizer = make_shared<Visualizer>(FLAGS_visualizer, true);
    }

    std::cout << "Number of threads: " << FLAGS_j << (FLAGS_solver_mutex ? "(with solver mutex)" : "") << std::endl;
    std::cout << "Using " << (FLAGS_margin_loss ? "margin loss" : "cross entropy") << std::endl;
    // grid_search();

    benchmark_all();


    // benchmark_task("multitasking");
    // benchmark_task("qa16_basic-induction");
    // benchmark_task("qa3_three-supporting-facts");
}
