#ifndef DALI_MODELS_QUSTION_ANSWERING_H
#define DALI_MODELS_QUSTION_ANSWERING_H

#include <gflags/gflags.h>
#include <string>
#include <vector>
#include "dali/core.h"
#include "dali/utils.h"

DECLARE_string(pretrained_vectors);

// Szymon's Vegas logic:
// This is an anonymous namespace - what happens in
// Anonymous namespace stays
// in anonymous namespace!
namespace {
    using std::vector;
    using std::string;
    using std::shared_ptr;
    using utils::Vocab;
    using std::tuple;
    using utils::reversed;
    using std::make_tuple;
    using utils::vsum;
}

template<typename R>
class AveragingModel {
    public:
        int EMBEDDING_SIZE = 300;
        int HIDDEN_SIZE = 100;
        bool SEPARATE_EMBEDDINGS = false;
        bool SVD_INIT = true;

        shared_ptr<Vocab> vocab;

        vector<int> OUTPUT_NN_SIZES = {HIDDEN_SIZE, 100, 100, 1};
        vector<typename NeuralNetworkLayer<R>::activation_t> OUTPUT_NN_ACTIVATIONS =
            { MatOps<R>::tanh, MatOps<R>::tanh, NeuralNetworkLayer<R>::identity };

        Mat<R> embedding;

        Mat<R> text_embedding;
        Mat<R> question_embedding;
        Mat<R> answer_embedding;

        StackedInputLayer<R> words_repr_to_hidden;

        NeuralNetworkLayer<R> output_classifier;

        AveragingModel(const AveragingModel& other, bool copy_w, bool copy_dw) {
            if (SEPARATE_EMBEDDINGS) {
                text_embedding = Mat<R>(other.text_embedding, copy_w, copy_dw);
                question_embedding = Mat<R>(other.question_embedding, copy_w, copy_dw);
                answer_embedding = Mat<R>(other.answer_embedding, copy_w, copy_dw);
            } else {
                embedding = Mat<R>(other.embedding, copy_w, copy_dw);
            }
            words_repr_to_hidden =
                    StackedInputLayer<R>(other.words_repr_to_hidden, copy_w, copy_dw);
            output_classifier = NeuralNetworkLayer<R>(other.output_classifier, copy_w, copy_dw);
            vocab = other.vocab;
        }

        AveragingModel(shared_ptr<Vocab> _vocab) {
            vocab = _vocab;
            auto weight_init = weights<R>::uniform(1.0/EMBEDDING_SIZE);
            if (!FLAGS_pretrained_vectors.empty()) {
                assert(!SEPARATE_EMBEDDINGS);
                int num_loaded = glove::load_relevant_vectors(
                        FLAGS_pretrained_vectors, embedding, *vocab, 1000000);
                std::cout << num_loaded << " out of " << vocab->word2index.size()
                          << " word embeddings preloaded from glove." << std::endl;
                assert (embedding.dims(1) == EMBEDDING_SIZE);
            }

            if (SEPARATE_EMBEDDINGS) {
                text_embedding = Mat<R>(vocab->word2index.size(), EMBEDDING_SIZE, weight_init);
                question_embedding = Mat<R>(vocab->word2index.size(), EMBEDDING_SIZE, weight_init);
                answer_embedding = Mat<R>(vocab->word2index.size(), EMBEDDING_SIZE, weight_init);
            } else {
                embedding = Mat<R>(vocab->word2index.size(), EMBEDDING_SIZE, weight_init);
            }
            words_repr_to_hidden = StackedInputLayer<R>({ EMBEDDING_SIZE,
                                                          EMBEDDING_SIZE,
                                                          EMBEDDING_SIZE
                                                         }, HIDDEN_SIZE);

            output_classifier = NeuralNetworkLayer<R>(OUTPUT_NN_SIZES, OUTPUT_NN_ACTIVATIONS);

            if (SVD_INIT) {
                // Don't use SVD for embeddings!
                auto params = words_repr_to_hidden.parameters();
                auto params2 = output_classifier.parameters();
                params.insert(params.end(), params2.begin(), params2.end());
                for (auto param: params) {
                    weights<R>::svd(weights<R>::gaussian(1.0))(param);
                }
                std::cout << "Initialized weights with SVD!" << std::endl;
            }
        }

        AveragingModel shallow_copy() const {
            return AveragingModel(*this, false, true);
        }

        Mat<R> average_embeddings(const vector<Vocab::ind_t>& words, Mat<R> embedding) {

            vector<Mat<R>> words_embeddings;
            for (auto word_idx: words) {
                assert (word_idx < vocab->index2word.size());
                words_embeddings.push_back(embedding[word_idx]);
            }

            return MatOps<R>::add(words_embeddings) / (R)words_embeddings.size();
        }

        Mat<R> answer_score(const vector<vector<string>>& broken_text,
                            const vector<string>& question,
                            const vector<string>& answer) {
            vector<string> text;
            for (auto& statement: broken_text) {
                for(auto& token: statement) {
                    text.emplace_back(token);
                }
            }
            Mat<R> text_repr     = average_embeddings(vocab->encode(text),
                    SEPARATE_EMBEDDINGS ? text_embedding : embedding);
            Mat<R> question_repr = average_embeddings(vocab->encode(question),
                    SEPARATE_EMBEDDINGS ? question_embedding : embedding);
            Mat<R> answer_repr   = average_embeddings(vocab->encode(answer),
                    SEPARATE_EMBEDDINGS ? answer_embedding : embedding);

            Mat<R> hidden = words_repr_to_hidden.activate({text_repr,
                                                           question_repr,
                                                           answer_repr}).tanh();

            return output_classifier.activate(hidden);
        }

        vector<Mat<R>> parameters() {
            vector<Mat<R>> params;
            if (SEPARATE_EMBEDDINGS) {
                params = { text_embedding, question_embedding, answer_embedding };
            } else {
                params = { embedding };
            }
            auto temp = words_repr_to_hidden.parameters();
            params.insert(params.end(), temp.begin(), temp.end());
            temp = output_classifier.parameters();
            params.insert(params.end(), temp.begin(), temp.end());
            return params;
        }

        vector<Mat<R>> answer_scores(const vector<vector<string>>& text,
                                     const vector<string>& question,
                                     const vector<vector<string>>& answers) {
            vector<Mat<R>> scores;
            for (auto& answer: answers) {
                scores.push_back(answer_score(text, question, answer));
            }

            return scores;
        }

        Mat<R> error(const vector<vector<string>>& text,
                     const vector<string>& question,
                     const vector<vector<string>>& answers,
                     int correct_answer) {
            auto scores = answer_scores(text, question, answers);

            R margin = 0.1;

            Mat<R> error(1,1);
            for (int aidx=0; aidx < answers.size(); ++aidx) {
                if (aidx == correct_answer) continue;
                error = error + MatOps<R>::max(scores[aidx] - scores[correct_answer] + margin, 0.0);
            }

            return error;
        }

        int predict(const vector<vector<string>>& text,
                    const vector<string>& question,
                    const vector<vector<string>>& answers) {
            auto scores = answer_scores(text, question, answers);

            return MatOps<R>::vstack(scores).argmax();
        }
};

template<typename R>
class RecurrentGate : public AbstractLayer<R> {
    int input_size;
    int hidden_size;

    public:
        RNN<R> recurrent;
        Layer<R> gate_classifier;

        RecurrentGate() {
        }

        RecurrentGate(int input_size, int hidden_size) :
                input_size(input_size),
                hidden_size(hidden_size),
                recurrent(input_size, hidden_size),
                gate_classifier(hidden_size, 1) {
        }

        RecurrentGate (const RecurrentGate<R>& other, bool copy_w, bool copy_dw) :
            input_size(other.input_size),
            hidden_size(other.hidden_size),
            recurrent(other.recurrent, copy_w, copy_dw),
            gate_classifier(other.gate_classifier, copy_w, copy_dw) {
        }

        Mat<R> initial_states() const {
            return Mat<R>(hidden_size, 1);
        }

        tuple<Mat<R>,Mat<R>> activate(Mat<R> input, Mat<R> prev_hidden) const {
            auto next_hidden = recurrent.activate(input, prev_hidden).tanh();
            auto output = gate_classifier.activate(next_hidden).sigmoid();
            return make_tuple(output, next_hidden);
        }

        virtual vector<Mat<R>> parameters() const {
            std::vector<Mat<R>> ret;

            auto rnn_params = recurrent.parameters();
            auto gc_params = gate_classifier.parameters();
            ret.insert(ret.end(), rnn_params.begin(), rnn_params.end());
            ret.insert(ret.end(), gc_params.begin(), gc_params.end());
            return ret;
        }
};

template <typename R>
class LolGate : public AbstractLayer<R> {

    public:
    int input1_size;
    int input2_size;
    int second_order_terms;
    int hidden;

    RecurrentGate<R> gate;
    SecondOrderCombinator<R> combinator;

    LolGate() {
    }

    LolGate(int input1_size, int input2_size, int second_order_terms) :
            input1_size(input1_size),
            input2_size(input2_size),
            second_order_terms(second_order_terms),
            gate(second_order_terms, 1),
            combinator(input1_size, input2_size, second_order_terms) {
    }

    LolGate (const LolGate<R>& other, bool copy_w, bool copy_dw) :
            input1_size(other.input1_size),
            input2_size(other.input2_size),
            second_order_terms(other.second_order_terms),
            gate(other.gate, copy_w, copy_dw),
            combinator(other.combinator, copy_w, copy_dw) {
    }

    Mat<R> initial_states() const {
        return gate.initial_states();
    }

    tuple<Mat<R>,Mat<R>> activate(Mat<R> input1, Mat<R> input2, Mat<R> prev_hidden) const {
        auto gate_input = combinator.activate(input1, input2);
        return gate.activate(gate_input, prev_hidden);
    }

    virtual vector<Mat<R>> parameters() const {
        std::vector<Mat<R>> ret;
        auto gate_params = gate.parameters();
        auto combinator_params = combinator.parameters();
        ret.insert(ret.end(), gate_params.begin(), gate_params.end());
        ret.insert(ret.end(), combinator_params.begin(), combinator_params.end());
        return ret;
    }
};

template<typename R>
class GatedLstmsModel {
    int EMBEDDING_SIZE = 300;
    int GATE_SECOND_ORDER = 40; // IDEA - dropout on second order features.
    vector<int> TEXT_STACKS = { 100, 100 };
    vector<int> HL_STACKS = { 100, 50, 50, 50 };
    double TEXT_DROPOUT = 0.2;
    double HL_DROPOUT = 0.5;
    double ERROR_MARGIN = 0.1;

    Mat<R> embeddings;

    StackedLSTM<R> single_fact_model;
    StackedLSTM<R> facts_model;
    StackedLSTM<R> question_model;
    StackedLSTM<R> answer_model;

    LolGate<R> fact_gate;
    LolGate<R> fact_word_gate;

    Mat<R> please_start_prediction;

    Layer<R>            decoder;

    shared_ptr<Vocab> vocab;
    public:

        vector<Mat<R>> parameters() {
            vector<Mat<R>> res;
            for (auto model: std::vector<AbstractLayer<R>*>({
                                &single_fact_model,
                                &facts_model,
                                &answer_model,
                                &fact_gate,
                                &fact_word_gate,
                                &decoder })) {
                auto params = model->parameters();
                res.insert(res.end(), params.begin(), params.end());
            }
            for (auto& matrix: { embeddings, please_start_prediction }) {
                res.emplace_back(matrix);
            }

            return res;
        }

        GatedLstmsModel<R> shallow_copy() {
            return GatedLstmsModel<R>(*this, false, true);
        }

        GatedLstmsModel(const GatedLstmsModel& other, bool copy_w, bool copy_dw) :
                embeddings(other.embeddings, copy_w, copy_dw),
                single_fact_model(other.single_fact_model, copy_w, copy_dw),
                facts_model(other.facts_model, copy_w, copy_dw),
                question_model(other.question_model, copy_w, copy_dw),
                answer_model(other.answer_model, copy_w, copy_dw),
                fact_gate(other.fact_gate, copy_w, copy_dw),
                fact_word_gate(other.fact_word_gate, copy_w, copy_dw),
                please_start_prediction(other.please_start_prediction, copy_w, copy_dw),
                decoder(other.decoder, copy_w, copy_dw) {
            vocab = other.vocab;
        }

        GatedLstmsModel(shared_ptr<Vocab> vocabulary) {
            vocab = vocabulary;
            size_t n_words = vocab->index2word.size();

            embeddings = Mat<R>(vocab->index2word.size(), EMBEDDING_SIZE,
                    weights<R>::uniform(1.0/EMBEDDING_SIZE));

            if (!FLAGS_pretrained_vectors.empty()) {
                int num_loaded = glove::load_relevant_vectors(
                        FLAGS_pretrained_vectors, embeddings, *vocab, 1000000);
                // consider glove embeddings a constant variable.
                // embeddings = MatOps<R>::consider_constant(embeddings);
                std::cout << num_loaded << " out of " << vocab->word2index.size()
                          << " word embeddings preloaded from glove." << std::endl;
                assert (embeddings.dims(1) == EMBEDDING_SIZE);
            }

            single_fact_model = StackedLSTM<R>(EMBEDDING_SIZE, TEXT_STACKS, true, true);
            question_model = StackedLSTM<R>(EMBEDDING_SIZE, TEXT_STACKS, true, true);
            answer_model = StackedLSTM<R>(EMBEDDING_SIZE, TEXT_STACKS, true, true);

            facts_model = StackedLSTM<R>(3*vsum(TEXT_STACKS), HL_STACKS, true, true);
            fact_gate = LolGate<R>(vsum(TEXT_STACKS), 2*vsum(TEXT_STACKS), GATE_SECOND_ORDER);
            fact_word_gate = LolGate<R>(EMBEDDING_SIZE, 2*vsum(TEXT_STACKS), GATE_SECOND_ORDER);
            please_start_prediction = Mat<R>(3*vsum(TEXT_STACKS), 1, weights<R>::uniform(1.0));

            decoder = Layer<R>(vsum(HL_STACKS), 1);
        }


        vector<Mat<R>> get_embeddings(const vector<string>& words) {
            vector<Mat<R>> seq;
            auto idxes = vocab->encode(words);
            for (auto& idx: idxes) {
                auto embedding = embeddings[idx];
                seq.push_back(embedding);
            }
            return seq;
        }

        vector<Mat<R>> gate_memory(vector<Mat<R>> seq,
                                const LolGate<R>& gate,
                                Mat<R> gate_input) {
            vector<Mat<R>> memory_seq;
            // By default initialized to zeros.
            auto prev_hidden = gate.initial_states();
            Mat<R> gate_activation;
            for (auto& embedding : seq) {
                // out_state: next_hidden, output
                std::tie(gate_activation, prev_hidden) =
                        gate.activate(embedding, gate_input, prev_hidden);
                // memory - gate activation - how much of that embedding do we keep.
                memory_seq.push_back(gate_activation.sigmoid());
            }
            return memory_seq;
        }

        vector<Mat<R>> apply_gate(vector<Mat<R>> memory,
                               vector<Mat<R>> seq) {
            assert(memory.size() == seq.size());
            vector<Mat<R>> gated_seq;
            for(int i=0; i < memory.size(); ++i) {
                gated_seq.push_back(seq[i].eltmul_broadcast_rowwise(memory[i]));
            }
            return gated_seq;
        }

        Mat<R> lstm_final_activation(const vector<Mat<R>>& embeddings,
                                     const StackedLSTM<R>& model,
                                     R dropout_value) {
            auto out_states = model.activate_sequence(model.initial_states(),
                                                     embeddings,
                                                     dropout_value);
            // out_state.second corresponds to LSTM hidden (as opposed to memory).
            return MatOps<R>::vstack(LSTM<R>::State::hiddens(out_states));
        }

        std::tuple<Mat<R>,Mat<R>,Mat<R>> answer_score(const vector<vector<string>>& facts,
                                                      const vector<string>& question,
                                                      const vector<string>& answer,
                                                      bool use_dropout) {
            auto question_hidden = lstm_final_activation(get_embeddings(reversed(question)),
                                                         question_model,
                                                         use_dropout ? TEXT_DROPOUT : 0.0);

            auto answer_hidden = lstm_final_activation(get_embeddings(reversed(answer)),
                                                       answer_model,
                                                       use_dropout ? TEXT_DROPOUT : 0.0);
            Mat<R> reading_context = MatOps<R>::vstack(question_hidden, answer_hidden);

            vector<Mat<R>> fact_representations;
            Mat<R> word_fact_gate_memory_sum(1,1);

            for (auto& fact: facts) {
                auto this_fact_embeddings = get_embeddings(reversed(fact));
                auto fact_word_gate_memory = gate_memory(this_fact_embeddings,
                                                         fact_word_gate,
                                                         reading_context);
                for (auto mem: fact_word_gate_memory) {
                    word_fact_gate_memory_sum = word_fact_gate_memory_sum + mem;
                }

                auto gated_embeddings = apply_gate(fact_word_gate_memory, this_fact_embeddings);

                auto fact_repr = lstm_final_activation(
                        gated_embeddings,
                        single_fact_model,
                        use_dropout ? TEXT_DROPOUT : 0.0);
                fact_representations.push_back(fact_repr);
            }

            auto fact_gate_memory = gate_memory(fact_representations, fact_gate, reading_context);

            auto gated_facts = apply_gate(fact_gate_memory, fact_representations);

            vector<Mat<R>> hl_input;
            for (auto& gate_fact : gated_facts) {
                hl_input.push_back(MatOps<R>::vstack(reading_context, gate_fact));
            }

            hl_input.push_back(please_start_prediction);

            auto hl_hidden = lstm_final_activation(
                    hl_input, facts_model, use_dropout ? HL_DROPOUT : 0.0);

            auto score = decoder.activate(hl_hidden);

            auto fact_gate_memory_sum = Mat<R>(1,1);
            for(auto& memory: fact_gate_memory)
                fact_gate_memory_sum = fact_gate_memory_sum + memory;

            fact_gate_memory_sum = fact_gate_memory_sum/fact_gate_memory.size();

            return std::make_tuple(score,
                                   fact_gate_memory_sum,
                                   word_fact_gate_memory_sum);
        }

        Mat<R> error(const vector<vector<string>>& facts,
                     const vector<string>& question,
                     const vector<vector<string>>& answers,
                     int correct_answer) {
            Mat<R> score, word_sparsity, fact_sparsity;

            vector<Mat<R>> scores;
            Mat<R> total_word_sparsity(1,1);
            Mat<R> total_fact_sparsity(1,1);

            for (auto& answer: answers) {
                std::tie(score, word_sparsity, fact_sparsity) =
                        answer_score(facts, question, answer, true);
                scores.push_back(score);
                total_word_sparsity = total_word_sparsity + word_sparsity;
                total_fact_sparsity = total_fact_sparsity + fact_sparsity;
            }

            Mat<R> margin_loss(1,1);
            for (int aidx=0; aidx < answers.size(); ++aidx) {
                if (aidx == correct_answer) continue;
                margin_loss = margin_loss +
                        MatOps<R>::max(scores[aidx] - scores[correct_answer] + ERROR_MARGIN, 0.0);
            }

            return margin_loss + total_fact_sparsity * 0.01 + total_word_sparsity * 0.0001;
        }

        int predict(const vector<vector<string>>& facts,
                    const vector<string>& question,
                    const vector<vector<string>>& answers) {
            Mat<R> score, word_sparsity, fact_sparsity;
            vector<Mat<R>> scores;
            for (auto& answer: answers) {
                std::tie(score, word_sparsity, fact_sparsity) =
                        answer_score(facts, question, answer, false);
                scores.push_back(score);
            }

            return MatOps<R>::vstack(scores).argmax();
        }
};


#endif
