#ifndef DALI_MODELS_QUSTION_ANSWERING_H
#define DALI_MODELS_QUSTION_ANSWERING_H

#include <gflags/gflags.h>
#include <string>
#include <vector>

#include "dali/core.h"
#include "dali/utils.h"

DECLARE_string(pretrained_vectors);

// This is an anonymous namespace - what happens in Anonymous namespace stays
// in anonymous namespace!
namespace {
    using std::vector;
    using std::string;
    using std::shared_ptr;
    using utils::Vocab;

    template<typename R>
    class AveragingModelInternal {
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

            AveragingModelInternal(const AveragingModelInternal& other, bool copy_w, bool copy_dw) {
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

            AveragingModelInternal(shared_ptr<Vocab> _vocab) {
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

            AveragingModelInternal shallow_copy() const {
                return AveragingModelInternal(*this, false, true);
            }

            Mat<R> average_embeddings(const vector<Vocab::ind_t>& words, Mat<R> embedding) {

                vector<Mat<R>> words_embeddings;
                for (auto word_idx: words) {
                    assert (word_idx < vocab->index2word.size());
                    words_embeddings.push_back(embedding[word_idx]);
                }

                return MatOps<R>::add(words_embeddings) / (R)words_embeddings.size();
            }

            Mat<R> answer_score(const vector<string>& text,
                                const vector<string>& question,
                                const vector<string>& answer) {
                Mat<R> text_repr     = average_embeddings(vocab->transform(text),
                        SEPARATE_EMBEDDINGS ? text_embedding : embedding);
                Mat<R> question_repr = average_embeddings(vocab->transform(question),
                        SEPARATE_EMBEDDINGS ? question_embedding : embedding);
                Mat<R> answer_repr   = average_embeddings(vocab->transform(answer),
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

            vector<Mat<R>> answer_scores(const vector<string>& text,
                                         const vector<string>& question,
                                         const vector<vector<string>>& answers) {
                vector<Mat<R>> scores;
                for (auto& answer: answers) {
                    scores.push_back(answer_score(text, question, answer));
                }

                return scores;
            }

            Mat<R> error(const vector<string>& text,
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

            int predict(const vector<string>& text,
                        const vector<string>& question,
                        const vector<vector<string>>& answers) {
                auto scores = answer_scores(text, question, answers);

                return MatOps<R>::vstack(scores).argmax();
            }
    };
}

// Here we are exporting symbols from anonymous namespace to the outside world!
template<typename R>
using AveragingModel=AveragingModelInternal<R>;

#endif
