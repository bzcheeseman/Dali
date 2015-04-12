#ifndef DALI_VISUALIZER_VISUALIZER_H
#define DALI_VISUALIZER_VISUALIZER_H
#include <json11.hpp>
#include <redox.hpp>
#include <string>

#include "dali/visualizer/EventQueue.h"
#include "dali/utils/core_utils.h"

// TODO: explain how this works
namespace visualizable {
    struct Visualizable;
    struct Sentence;
    struct Sentences;
    struct QA;
    struct ClassifierExample;

    typedef std::shared_ptr<Visualizable> visualizable_ptr;
    typedef std::shared_ptr<Sentence> sentence_ptr;
    typedef std::shared_ptr<Sentences> sentences_ptr;
    typedef std::shared_ptr<ClassifierExample> classifier_example_ptr;

    namespace {
        using json11::Json;
        using utils::assert2;

        template<typename T>
        std::vector<Json> to_json_array(const std::vector<T>& vec) {
            std::vector<Json> res;
            for(auto& item: vec) {
                res.push_back(Json(item));
            }
            return res;
        }

        std::vector<sentence_ptr> sentence_vector(const std::vector<std::vector<std::string>>& vec) {
            std::vector<sentence_ptr> res;
            for (auto& sentence: vec) {
                res.push_back(std::make_shared<Sentence>(sentence));
            }
            return res;
        }

    }

    struct Visualizable {
        virtual Json to_json() = 0;
    };

    struct Sentence : public Visualizable {
        std::vector<std::string> tokens;
        Sentence(std::vector<std::string> tokens) : tokens(tokens) {
        }

        virtual Json to_json() override {
            return Json::object {
                { "data_type", "sentence" },
                { "sentence", Json::array(to_json_array(tokens))},
            };
        }
    };

    struct Sentences : public Visualizable {
        std::vector<sentence_ptr> sentences;
        Sentences(std::vector<sentence_ptr> sentences) : sentences(sentences) {
        }
        Sentences(std::vector<std::vector<std::string>> vec) : sentences(sentence_vector(vec)) {
        }

        virtual Json to_json() override {
            std::vector<Json> sentences_json;
            for (auto& sentence: sentences) {
                sentences_json.push_back(sentence->to_json());
            }

            return Json::object {
                { "data_type", "sentences" },
                { "sentences", Json::array(sentences_json)},
            };
        }
    };

    struct QA : public Visualizable {
        visualizable_ptr context;
        sentence_ptr question;
        sentence_ptr answer;

        QA(visualizable_ptr context, sentence_ptr question, sentence_ptr answer) :
                context(context), question(question), answer(answer) {
        }

        virtual Json to_json() override {
            return Json::object {
                { "data_type", "qa"},
                { "context", context->to_json()},
                { "question", question->to_json()},
                { "answer", answer->to_json()},
            };
        }
    };

    struct ClassifierExample : public Visualizable {
        visualizable_ptr input;
        visualizable_ptr output;

        ClassifierExample(visualizable_ptr input, visualizable_ptr output) :
                input(input), output(output) {
        }

        virtual Json to_json() override {
            return Json::object {
                { "data_type", "classifier_example"},
                { "input", input->to_json()},
                { "output", output->to_json()},
            };
        }
    };

    struct FiniteDistribution : public Visualizable {
        std::vector<double> distribution;
        std::vector<std::string> labels;
        int num_examples;

        FiniteDistribution(const std::vector<double>& distribution,
                           const std::vector<std::string>& labels,
                           int max_examples = 5) :
                distribution(distribution),
                labels(labels) {

            assert2(labels.size() == distribution.size(),
                    "FiniteDistribution visualizer: sizes of labels and distribution differ");
            num_examples = std::min(max_examples, (int)distribution.size());

        }

        virtual Json to_json() override {
            std::vector<std::string> output_labels(num_examples);
            std::vector<double> output_probs(num_examples);
            // Pick max_examples best answers;

            std::vector<bool> taken(distribution.size());
            for(int idx = 0; idx < distribution.size(); ++idx) taken[idx] = false;

            for(int iters = 0; iters < num_examples; ++iters) {
                int best_index = -1;
                for (int i=0; i<distribution.size(); ++i) {
                    if (taken[i]) continue;
                    if (best_index == -1 || distribution[i] > distribution[best_index]) best_index = i;
                }
                assert2(best_index != -1, "Szymon fucked up");

                taken[best_index] = true;
                output_probs[iters] = distribution[best_index];
                output_labels[iters] = labels[best_index];
            }
            return Json::object {
                { "data_type", "finite_distribution"},
                { "probabilities", to_json_array(output_probs) },
                { "labels", to_json_array(output_labels) },
            };
        }
    };
}

// TODO: explain what this does
class Visualizer {
    private:
        std::string my_namespace;
        std::shared_ptr<redox::Redox> rdx;
        EventQueue eq;
        EventQueue::repeating_t pinging;

        // TODO(szymon): this kind of connection state should
        // be handled by redox.. Remove this bool in the future.
        bool connected = false;

    public:
        Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> rdx=nullptr);
        void feed(const json11::Json& obj);
        void feed(const std::string& str);
};

#endif
