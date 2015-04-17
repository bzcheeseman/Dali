#ifndef DALI_VISUALIZER_VISUALIZER_H
#define DALI_VISUALIZER_VISUALIZER_H

#include <gflags/gflags.h>
#include <json11.hpp>
#include <redox.hpp>
#include <string>

#include "dali/visualizer/EventQueue.h"
#include "dali/utils/core_utils.h"

DECLARE_string(visualizer_hostname);
DECLARE_int32(visualizer_port);


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
        std::vector<double> weights;

        Sentence(std::vector<std::string> tokens) : tokens(tokens) {
        }

        void set_weights(const std::vector<double>& _weights) {
            weights = _weights;
        }

        virtual Json to_json() override {
            return Json::object {
                { "type", "sentence" },
                { "weights", weights },
                { "words", tokens },
            };
        }
    };

    struct Sentences : public Visualizable {
        std::vector<sentence_ptr> sentences;
        std::vector<double> weights;

        Sentences(std::vector<sentence_ptr> sentences) : sentences(sentences) {
        }
        Sentences(std::vector<std::vector<std::string>> vec) : sentences(sentence_vector(vec)) {
        }

        void set_weights(const std::vector<double>& _weights) {
            weights = _weights;
        }

        virtual Json to_json() override {
            std::vector<Json> sentences_json;
            for (auto& sentence: sentences) {
                sentences_json.push_back(sentence->to_json());
            }

            return Json::object {
                { "type", "sentences" },
                { "weights", weights },
                { "sentences", sentences_json},
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
                { "type", "qa"},
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
                { "type", "classifier_example"},
                { "input", input->to_json()},
                { "output", output->to_json()},
            };
        }
    };

    struct FiniteDistribution : public Visualizable {
        static const std::vector<double> empty_vec;


        std::vector<double> distribution;
        std::vector<double> scores;
        std::vector<std::string> labels;
        int num_examples;


        FiniteDistribution(const std::vector<double>& distribution,
                           const std::vector<double>& scores,
                           const std::vector<std::string>& labels,
                           int max_examples = 5) :
                distribution(distribution),
                scores(scores),
                labels(labels) {

            assert2(labels.size() == distribution.size(),
                    "FiniteDistribution visualizer: sizes of labels and distribution differ");
            num_examples = std::min(max_examples, (int)distribution.size());

        }

        FiniteDistribution(const std::vector<double>& distribution,
                   const std::vector<std::string>& labels,
                   int max_examples = 5) : FiniteDistribution(distribution,
                                                              empty_vec,
                                                              labels,
                                                              max_examples) {
        }

        virtual Json to_json() override {
            std::vector<std::string> output_labels(num_examples);
            std::vector<double> output_probs(num_examples);
            std::vector<double> output_scores(num_examples);

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
                if (!scores.empty())
                    output_scores[iters] = scores[best_index];
            }
            if (scores.empty()) {
                return Json::object {
                    { "type", "finite_distribution"},
                    { "probabilities", output_probs },
                    { "labels", output_labels },
                };
            } else {
                return Json::object {
                    { "type", "finite_distribution"},
                    { "scores", output_scores },
                    { "probabilities", output_probs },
                    { "labels", output_labels },
                };
            }
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
        class duplicate_name_error : public std::runtime_error {
            public:
                explicit duplicate_name_error(const std::string& what_arg);
                explicit duplicate_name_error(const char* what_arg);
        };
        Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> rdx=nullptr);
        void feed(const json11::Json& obj);
        void feed(const std::string& str);
};

#endif
