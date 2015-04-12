#ifndef DALI_VISUALIZER_VISUALIZER_H
#define DALI_VISUALIZER_VISUALIZER_H

#include <json11.hpp>
#include <redox.hpp>
#include <string>

#include "dali/visualizer/EventQueue.h"



namespace visualizable {
    using json11::Json;

    namespace {
        std::vector<Json> to_json_array(const std::vector<std::string>& vs) {
            std::vector<Json> res;
            for(auto& str: vs) {
                res.push_back(Json(str));
            }
            return res;
        }
    }

    struct Visualizable;
    struct Sentence;
    struct Sentences;
    struct QA;
    struct ClassifierExample;

    typedef std::shared_ptr<Visualizable> visualizable_ptr;
    typedef std::shared_ptr<Sentence> sentence_ptr;
    typedef std::shared_ptr<Sentences> sentences_ptr;
    typedef std::shared_ptr<ClassifierExample> classifier_example_ptr;

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
}

class Visualizer {
    private:
        std::string my_namespace;
        std::shared_ptr<redox::Redox> rdx;
        EventQueue eq;
        EventQueue::repeating_t pinging;

    public:
        Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> rdx=nullptr);
        void feed(const json11::Json& obj);
        void feed(const std::string& str);
};


#endif
