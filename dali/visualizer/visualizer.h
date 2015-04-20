#ifndef DALI_VISUALIZER_VISUALIZER_H
#define DALI_VISUALIZER_VISUALIZER_H

#include <gflags/gflags.h>
#include <json11.hpp>
#include <redox.hpp>
#include <memory>
#include <chrono>
#include <functional>
#include <string>

#include "dali/visualizer/EventQueue.h"
#include "dali/utils/core_utils.h"
#include "dali/mat/Mat.h"

// to import Throttled
#include "dali/utils/Reporting.h"

DECLARE_string(visualizer_hostname);
DECLARE_int32(visualizer_port);
DECLARE_string(visualizer);

// TODO: Szymon explain how this works
namespace visualizable {

    struct Visualizable {
        virtual json11::Json to_json() = 0;
    };

    template<typename R>
    struct Sentence : public Visualizable {
        std::vector<std::string> tokens;
        std::vector<R> weights;

        Sentence(std::vector<std::string> tokens);

        void set_weights(const std::vector<R>& _weights);
        void set_weights(const Mat<R>& _weights);

        virtual json11::Json to_json() override;
    };

    template<typename R>
    std::vector<std::shared_ptr<Sentence<R>>> sentence_vector(const std::vector<std::vector<std::string>>& vec);

    template<typename R>
    struct Sentences : public Visualizable {
        typedef std::shared_ptr<Sentence<R>> sentence_ptr;
        std::vector<sentence_ptr> sentences;
        std::vector<R> weights;

        Sentences(std::vector<sentence_ptr> sentences);
        Sentences(std::vector<std::vector<std::string>> vec);

        void set_weights(const std::vector<R>& _weights);
        void set_weights(const Mat<R>& _weights);

        virtual json11::Json to_json() override;
    };

    template<typename R>
    struct QA : public Visualizable {
        typedef std::shared_ptr<Sentence<R>> sentence_ptr;
        typedef std::shared_ptr<Visualizable> visualizable_ptr;
        visualizable_ptr context;
        sentence_ptr question;
        sentence_ptr answer;

        QA(visualizable_ptr context, sentence_ptr question, sentence_ptr answer);

        virtual json11::Json to_json() override;
    };

    struct ClassifierExample : public Visualizable {
        typedef std::shared_ptr<Visualizable> visualizable_ptr;

        visualizable_ptr input;
        visualizable_ptr output;

        ClassifierExample(visualizable_ptr input, visualizable_ptr output);

        virtual json11::Json to_json() override;
    };

    template<typename R>
    struct FiniteDistribution : public Visualizable {
        static const std::vector<R> empty_vec;
        std::vector<R> distribution;
        std::vector<R> scores;
        std::vector<std::string> labels;
        int top_picks;

        FiniteDistribution(const std::vector<R>& distribution,
                           const std::vector<R>& scores,
                           const std::vector<std::string>& labels,
                           int max_top_picks = -1);

        FiniteDistribution(const std::vector<R>& distribution,
               const std::vector<std::string>& labels,
               int max_top_picks = -1);

        virtual json11::Json to_json() override;
    };
}

// TODO: explain what this does
class Visualizer {
    private:
        std::string my_namespace;
        std::shared_ptr<redox::Redox> rdx;
        EventQueue eq;
        EventQueue::repeating_t pinging;
        Throttled throttle;

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
        Visualizer(std::string my_namespace, bool rename_if_needed, std::shared_ptr<redox::Redox> rdx=nullptr);

        void feed(const json11::Json& obj);
        void feed(const std::string& str);
        void throttled_feed(Throttled::Clock::duration time_between_feeds, std::function<json11::Json()> f);
};

#endif
