#include "visualizer.h"
#include "dali/utils/core_utils.h"

using utils::MS;
using std::string;
using json11::Json;
using utils::assert2;

using namespace std::placeholders;

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port, 6379, "Default port to be used by visualizer.");
DEFINE_string(visualizer, "", "What to name the visualization job.");

namespace visualizable {

    typedef std::shared_ptr<Visualizable> visualizable_ptr;
    typedef std::shared_ptr<ClassifierExample> classifier_example_ptr;

    template<typename R>
    std::vector<std::shared_ptr<Sentence<R>>> sentence_vector(const std::vector<std::vector<std::string>>& vec) {
        std::vector<std::shared_ptr<Sentence<R>>> res;
        for (auto& sentence: vec) {
            res.push_back(std::make_shared<Sentence<R>>(sentence));
        }
        return res;
    }

    template std::vector<std::shared_ptr<Sentence<float>>> sentence_vector(const std::vector<std::vector<std::string>>& vec);
    template std::vector<std::shared_ptr<Sentence<double>>> sentence_vector(const std::vector<std::vector<std::string>>& vec);

    /** Sentence **/

    template<typename R>
    Sentence<R>::Sentence(std::vector<std::string> tokens) : tokens(tokens) {}

    template<typename R>
    void Sentence<R>::set_weights(const std::vector<R>& _weights) {
        weights = _weights;
    }

    template<typename R>
    void Sentence<R>::set_weights(const Mat<R>& _weights) {
        weights = std::vector<R>(
            _weights.w().data(),
            _weights.w().data() + _weights.number_of_elements());
    }

    template<typename R>
    json11::Json Sentence<R>::to_json() {
        return Json::object {
            { "type", "sentence" },
            { "weights", weights },
            { "words", tokens },
        };
    }

    template class Sentence<float>;
    template class Sentence<double>;

    /** Sentences **/

    template<typename R>
    Sentences<R>::Sentences(std::vector<sentence_ptr> sentences) : sentences(sentences) {
    }
    template<typename R>
    Sentences<R>::Sentences(std::vector<std::vector<std::string>> vec) : sentences(sentence_vector<R>(vec)) {
    }

    template<typename R>
    void Sentences<R>::set_weights(const std::vector<R>& _weights) {
        weights = _weights;
    }

    template<typename R>
    void Sentences<R>::set_weights(const Mat<R>& _weights) {
        weights = std::vector<R>(
            _weights.w().data(),
            _weights.w().data() + _weights.number_of_elements());
    }

    template<typename R>
    json11::Json Sentences<R>::to_json() {
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

    template class Sentences<float>;
    template class Sentences<double>;

    /** QA **/

    template<typename R>
    QA<R>::QA(visualizable_ptr context, sentence_ptr question, sentence_ptr answer) :
        context(context), question(question), answer(answer) {
    }

    template<typename R>
    json11::Json QA<R>::to_json() {
        return Json::object {
            { "type", "qa"},
            { "context", context->to_json()},
            { "question", question->to_json()},
            { "answer", answer->to_json()},
        };
    }

    template class QA<float>;
    template class QA<double>;

    /** ClassifierExample **/

    ClassifierExample::ClassifierExample(visualizable_ptr input, visualizable_ptr output) :
            input(input), output(output) {
    }

    json11::Json ClassifierExample::to_json() {
        return Json::object {
            { "type", "classifier_example"},
            { "input", input->to_json()},
            { "output", output->to_json()},
        };
    }

    /** Finite Distribution **/
    template<typename R>
    FiniteDistribution<R>::FiniteDistribution(
                           const std::vector<R>& distribution,
                           const std::vector<R>& scores,
                           const std::vector<std::string>& labels,
                           int max_top_picks) :
        distribution(distribution),
        scores(scores),
        labels(labels) {
        assert2(labels.size() == distribution.size(),
                "FiniteDistribution visualizer: sizes of labels and distribution differ");
        if (max_top_picks > 0) {
            top_picks = std::min(max_top_picks, (int)distribution.size());
        } else {
            top_picks = distribution.size();
        }
    }

    template<typename R>
    FiniteDistribution<R>::FiniteDistribution(const std::vector<R>& distribution,
               const std::vector<std::string>& labels,
               int max_top_picks) :
        FiniteDistribution(
            distribution,
            empty_vec,
            labels,
            max_top_picks) {}

    template<typename R>
    json11::Json FiniteDistribution<R>::to_json() {
        std::vector<std::string> output_labels(top_picks);
        std::vector<double> output_probs(top_picks);
        std::vector<double> output_scores(top_picks);

        // Pick top k best answers;

        std::vector<bool> taken(distribution.size());
        for(int idx = 0; idx < distribution.size(); ++idx) taken[idx] = false;

        for(int iters = 0; iters < top_picks; ++iters) {
            int best_index = -1;
            for (int i=0; i < distribution.size(); ++i) {
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

    template class FiniteDistribution<float>;
    template class FiniteDistribution<double>;

    template<typename R>
    const std::vector<R> FiniteDistribution<R>::empty_vec;

    // template<> const std::vector<double> FiniteDistribution<double>::empty_vec;
    // template<> const std::vector<float> FiniteDistribution<float>::empty_vec;

}

void Visualizer::connected_callback(int status) {
    rdx_state.store(status);
}

bool Visualizer::ensure_connection() {
    std::unique_lock<std::mutex> guard(connection_mutex);
    if (rdx_state.load() != redox::Redox::CONNECTED &&
            rdx_state.load() != redox::Redox::NOT_YET_CONNECTED) {
        rdx.reset();
        rdx = std::make_shared<redox::Redox>(std::cout, redox::log::Off);

        rdx_state.store(redox::Redox::NOT_YET_CONNECTED);

        rdx->connect(FLAGS_visualizer_hostname,
                 FLAGS_visualizer_port,
                 std::bind(&Visualizer::connected_callback, this, _1));
    }
    return rdx_state.load() == redox::Redox::CONNECTED;
}

Visualizer::Visualizer(std::string my_namespace,
                       bool rename_if_needed) :
        my_namespace(my_namespace),
        rename_if_needed(rename_if_needed),
        rdx_state(redox::Redox::DISCONNECTED) {


    // then we ping the visualizer regularly:
    pinging = eq.run_every([this]() {
        if (!ensure_connection()) {
            name_initialized = false;
            return;
        }

        if (!name_initialized)
            name_initialized = update_name();

        if (!name_initialized)
            return;

        // Expire at 2 seconds
        std::string namespace_key = MS() << "namespace_" << this->my_namespace;
        rdx->command<string>({"SET", namespace_key, "1", "EX", "2"});

    }, std::chrono::seconds(1));
}

bool Visualizer::update_name() {
    int increment = 0;

    while (true) {
        std::string namespace_key;
        if (increment == 0) {
            namespace_key = MS() << "namespace_" << this->my_namespace;
        } else {
            namespace_key = MS() << "namespace_" << this->my_namespace << "_" << increment;
        }
        bool taken;
        try {
            auto& key_exists = rdx->commandSync<int>({"EXISTS", namespace_key});
            if (!key_exists.ok())
                return false;
            taken = (key_exists.reply() == 1);
            key_exists.free();
        } catch (std::runtime_error e) {
            return false;
        }

        if(taken) {
            if (rename_if_needed) {
                std::cout << "Duplicate Visualizer name : \"" << this->my_namespace  << "_" << increment
                          << "\". Retrying with \"" << this->my_namespace << "_" << increment + 1 << "\"" << std::endl;
                increment++;
                continue;
            } else {
                // give up immediately on renaming
                // and throw duplicate name error:
                throw Visualizer::duplicate_name_error(MS() << "VISUALIZER ERROR: visualizer name already in use: " << my_namespace);
            }
        } else {
            if (increment != 0)
                my_namespace = MS() << my_namespace << "_" << increment;
            break;
        }
    }
    return true;
}


void Visualizer::feed(const json11::Json& obj) {
    if (!ensure_connection())
        return;

    rdx->publish(MS() << "feed_" << my_namespace, obj.dump());
}

void Visualizer::feed(const std::string& str) {
    Json str_as_json = Json::object {
        { "type", "report" },
        { "data", str },
    };
    feed(str_as_json);
}

void Visualizer::throttled_feed(Throttled::Clock::duration time_between_feeds, std::function<json11::Json()> f) {
    throttle.maybe_run(time_between_feeds, [&f, this]() {
        feed(f());
    });
}

Visualizer::duplicate_name_error::duplicate_name_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
Visualizer::duplicate_name_error::duplicate_name_error(const char* what_arg) : std::runtime_error(what_arg) {}

