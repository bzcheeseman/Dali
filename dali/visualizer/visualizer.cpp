#include "visualizer.h"
#include "dali/utils/core_utils.h"

using utils::MS;
using std::string;
using json11::Json;
using utils::assert2;

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port, 6379, "Default port to be used by visualizer.");

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
Visualizer::Visualizer(std::string my_namespace, std::shared_ptr<redox::Redox> other_rdx) : Visualizer(my_namespace, false, other_rdx) {}

Visualizer::Visualizer(std::string my_namespace, bool rename_if_needed, std::shared_ptr<redox::Redox> other_rdx) :
        my_namespace(my_namespace) {
    if(!other_rdx) {
        // if no redox connection was passed we create
        // one on the fly:
        rdx = std::make_shared<redox::Redox>(std::cout, redox::log::Off);
        // next we try to connect:
        std::cout << "Visualizer connecting to redis "
                  << FLAGS_visualizer_hostname << ":" << FLAGS_visualizer_port << std::endl;
        connected = rdx->connect(FLAGS_visualizer_hostname, FLAGS_visualizer_port);
    } else {
        rdx = other_rdx;
        connected = true;
    }
    if (!connected) {
        throw std::runtime_error("VISUALIZER ERROR: can't connect to redis.");
        return;
    }

    std::cout << "my_namespace = " << this->my_namespace << std::endl;
    std::string namespace_key = MS() << "namespace_" << this->my_namespace;
    auto& key_exists = rdx->commandSync<int>({"EXISTS", namespace_key});
    assert(key_exists.ok());
    bool taken = key_exists.reply() == 1;
    key_exists.free();
    if (taken) {
        if (rename_if_needed) {
            int increment = 1;
            if (!other_rdx) {
                std::cout << "Duplicate Visualizer name : \"" << this->my_namespace
                          << "\". Retrying with \"" << this->my_namespace << "_" << increment << "\"" << std::endl;
            }
            // iterate until a suitable name is found
            while (true) {
                if (increment > 500) {
                    throw Visualizer::duplicate_name_error(MS() << "VISUALIZER ERROR: visualizer name already in use: " << my_namespace);
                }
                namespace_key = MS() << "namespace_" << this->my_namespace << "_" << increment;
                auto& alternate_key_exists = rdx->commandSync<int>({"EXISTS", namespace_key});
                assert(alternate_key_exists.ok());
                taken = alternate_key_exists.reply() == 1;
                alternate_key_exists.free();
                if (taken) {
                    if (!other_rdx) {
                        std::cout << "Duplicate Visualizer name : \"" << this->my_namespace  << "_" << increment
                                  << "\". Retrying with \"" << this->my_namespace << "_" << increment + 1 << "\"" << std::endl;
                    }
                    increment++;
                    continue;
                } else {
                    // make note of new namespace:
                    this->my_namespace = MS() << "_" << increment;
                    break;
                }
            }
        } else {
            // give up immediately on renaming
            // and throw duplicate name error:
            throw Visualizer::duplicate_name_error(MS() << "VISUALIZER ERROR: visualizer name already in use: " << my_namespace);
        }
    }
    // then we ping the visualizer regularly:
    pinging = eq.run_every([this, namespace_key]() {
        if (!connected) return;
        // Expire at 2 seconds
        rdx->command<string>({"SET", namespace_key, "1", "EX", "2"});
    }, std::chrono::seconds(1));
}

void Visualizer::feed(const json11::Json& obj) {
    if (!connected) return;
    rdx->publish(MS() << "feed_" << my_namespace, obj.dump());
}

void Visualizer::feed(const std::string& str) {
    Json str_as_json = Json::object {
        { "type", "report" },
        { "data", str },
    };
    feed(str_as_json);
}

Visualizer::duplicate_name_error::duplicate_name_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
Visualizer::duplicate_name_error::duplicate_name_error(const char* what_arg) : std::runtime_error(what_arg) {}

