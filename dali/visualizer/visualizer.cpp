#include "visualizer.h"

#include <memory>
#include <future>

#include "dali/utils/core_utils.h"

using utils::MS;
using std::string;
using json11::Json;
using utils::assert2;

using namespace std::placeholders;
using std::vector;
using std::shared_ptr;
using std::chrono::milliseconds;

DEFINE_string(visualizer_hostname, "127.0.0.1", "Default hostname to be used by visualizer.");
DEFINE_int32(visualizer_port, 6379, "Default port to be used by visualizer.");
DEFINE_string(visualizer, "", "What to name the visualization job.");

namespace visualizable {

    typedef std::shared_ptr<Visualizable> visualizable_ptr;
    typedef std::shared_ptr<GridLayout> grid_layout_ptr;

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
            { "spaces", this->spaces },
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
    ParallelSentence<R>::ParallelSentence(sentence_ptr sentence1, sentence_ptr sentence2) :
            sentence1(sentence1), sentence2(sentence2) {
    }

    template<typename R>
    json11::Json ParallelSentence<R>::to_json() {
        return Json::object {
            { "type", "parallel_sentence" },
            { "sentence1", sentence1->to_json()},
            { "sentence2", sentence2->to_json()}
        };
    };

    template class ParallelSentence<float>;
    template class ParallelSentence<double>;

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

    /** GridLayout **/

    void GridLayout::add_in_column(int column, visualizable_ptr vis) {
        while (grid.size() <= column)
            grid.emplace_back();
        grid[column].push_back(vis);
    }

    json11::Json GridLayout::to_json() {
        vector<vector<json11::Json>> grid_as_json;
        for (auto& column: grid) {
            vector<json11::Json> column_contents;
            for(auto& vis: column) {
                column_contents.push_back(vis->to_json());
            }
            grid_as_json.push_back(column_contents);
        }
        return Json::object {
            { "type", "grid_layout"},
            { "grid", grid_as_json},
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

    template<typename T>
    Probability<T>::Probability(T probability) : probability(probability) {
    }

    template<typename T>
    json11::Json Probability<T>::to_json() {
        return Json::object {
            { "type", "probability"},
            { "probability", probability },
        };
    }

    template class Probability<float>;
    template class Probability<double>;

    Message::Message(std::string content) : content(content) {
    }

    json11::Json Message::to_json() {
        return Json::object {
            { "type", "message"},
            { "content", content },
        };
    }

    Tree::Tree(string label) :
            label(label) {
    }

    Tree::Tree(std::initializer_list<std::shared_ptr<Tree>> children) :
            children(vector<std::shared_ptr<Tree>>(children)) {
    }

    Tree::Tree(vector<std::shared_ptr<Tree>> children) :
            children(children) {
    }

    Tree::Tree(string label, std::initializer_list<shared_ptr<Tree>> children) :
            label(label),
            children(vector<std::shared_ptr<Tree>>(children)) {
    }


    Tree::Tree(string label, vector<shared_ptr<Tree>> children) :
            label(label),
            children(children) {
    }

    json11::Json Tree::to_json() {
        vector<json11::Json> children_as_json;
        std::transform(children.begin(), children.end(), std::back_inserter(children_as_json),
                [this](shared_ptr<Tree> child) {
            auto child_json = child->to_json();
            return child_json;
        });

        if (label.empty()) {

            return Json::object {
                { "type", "tree" },
                { "children", children_as_json},
            };
        } else {
            return Json::object {
                { "type", "tree" },
                { "label", label },
                { "children", children_as_json},
            };
        }
    }
}

#ifdef DALI_USE_VISUALIZER
    bool Visualizer::ensure_connection() {
        try {
            std::unique_lock<std::mutex> guard(connection_mutex);
            if (rdx_state.load() != redox::Redox::CONNECTED &&
                    rdx_state.load() != redox::Redox::NOT_YET_CONNECTED) {
                rdx.reset();

                rdx = std::make_shared<redox::Redox>(std::cout, redox::log::Off);


                rdx_state.store(redox::Redox::NOT_YET_CONNECTED);

                rdx->connect(FLAGS_visualizer_hostname,
                         FLAGS_visualizer_port,
                         std::bind(&Visualizer::rdx_connected_callback, this, _1));
            }

            if (callcenter_state.load() != redox::Redox::CONNECTED &&
                    callcenter_state.load() != redox::Redox::NOT_YET_CONNECTED) {
                subscription_active = false;
                callcenter_main_phoneline.reset();
                callcenter_main_phoneline =
                        std::make_shared<redox::Subscriber>(std::cout, redox::log::Off);
                callcenter_main_phoneline->connect(
                    FLAGS_visualizer_hostname,
                    FLAGS_visualizer_port,
                    std::bind(&Visualizer::callcenter_connected_callback, this, _1));
            }

            return rdx_state.load() == redox::Redox::CONNECTED;
        } catch (std::system_error) {
            return false;
        }
    }

    void Visualizer::rdx_connected_callback(int status) {
        rdx_state.store(status);
    }

    void Visualizer::callcenter_connected_callback(int status) {
        callcenter_state.store(status);
    }

    Visualizer::Visualizer(std::string name) :
            my_uuid(sole::uuid4().str()),
            my_name(name),
            running(true),
            rdx_state(redox::Redox::DISCONNECTED),
            callcenter_state(redox::Redox::DISCONNECTED)  {
        // then we ping the visualizer regularly:

        register_function("whoami", std::bind(&Visualizer::whoami, this, _1, _2));
        ping_thread = std::make_shared<std::thread>(&Visualizer::ping, this);
    }
    Visualizer::~Visualizer() {
        running = false;
        if (ping_thread != nullptr) {
            ping_thread->join();
        }
        if (callcenter_state == redox::Redox::CONNECTED) {
            callcenter_main_phoneline->disconnect();
        }
        if (rdx_state == redox::Redox::CONNECTED) {
            rdx->disconnect();
        }
    }

    void Visualizer::ping() {
        while (running) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            if (!ensure_connection()) {
                return;
            }

            subscription_active = subscription_active && verify_subscription_active();

            if (!subscription_active && callcenter_state == redox::Redox::CONNECTED) {
                update_subscriber();
            }

            feed(Json::object {
                { "type", "heartbeat" },
            });
        }
    }

    void Visualizer::whoami(std::string fname, json11::Json ignored) {
        feed(Json::object {
                { "type", "whoami" },
                { "name", my_name},
        });
    }

    bool Visualizer::verify_subscription_active() {
        auto requests_namespace = "callcenter_" + this->my_uuid;

        for (auto& topic: callcenter_main_phoneline->subscribedTopics()) {
            if (topic == requests_namespace) {
                return true;
            }
        }
        return true;
    }

    void Visualizer::update_subscriber() {
        // if we previously listened for requests for a different name, then
        // we stop
                std::this_thread::sleep_for(milliseconds(1000));

        for (auto &topic : callcenter_main_phoneline->subscribedTopics()) {
            callcenter_main_phoneline->unsubscribe(topic);
        }
        auto requests_namespace = "callcenter_" + this->my_uuid;
        // get ready to handle incoming requests:
        callcenter_main_phoneline->subscribe(requests_namespace,
                [this, requests_namespace](const string& topic, const string& msg) {
            std::lock_guard<std::mutex> guard(this->callcenter_mutex);

            assert2(topic == requests_namespace,
                    "Visualizer: Received message from unexpected channel.");

            std::string error;
            auto msg_json = json11::Json::parse(msg, error);

            if (!error.empty()) {
                std::cout << "VISUALIZER WARNING: error in requestion json: " << error << std::endl;
                return;
            }

            if (msg_json["name"].is_null()) {
                std::cout << "VISUALIZER WARNING: received request without function name." << std::endl;
                return;
            }
            auto& name = msg_json["name"].string_value();
            if (this->callcenter_name_to_lambda.find(name) == this->callcenter_name_to_lambda.end()) {
                std::cout << "VISUALIZER WARNING: Requested function <" << name << "> not supported (did you forget to register?)." << std::endl;
                return;
            }

            auto& f = this->callcenter_name_to_lambda.at(name);

            f(name, msg_json["payload"]);

        });
        subscription_active = true;
    }

    void Visualizer::register_function(std::string name, function_t lambda) {
        std::lock_guard<std::mutex> guard(callcenter_mutex);
        callcenter_name_to_lambda[name] = lambda;
    }


    void Visualizer::feed(const json11::Json& obj) {
        if (!ensure_connection())
            return;

        rdx->publish(MS() << "updates_" << my_uuid, obj.dump());
    }

    void Visualizer::feed(const std::string& str) {
        Json str_as_json = Json::object {
            { "type", "report" },
            { "data", str },
        };
        feed(str_as_json);
    }

    void Visualizer::throttled_feed(Throttled::Clock::duration time_between_feeds,
                                    std::function<json11::Json()> f) {
        throttle.maybe_run(time_between_feeds, [&f, this]() {
            feed(f());
        });
    }

#else
    bool Visualizer::verify_subscription_active() {return false;}
    void Visualizer::ping() {}
    bool Visualizer::ensure_connection() { return false; }
    void Visualizer::rdx_connected_callback(int status) {}
    void Visualizer::callcenter_connected_callback(int status) {}
    Visualizer::Visualizer(std::string my_name) : running(false) {
        std::cout << "WARNING: Dali was compiled without visualizer - Visualizer class won't work very well." << std::endl;
    }
    void Visualizer::register_function(std::string name, function_t lambda) {}
    void Visualizer::whoami(std::string fname, json11::Json ignored) {}
    void Visualizer::update_subscriber() { }
    void Visualizer::feed(const json11::Json& obj) {}
    void Visualizer::feed(const std::string& str) {}
    void Visualizer::throttled_feed(Throttled::Clock::duration time_between_feeds, std::function<json11::Json()> f) {}
#endif


