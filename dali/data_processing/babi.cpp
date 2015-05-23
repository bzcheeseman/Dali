#include "babi.h"

#include <atomic>

#include "dali/utils.h"

using std::atomic;
using std::make_shared;
using std::string;
using std::vector;
using utils::random_minibatches;

namespace babi {
    void Item::to_stream(std::ostream& str) const {
        assert(NULL == "Item should be subclassed.");
    }


    std::ostream& operator<<(std::ostream& str, const Item& data) {
        data.to_stream(str);
        return str;
    }

    QA::QA(const VS question,
           const VS answer,
           const std::vector<int> supporting_facts) :
            question(question),
            answer(answer),
            supporting_facts(supporting_facts) {
    }

    void QA::to_stream(std::ostream& str) const {
        str << utils::join(question, " ")   << "\t";
        str << utils::join(answer,",") << "\t";
        vector<string> s;
        std::transform(supporting_facts.begin(), supporting_facts.end(),
                       std::back_inserter(s), [](int x) {
            return std::to_string(x);
        });
        str << utils::join(s, " ");
    }

    Fact::Fact(const VS fact) :
        fact(fact) {
    }

    void Fact::to_stream(std::ostream& str) const {
        str << utils::join(fact, " ");
    }

    /* Parser */

    vector<Story> Parser::parse_file(const string& filename,
                                     const int& num_questions) {
        // std::cout << file << std::endl;
        // auto f = make_shared<Fact>(vector<string>({"here", "i", "am"}));
        // auto q = make_shared<QA>(vector<string>({"where","is","wally"}),
        //                                vector<string>{"kitchen"},
        //                                vector<int>({0}));
        // Story story = {f, q};
        // vector<Story> res = { story };
        // return res;

        if (!utils::file_exists(filename)) {
            std::stringstream error_msg;
            error_msg << "Error: File \"" << filename << "\" does not exist, cannot parse file.";
            throw std::runtime_error(error_msg.str());
        }

        std::ifstream file(filename);
        // file exists
        assert(file.good());

        vector<Story> result;
        Story current_story;

        int last_story_id = -1, story_id;
        int questions_so_far = 0;

        string line_buffer;
        while(std::getline(file, line_buffer)) {
            // Read story id. Non-increasing id is indication
            // of new story.
            std::stringstream line(line_buffer);

            line >> story_id;
            if (last_story_id != -1 && last_story_id >= story_id) {
                if (questions_so_far >= num_questions)
                    break;
                result.push_back(current_story);
                current_story.clear();
            }
            last_story_id = story_id;

            // Parse question or fact.
            vector<string> tokens;
            bool is_question;

            while(true) {
                string token;
                line >> token;
                assert(!token.empty());
                char lastc = token[token.size()-1];
                if (lastc == '.') {
                    tokens.push_back(token.substr(0, token.size()-1));
                    tokens.push_back(".");
                    is_question = false;
                    break;
                } else if (lastc == '?') {
                    tokens.push_back(token.substr(0, token.size()-1));
                    tokens.push_back("?");
                    questions_so_far += 1;
                    is_question = true;
                    break;
                } else {
                    tokens.push_back(token);
                }
            }
            if (is_question) {
                string comma_separated_answer;
                line >> comma_separated_answer;

                vector<string> answer = utils::split(comma_separated_answer, ',');
                vector<int> supporting_facts;
                int supporting_fact;
                while(line >> supporting_fact) {
                    // make it 0 indexed.
                    supporting_facts.push_back(supporting_fact - 1);
                }
                current_story.push_back(make_shared<QA>(tokens,
                                                        answer,
                                                        supporting_facts));
            } else {
                current_story.push_back(make_shared<Fact>(tokens));
            }
        }
        result.push_back(current_story);


        return result;
    }

    string Parser::data_dir() {
        return utils::dir_join({ STR(DALI_DATA_DIR), "babi", "babi" });
    }

    vector<string> tasks() {
        // TODO read from disk.
        return {
            "qa1_single-supporting-fact",
            "qa2_two-supporting-facts",
            "qa3_three-supporting-facts",
            "qa4_two-arg-relations",
            "qa5_three-arg-relations",
            "qa6_yes-no-questions",
            "qa7_counting",
            "qa8_lists-sets",
            "qa9_simple-negation",
            "qa10_indefinite-knowledge",
            "qa11_basic-coreference",
            "qa12_conjunction",
            "qa13_compound-coreference",
            "qa14_time-reasoning",
            "qa15_basic-deduction",
            "qa16_basic-induction",
            "qa17_positional-reasoning",
            "qa18_size-reasoning",
            "qa19_path-finding",
            "qa20_agents-motivations"
        };
    }

    vector<Story> Parser::training_data(const string& task,
                                        const int& num_questions,
                                        bool shuffled) {
        if (task == "multitasking") {
            vector<Story> stories;
            for (std::string sub_task : tasks()) {
                auto task_stories = training_data(sub_task, num_questions, shuffled);
                stories.insert(stories.end(), task_stories.begin(), task_stories.end());
            }
            return stories;
        }
        string filename = utils::join({task, "_train.txt"});
        string filepath = utils::dir_join({data_dir(),
                                           shuffled ? "shuffled" : "en",
                                           filename});
        return parse_file(filepath, num_questions);
    }

    vector<Story> Parser::testing_data(const string& task,
                         const int& num_questions,
                         bool shuffled) {
        if (task == "multitasking") {
            vector<Story> stories;
            for (std::string sub_task : tasks()) {
                auto task_stories = testing_data(sub_task, num_questions, shuffled);
                stories.insert(stories.end(), task_stories.begin(), task_stories.end());
            }
            return stories;
        }

        string filename = utils::join({task, "_test.txt"});
        string filepath = utils::dir_join({data_dir(),
                                           shuffled ? "shuffled" : "en",
                                           filename});
        return parse_file(filepath, num_questions);
    }

    utils::Generator<sp_ret_t> story_parser(const Story& story) {
        return utils::make_generator<sp_ret_t>([&story](utils::yield_t<sp_ret_t> yield) {
            int story_idx = 0;
            vector<vector<string>> facts_so_far;
            while (story_idx < story.size()) {
                auto item_ptr = story[story_idx++];
                if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                    facts_so_far.push_back(f->fact);
                } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                    yield(std::make_tuple(facts_so_far, qa));
                }
            }
        });
    }

    /* helper functions */

    double task_accuracy(const std::string& task, prediction_fun predict, int num_threads) {
        return accuracy(Parser::testing_data(task), predict, num_threads);
    }

    double accuracy(const vector<Story>& data, prediction_fun predict, int num_threads) {
        ThreadPool pool(num_threads);

        atomic<int> correct_questions(0);
        atomic<int> total_questions(0);

        auto batches = random_minibatches(data.size(), num_threads);

        for (int bidx = 0; bidx < batches.size(); ++bidx) {
            pool.run([&batches, &data, &correct_questions, &total_questions,
                      &predict, bidx]() {
                auto batch = batches[bidx];
                for (auto& story_idx : batch) {
                    auto& story = data[story_idx];
                    vector<VS> facts_so_far;
                    QA* qa;
                    for (auto story : story_parser(story)) {
                        std::tie(facts_so_far, qa) = story;
                        VS answer = predict(facts_so_far, qa->question);
                        if(utils::vs_equal(answer, qa->answer))
                            ++correct_questions;
                        ++total_questions;
                    }
                }
            });
        }

        pool.wait_until_idle();

        double accuracy = (double)correct_questions/total_questions;

        return accuracy;
    }

    vector<string> vocab_from_data(const vector<babi::Story>& data) {
        std::set<string> vocab_set;
        for (auto& story : data) {
            for(auto& item_ptr : story) {
                if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                    vocab_set.insert(f->fact.begin(), f->fact.end());
                } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                    vocab_set.insert(qa->question.begin(), qa->question.end());
                    vocab_set.insert(qa->answer.begin(), qa->answer.end());
                }
            }
        }
        vector<string> vocab_as_vector;
        vocab_as_vector.insert(vocab_as_vector.end(), vocab_set.begin(), vocab_set.end());
        return vocab_as_vector;
    }

    void compare_results(std::vector<double> our_results) {
        std::vector<double> facebook_lstm_results = {
            50,  20,  20,  61,  70, 48,  49, 45, 64,  44, 72,  74,  94,  27, 21,  23,  51, 52, 8,  91
        };

        std::vector<double> facebook_best_results = {
            100, 100, 100, 100, 98, 100, 85, 91, 100, 98, 100, 100, 100, 99, 100, 100, 65, 95, 36, 100
        };

        std::vector<double> facebook_multitask_results = {
            100, 100, 98,  80,  99, 100, 86, 93, 100, 98, 100, 100, 100, 99, 100, 94,  72, 93, 19, 100
        };

        std::cout << "Babi Benchmark Results" << std::endl
                  << std::endl
                  << "Columns are (from left): task name, our result," << std::endl
                  << "Facebook's LSTM result, Facebook's best result," << std::endl
                  << "Facebook's multitask result." << std::endl
                  << "===============================================" << std::endl;

        std::function<std::string(std::string)> moar_whitespace = [](std::string input,
                                                                     int num_whitespace=30) {
            assert(input.size() < num_whitespace);
            std::stringstream ss;
            ss << input;
            num_whitespace -= input.size();
            while(num_whitespace--) ss << ' ';
            return ss.str();
        };
        auto task_list = tasks();
        for (int task_id = 0; task_id < task_list.size(); ++task_id) {
            char special = ' ';
            if (our_results[task_id] >= facebook_lstm_results[task_id]) special = '*';
            if (our_results[task_id] >= facebook_best_results[task_id]) special = '!';


            std::cout << std::setprecision(1) << std::fixed
                      << moar_whitespace(task_list[task_id]) << "\t"
                      << special << our_results[task_id] << special << "\t"
                      << facebook_lstm_results[task_id] << "\t"
                      << facebook_best_results[task_id] << "\t"
                      << facebook_multitask_results[task_id] << std::endl;
        }
    }


};


