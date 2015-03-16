#include "babi.h"

using std::string;
using std::vector;
using std::make_shared;

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

    VS Parser::tasks() {
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
            for (std::string sub_task : {       "qa1_single-supporting-fact",
                                                "qa2_two-supporting-facts",
                                                "qa3_three-supporting-facts",
                                                "qa4_two-arg-relations",
                                                "qa5_three-arg-relations",
                                                "qa6_yes-no-questions",
                                                "qa7_counting",
                                                // "qa8_lists-sets",
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
                                                // "qa19_path-finding",
                                                "qa20_agents-motivations"}) {
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
            for (std::string sub_task : {       "qa1_single-supporting-fact",
                                                "qa2_two-supporting-facts",
                                                "qa3_three-supporting-facts",
                                                "qa4_two-arg-relations",
                                                "qa5_three-arg-relations",
                                                "qa6_yes-no-questions",
                                                "qa7_counting",
                                                // "qa8_lists-sets",
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
                                                // "qa19_path-finding",
                                                "qa20_agents-motivations"}) {
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


    double task_accuracy(std::shared_ptr<Model> m, const std::string& task) {
        int correct_questions = 0;
        int total_questions = 0;
        for (auto& story : Parser::testing_data(task)) {
            m->new_story();
            for (auto& item_ptr : story) {
                if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                    m->fact(f->fact);
                } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                    VS answer = m->question(qa->question);
                    if(utils::vs_equal(answer, qa->answer)) {
                        ++correct_questions;
                    }
                    ++total_questions;
                } else {
                    assert(NULL == "Unknown subclass of babi::Item");
                }
            }
        }

        double accuracy = 100.0*(double)correct_questions/total_questions;
        std::stringstream ss3;
        ss3 << task << " all done - accuracy on test set: "<< accuracy;
        ThreadPool::print_safely(ss3.str());
        return accuracy;
    }


    /* StoryParser */

    StoryParser::StoryParser(const Story* story) : story(story), story_idx(0) {
        advance();
    }

    void StoryParser::advance() {
        next_qa = NULL;
        while (story_idx < story->size()) {
            auto item_ptr = (*story)[story_idx++];
            if (Fact* f = dynamic_cast<Fact*>(item_ptr.get())) {
                facts_so_far.push_back(f->fact);
            } else if (QA* qa = dynamic_cast<QA*>(item_ptr.get())) {
                next_qa = qa;
                break;
            }
        }
    }

    sp_ret_t StoryParser::next() {
        sp_ret_t ret = std::make_tuple(facts_so_far, next_qa);
        advance();
        return ret;
    }

    bool StoryParser::done() {
        return next_qa == NULL;
    }
};


