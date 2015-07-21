#include "babi.h"

using std::string;
using std::vector;
using utils::Vocab;

namespace babi {
    template<typename word_t>
    uint Story<word_t>::size() const {
        return question_fidx.size();
    }

    template<typename word_t>
    void Story<word_t>::print() const {
        for (int i = 0; i <facts.size(); ++i) {
            std::cout << i << " ";
            for (auto& word: facts[i]) {
                std::cout << word << " ";
            }

            int qidx = -1;
            for (int q = 0; q < question_fidx.size(); ++q) {
                if (i == question_fidx[q]) {
                    qidx = q;
                }
            }
            if (qidx != -1) {
                for (auto& word: answers[qidx]) {
                    std::cout << word << " ";
                }
                for (auto sf : supporting_facts[qidx]) {
                    std::cout << sf << " ";
                }
            }
            std::cout << std::endl;
        }

    }

    template<typename word_t>
    QA<word_t> Story<word_t>::get(int target_question_idx) const {
        QA<word_t> result;

        assert(0 <= target_question_idx &&
               target_question_idx < question_fidx.size());

        std::unordered_map<uint, uint> mapped_fidx;
        uint start_idx = 0;
        for (uint qidx = 0; qidx <= target_question_idx; ++qidx) {
            for (uint fidx = start_idx; fidx < question_fidx[qidx]; ++fidx) {
                result.facts.push_back(facts[fidx]);
                mapped_fidx[fidx] = result.facts.size() - 1;
            }
            start_idx = question_fidx[qidx] + 1;
        }
        for(auto supporting_fact : supporting_facts[target_question_idx]) {
            result.supporting_facts.emplace_back(mapped_fidx[supporting_fact]);
        }

        result.question = facts[question_fidx[target_question_idx]];
        result.answer = answers[target_question_idx];

        return result;
    }

    template class QA<uint>;
    template class QA<string>;
    template class Story<uint>;
    template class Story<string>;


    Story<string> decode(Story<uint> story, const Vocab& vocab, bool strip_eos) {
        Story<string> output;
        output.question_fidx = story.question_fidx;
        output.supporting_facts = story.supporting_facts;
        for (auto& fact: story.facts) {
            output.facts.emplace_back(vocab.decode(fact, strip_eos));
        }
        for (auto& answer: story.answers) {
            output.answers.emplace_back(vocab.decode(answer, strip_eos));
        }

        return output;
    }

    Story<uint> encode(Story<string> story, const Vocab& vocab, bool add_eos) {
        Story<uint> output;
        output.question_fidx = story.question_fidx;
        output.supporting_facts = story.supporting_facts;
        for (auto& fact: story.facts) {
            output.facts.emplace_back(vocab.encode(fact, add_eos));
        }
        for (auto& answer: story.answers) {
            output.answers.emplace_back(vocab.encode(answer, add_eos));
        }

        return output;
    }

    std::tuple<vector<Story<uint>>, Vocab> encode_dataset(
            const vector<Story<string>>& input, bool add_eos, uint min_occurence) {
        std::unordered_map<string, uint> occurence;
        for (auto& story : input) {
            for (auto& fact: story.facts) {
                for (auto& word: fact) {
                    occurence[word] += 1;
                }
            }
            for (auto& answer: story.answers) {
                for (auto& word: answer) {
                    occurence[word] += 1;
                }
            }
        }
        vector<string> words;

        for (auto kv : occurence) {
            if (kv.second >= min_occurence) {
                words.push_back(kv.first);
            }
        }
        words.push_back(utils::end_symbol);
        Vocab vocab(words, true);

        vector<Story<uint>> results;
        for (auto& story: input) {
            results.push_back(encode(story, vocab, add_eos));
        }
        return std::make_tuple(results, vocab);
    }

    static std::tuple<int, vector<std::string>> parse_fact(std::string fact) {
        std::stringstream line(fact);

        int line_no;
        vector<string> tokenized;

        line >> line_no;
        while(true) {
            string token;
            line >> token;
            if (token.empty())
                break;
            tokenized.push_back(token);
        }
        return std::make_tuple(line_no, tokenized);
    }

    static vector<std::string> parse_answer(std::string answer) {
        std::stringstream line(answer);

        vector<string> tokenized;

        while(true) {
            string token;
            line >> token;
            if (token.empty())
                break;
            tokenized.push_back(token);
        }
        return tokenized;
    }

    static vector<uint> parse_supporting(std::string sup) {
        std::stringstream line(sup);

        vector<uint> supporting;

        while(true) {
            string token;
            line >> token;
            if (token.empty())
                break;
            supporting.push_back(utils::from_string<uint>(token) - 1);
        }
        return supporting;
    }

    vector<Story<string>> parse_file(const string& filename) {
        if (!utils::file_exists(filename)) {
            std::stringstream error_msg;
            error_msg << "Error: File \"" << filename << "\" does not exist, cannot parse file.";
            throw std::runtime_error(error_msg.str());
        }

        std::ifstream file(filename);
        // file exists
        assert(file.good());

        vector<Story<string>> results;
        string line_buffer;
        int last_line_no = -1;

        while(std::getline(file, line_buffer)) {
            // Read story id. Non-increasing id is indication
            // of new story.
            std::stringstream line(line_buffer);
            vector<string> chunks;
            while(true) {
                string chunk;
                std::getline(line, chunk, '\t');
                if (chunk.empty()) {
                    break;
                } else {
                    chunks.push_back(chunk);
                }
            }
            utils::assert2(chunks.size() == 1 || chunks.size() == 3, "Error parsing QA file.");

            int line_number;
            vector<string> fact;

            std::tie(line_number, fact) = parse_fact(chunks[0]);

            line >> line_number;
            if (last_line_no == -1 || line_number <= last_line_no) {
                results.emplace_back();
            }
            last_line_no = line_number;

            // store the fact.
            results.back().facts.push_back(fact);

            bool is_question = chunks.size() == 3;

            // if this is a question store its index.
            if (is_question) {
                results.back().question_fidx.push_back(
                        results.back().facts.size() - 1);
            }

            // if this ia question do read in the answer and supporting facts.
            if (is_question) {
                auto answer = parse_answer(chunks[1]);
                results.back().answers.push_back(answer);

                auto supporting = parse_supporting(chunks[2]);

                results.back().supporting_facts.push_back(supporting);
            }
        }

        return results;
    }

    string data_dir() {
        return utils::dir_join({ STR(DALI_DATA_DIR), "babi", "tasks" });
    }


    std::vector<Story<std::string>> dataset(std::string task_prefix,
                                            std::string train_or_test,
                                            std::string dataset_type) {
        auto dataset_path = utils::dir_join({
            data_dir(),
            dataset_type,
            utils::prefix_match(tasks(), task_prefix) + "_" + train_or_test + ".txt"
        });
        assert2(utils::file_exists(dataset_path),
                utils::MS() << "File " << dataset_path << " does not exist.");
        return parse_file(dataset_path);
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

}
