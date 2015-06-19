#include "babi2.h"

using std::string;
using std::vector;
using utils::Vocab;

namespace babi {
    template<typename word_t>
    uint Story<word_t>::size() {
        return question_fidx.size();
    }

    template<typename word_t>
    QA<word_t> Story<word_t>::get(int target_question_idx) {
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


    std::tuple<Story<uint>, Vocab> encode_dataset(Story<string> input) {
        vector<string> words;
        for (auto& fact: input.facts) {
            for (auto& word: fact) {
                words.push_back(word);
            }
        }
        for (auto& answer: input.answers) {
            for (auto& word: answer) {
                words.push_back(word);
            }
        }
        std::sort(words.begin(), words.end());
        words.erase(std::unique(words.begin(), words.end()), words.end());
        Vocab vocab(words, false);
        Story<uint> output;
        output.question_fidx = input.question_fidx;
        output.supporting_facts = input.supporting_facts;
        for (auto& fact: input.facts) {
            output.facts.emplace_back(vocab.encode(fact));
        }
        for (auto& answer: input.answers) {
            output.answers.emplace_back(vocab.encode(answer));
        }
        return std::make_tuple(output, vocab);
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
            int line_number;
            line >> line_number;
            if (last_line_no == -1 || line_number <= last_line_no) {
                results.emplace_back();
            }
            last_line_no = line_number;

            // parse the fact.
            vector<string> fact;
            while(true) {
                string token;
                line >> token;
                if (token.back() == '.' || token.back() == '?') {
                    fact.emplace_back(token.begin(), token.end() - 1);
                    fact.emplace_back(token.end()-1, token.end());
                    break;
                } else {
                    fact.push_back(token);
                }
            }

            // store the fact.
            results.back().facts.push_back(fact);

            // if this is a question store its index.
            if (fact.back().back() == '?') {
                results.back().question_fidx.push_back(
                        results.back().facts.size() - 1);
            }

            // if this ia question do read in the answer and supporting facts.
            if (fact.back().back() == '?') {
                string comma_separated_answer;
                line >> comma_separated_answer;

                vector<string> answer = utils::split(comma_separated_answer, ',');
                results.back().answers.push_back(answer);

                results.back().supporting_facts.emplace_back();

                int supporting_fact;
                while(line >> supporting_fact) {
                    results.back().supporting_facts.back().push_back(supporting_fact - 1);
                }
            }
        }

        return results;
    }


}
