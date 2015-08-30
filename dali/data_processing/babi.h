#ifndef DALI_DATA_PROCESSING_BABI2_H
#define DALI_DATA_PROCESSING_BABI2_H

#include <string>
#include <vector>
#include <unordered_map>

#include "dali/utils/vocab.h"

namespace babi {
    typedef std::unordered_map<std::string, std::string> properties_t;

    template<typename word_t>
    struct QA {
        std::vector<std::vector<word_t>> facts;
        std::vector<word_t> question;
        std::vector<word_t> answer;
        std::vector<uint> supporting_facts;
        properties_t properties;
    };

    template<typename word_t>
    struct Story {
        std::vector<std::vector<word_t>> facts;
        std::vector<uint> question_fidx;
        std::vector<std::vector<uint>> supporting_facts;
        std::vector<std::vector<word_t>> answers;
        std::vector<properties_t> properties;

        uint size() const;

        void print() const;
        void print_raw() const;

        QA<word_t> get(int target_question_idx) const;
    };

    Story<std::string> decode(Story<uint> story, const utils::Vocab& vocab, bool strip_eos=true);
    Story<uint> encode(Story<std::string> story, const utils::Vocab& vocab, bool add_eos=false);

    std::vector<Story<uint>> encode_dataset(
            const std::vector<Story<std::string>>& input,
            utils::Vocab* vocab,
            bool add_eos=false,
            uint min_occurence=1);

    std::vector<Story<std::string>> parse_file(const std::string& filename);

    std::string data_dir();

    std::vector<Story<std::string>> dataset(std::string task_prefix,
                                            std::string train_or_test,
                                            std::string dataset_prefix="en");

    // List of all the babi tasks
    std::vector<std::string> tasks();

    // Takes as argument list of results for all the 20 tasks
    // and prints them side by side with facebook results.
    void compare_results(std::vector<double> our_results);
}


#endif
