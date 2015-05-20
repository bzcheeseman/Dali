#include "dali/data_processing/Paraphrase.h"

using std::vector;
using std::string;

namespace paraphrase {
    vector<string> get_vocabulary(const ner_full_dataset& examples, int min_occurence) {
        std::map<string, uint> word_occurences;
        string word;
        for (auto& example : examples)
            for (auto& word : example.first) word_occurences[word] += 1;
        vector<string> list;
        for (auto& key_val : word_occurences)
            if (key_val.second >= min_occurence)
                list.emplace_back(key_val.first);
        list.emplace_back(utils::end_symbol);
        return list;
    }
};
