#include "dali/data_processing/NER.h"

using std::string;
using std::vector;
using utils::assert2;

namespace NER {
    vector<example_t> load(string path, string start_symbol) {
        bool label_on_left = false;
        auto dataset = utils::load_tokenized_labeled_corpus(
            path,
            label_on_left,
            '\t'
        );
        vector< example_t > split_dataset;
        example_t current_example;
        for (auto& line : dataset) {
            if (line.first.front() == start_symbol) {
                if (current_example.first.size() > 0) {
                    split_dataset.emplace_back(current_example);
                    current_example = example_t();
                }
            } else {
                current_example.first.emplace_back(line.first.front());
                current_example.second.emplace_back(line.second);
            }
        }
        if (current_example.first.size() > 1) {
            split_dataset.emplace_back(current_example);
        }
        return split_dataset;
    }
}
