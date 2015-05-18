#include "dali/data_processing/NER.h"

using std::string;
using std::vector;
using utils::assert2;

namespace NER {

    NER_Loader::NER_Loader() : data_column(0), label_column(-1) {}

    vector<example_t> NER_Loader::convert_tsv(const utils::tokenized_labeled_dataset& tsv_data) {
        vector< example_t > split_dataset;
        example_t current_example;
        for (auto& line : tsv_data) {
            if (line[0].front() == start_symbol) {
                if (current_example.first.size() > 0) {
                    split_dataset.emplace_back(current_example);
                    current_example = example_t();
                }
            } else {
                if (data_column < 0) {
                    assert2(line.size() + data_column > 0, "Accessing a negatively valued column.");
                    current_example.first.emplace_back(line[line.size() + data_column].front());
                } else {
                    assert2(line.size() > data_column, "Accessing a non-existing column");
                    current_example.first.emplace_back(line[data_column].front());
                }

                if (label_column < 0) {
                    assert2(line.size() + label_column > 0, "Accessing a negatively valued column.");
                    current_example.second.emplace_back(line[line.size() + label_column].front());
                } else {
                    assert2(line.size() > label_column, "Accessing a non-existing column");
                    current_example.second.emplace_back(line[label_column].front());
                }
            }
        }
        if (current_example.first.size() > 1) {
            split_dataset.emplace_back(current_example);
        }
        return split_dataset;
    }

    vector<example_t> load(string path, string start_symbol) {
        auto tsv_data = utils::load_tsv(
            path,
            -1,
            '\t'
        );
        auto loader = NER_Loader();
        loader.start_symbol = start_symbol;
        return loader.convert_tsv(tsv_data);
    }
}
