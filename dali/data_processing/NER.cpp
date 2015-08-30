#include "dali/data_processing/NER.h"
#include "dali/utils/ThreadPool.h"

using std::string;
using std::vector;
using utils::assert2;
using std::atomic;

namespace NER {

    NER_Loader::NER_Loader() : data_column(0), label_column(-1) {}

    ner_full_dataset NER_Loader::convert_tsv(const utils::tokenized_labeled_dataset& tsv_data) {
        ner_full_dataset split_dataset;
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
                } else {
                    assert2(line.size() > data_column, "Accessing a non-existing column");
                    current_example.first.emplace_back(line[data_column].front());
                }

                if (label_column < 0) {
                    assert2(line.size() + label_column > 0, "Accessing a negatively valued column.");
                    current_example.second.emplace_back(line[line.size() + label_column].front());
                } else {
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

    ner_full_dataset load(string path, string start_symbol) {
        auto tsv_data = utils::load_tsv(
            path,
            -1,
            '\t'
        );
        auto loader = NER_Loader();
        loader.start_symbol = start_symbol;
        return loader.convert_tsv(tsv_data);
    }

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

    vector<string> get_label_vocabulary(const ner_full_dataset& examples) {
        std::map<string, uint> label_occurences;
        for (auto& example : examples) {
            for (auto& label : example.second) {
                label_occurences[label] += 1;
            }
        }
        vector<string> labels;
        labels.reserve(label_occurences.size());
        for (auto& key_val : label_occurences)
            labels.emplace_back(key_val.first);

        std::sort(labels.begin(), labels.end(), [&label_occurences](const string& a, const string& b) {
            return label_occurences[a] > label_occurences[b];
        });

        return labels;
    }

    ner_minibatch_dataset convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        const utils::Vocab& label_vocab,
        ner_full_dataset& examples,
        int minibatch_size) {

        ner_minibatch_dataset dataset;

        auto to_index_pair = [&word_vocab, &label_vocab](example_t& example) {
            return std::tuple<std::vector<uint>, std::vector<uint>>(
                word_vocab.encode(example.first),
                label_vocab.encode(example.second)
            );
        };

        if (dataset.size() == 0)
            dataset.emplace_back(0);

        for (auto& example : examples) {

            // create new minibatch
            if (dataset[dataset.size()-1].size() == minibatch_size) {
                dataset.emplace_back(0);
                dataset.back().reserve(minibatch_size);
            }

            // add example
            dataset[dataset.size()-1].emplace_back(
                to_index_pair(example)
            );
        }

        return dataset;
    }

    double average_recall(
        ner_minibatch_dataset& dataset,
        std::function<vector<uint>(vector<uint>&)> predict,
        int num_threads) {
        ReportProgress<double> journalist("Average recall", dataset.size());
        atomic<int> seen_minibatches(0);
        atomic<int> correct(0);
        atomic<int> total(0);
        ThreadPool pool(num_threads);

        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool.run([batch_id, &predict, &dataset, &correct, &total, &journalist, &seen_minibatches]() {
                auto& minibatch = dataset[batch_id];
                for (auto& example : minibatch) {
                    auto prediction = predict(std::get<0>(example));
                    for (auto it_pred = prediction.begin(),
                              it_label = std::get<1>(example).begin();
                        it_pred < prediction.end() && it_label < std::get<1>(example).end();
                        it_label++, it_pred++) {
                        if (*it_pred == *it_label)
                            correct += 1;
                    }
                    total += std::get<0>(example).size();
                }
                seen_minibatches += 1;
                journalist.tick(seen_minibatches, 100.0 * ((double) correct / (double) total));
            });
        }
        pool.wait_until_idle();
        journalist.done();
        return 100.0 * ((double) correct / (double) total);
    }
}
