#include "dali/data_processing/Paraphrase.h"

using std::vector;
using std::string;
using utils::assert2;
using utils::from_string;

namespace paraphrase {

    ParaphraseLoader::ParaphraseLoader() : sentence1_column(0), sentence2_column(1), similarity_column(2) {}

    paraphrase_full_dataset ParaphraseLoader::convert_tsv(const utils::tokenized_labeled_dataset& tsv_data) {
        assert2(similarity_score_extractor != NULL,
            "Must pass a similarity score extractor to convert similarity scores from strings to doubles."
        );
        paraphrase_full_dataset split_dataset;
        for (auto& line : tsv_data) {
            example_t example;
            if (sentence1_column < 0) {
                assert2(line.size() + sentence1_column > 0, "Accessing a negatively valued column.");
                std::get<0>(example) = line[line.size() + sentence1_column];
            } else {
                assert2(line.size() > sentence1_column, "Accessing a non-existing column");
                std::get<0>(example) = line[sentence1_column];
            }
            if (sentence2_column < 0) {
                assert2(line.size() + sentence2_column > 0, "Accessing a negatively valued column.");
                std::get<1>(example) = line[line.size() + sentence2_column];
            } else {
                assert2(line.size() > sentence2_column, "Accessing a non-existing column");
                std::get<1>(example) = line[sentence2_column];
            }
            if (similarity_column < 0) {
                assert2(line.size() + similarity_column > 0, "Accessing a negatively valued column.");
                std::get<2>(example) = similarity_score_extractor(
                    utils::join(line[line.size() + similarity_column])
                );
            } else {
                assert2(line.size() > similarity_column, "Accessing a non-existing column");
                std::get<2>(example) = similarity_score_extractor(
                    utils::join(line[similarity_column])
                );
            }
            split_dataset.emplace_back(example);
        }
        return split_dataset;
    }

    vector<string> get_vocabulary(const paraphrase_full_dataset& examples, int min_occurence) {
        std::map<string, uint> word_occurences;
        string word;
        for (auto& example : examples) {
            for (auto& word : std::get<0>(example)) word_occurences[word] += 1;
            for (auto& word : std::get<1>(example)) word_occurences[word] += 1;
        }
        vector<string> list;
        for (auto& key_val : word_occurences)
            if (key_val.second >= min_occurence)
                list.emplace_back(key_val.first);
        list.emplace_back(utils::end_symbol);
        return list;
    }

    namespace STS_2015 {
        paraphrase_full_dataset load_train(std::string path) {
            auto loader = ParaphraseLoader();
            loader.sentence1_column  = 2;
            loader.sentence2_column  = 3;
            loader.similarity_column = 4;

            vector<string> paraphrase_list = {"(3, 2)", "(4, 1)", "(5, 0)"};
            vector<string> non_paraphrase_list = {"(1, 4)", "(0, 5)"};
            loader.similarity_score_extractor = [&paraphrase_list, &non_paraphrase_list](const string& number_column) {
                if (number_column.size() > 1) {
                    return (
                        (double) from_string<int>(string(
                            number_column.begin() + 1,
                            number_column.begin() + 2)
                        ) / 5.0
                    );
                } else {
                    assert2("Number column should have format \"(A, B)\".");
                    return 0.5;
                }
            };
            auto tsv_data = utils::load_tsv(
                path,
                -1,
                '\t'
            );
            return loader.convert_tsv(tsv_data);
        }
        paraphrase_full_dataset load_test(std::string path) {
            auto loader = ParaphraseLoader();
            loader.sentence1_column = 2;
            loader.sentence2_column = 3;
            loader.similarity_column = 4;
            loader.similarity_score_extractor = [](const string& number_column) {
                auto num = utils::from_string<int>(number_column);
                return ((double) num) / 5.0;
            };
            auto tsv_data = utils::load_tsv(
                path,
                -1,
                '\t'
            );
            return loader.convert_tsv(tsv_data);
        }
        paraphrase_full_dataset load_dev(std::string path) {
            return load_train(path);
        }
    }

    paraphrase_minibatch_dataset convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        paraphrase_full_dataset& examples,
        int minibatch_size) {
        paraphrase_minibatch_dataset dataset;

        auto to_index_pair = [&word_vocab](example_t& example) {
            return numeric_example_t(
                word_vocab.encode(std::get<0>(example)),
                word_vocab.encode(std::get<1>(example)),
                std::get<2>(example)
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

    double pearson_correlation(
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads) {
        return 0.0;
    }

    double binary_accuracy(
        paraphrase_minibatch_dataset& dataset,
        std::function<Label(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads) {
        return 0.0;
    }

    double binary_accuracy(
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads) {

        std::function<Label(std::vector<uint>&, std::vector<uint>&)> new_predict_func =
                [&predict](std::vector<uint>& s1, std::vector<uint>& s2) {
            auto pred = predict(s1, s2);
            if (pred <= 0.4) {
                return NOT_PARAPHRASE;
            } else if (pred >= 0.6) {
                return PARAPHRASE;
            } else {
                return UNDECIDED;
            }
        };
        return binary_accuracy(dataset, new_predict_func, num_threads);
    }
};
