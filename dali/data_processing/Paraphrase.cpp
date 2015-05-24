#include "dali/data_processing/Paraphrase.h"

using namespace std::placeholders;
using std::vector;
using std::string;
using utils::assert2;
using utils::from_string;

namespace paraphrase {

    utils::Generator<example_t> generate_examples(ParaphraseLoader& para_loader, std::string fname) {
        auto gen = utils::make_generator<example_t>([para_loader, fname](utils::yield_t<example_t> yield) {
            auto row_gen = utils::generate_tsv_rows(fname);
            for (auto row : row_gen)
                yield(para_loader.tsv_row_to_example(row));
        });
        return gen;
    }

    utils::Generator<example_t> generate_examples(std::string fname, similarity_score_extractor_t similarity_score_extractor) {
        auto para_loader = ParaphraseLoader();
        para_loader.similarity_score_extractor = similarity_score_extractor;
        return generate_examples(para_loader, fname);
    }

    paraphrase_full_dataset load(ParaphraseLoader& para_loader, std::string path) {
        auto gen = generate_examples(para_loader, path);
        paraphrase_full_dataset examples;
        for (auto example : gen)
            examples.emplace_back(example);
        return examples;
    }

    paraphrase_full_dataset load(std::string path, similarity_score_extractor_t similarity_score_extractor) {
        auto para_loader = ParaphraseLoader();
        para_loader.similarity_score_extractor = similarity_score_extractor;
        return load(para_loader, path);
    }

    ParaphraseLoader::ParaphraseLoader() : sentence1_column(0), sentence2_column(1), similarity_column(2) {}

    example_t ParaphraseLoader::tsv_row_to_example(utils::row_t& row) const {
        example_t example;
        if (sentence1_column < 0) {
            assert2(row.size() + sentence1_column > 0, "Accessing a negatively valued column.");
            std::get<0>(example) = row[row.size() + sentence1_column];
        } else {
            assert2(row.size() > sentence1_column, "Accessing a non-existing column");
            std::get<0>(example) = row[sentence1_column];
        }
        if (sentence2_column < 0) {
            assert2(row.size() + sentence2_column > 0, "Accessing a negatively valued column.");
            std::get<1>(example) = row[row.size() + sentence2_column];
        } else {
            assert2(row.size() > sentence2_column, "Accessing a non-existing column");
            std::get<1>(example) = row[sentence2_column];
        }
        if (similarity_column < 0) {
            assert2(row.size() + similarity_column > 0, "Accessing a negatively valued column.");
            std::get<2>(example) = similarity_score_extractor(
                utils::join(row[row.size() + similarity_column])
            );
        } else {
            assert2(row.size() > similarity_column, "Accessing a non-existing column");
            std::get<2>(example) = similarity_score_extractor(
                utils::join(row[similarity_column])
            );
        }
        return example;
    }

    paraphrase_full_dataset ParaphraseLoader::convert_tsv(const utils::tokenized_labeled_dataset& tsv_data) const {
        assert2(similarity_score_extractor != NULL,
            "Must pass a similarity score extractor to convert similarity scores from strings to doubles."
        );
        paraphrase_full_dataset split_dataset;
        for (auto line : tsv_data)
            split_dataset.emplace_back(tsv_row_to_example(line));
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

    vector<string> get_vocabulary(utils::Generator<example_t>& examples, int min_occurence) {
        std::map<string, uint> word_occurences;
        string word;
        for (auto example : examples) {
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

        utils::Generator<example_t> generate_train(std::string path) {
            auto loader = ParaphraseLoader();
            loader.sentence1_column  = 2;
            loader.sentence2_column  = 3;
            loader.similarity_column = 4;

            auto score_map = std::map<string, double> {
                {"(0,5)", 0.0},
                {"(1,4)", 0.0},
                {"(2,3)", 0.5},
                {"(3,2)", 1.0},
                {"(4,1)", 1.0},
                {"(5,0)", 1.0},
            };

            loader.similarity_score_extractor = [score_map](const string& score_str) {return score_map.at(score_str);};
            return generate_examples(loader, path);
        }

        paraphrase_full_dataset load_train(std::string path) {
            auto gen = STS_2015::generate_train(path);
            paraphrase_full_dataset examples;
            for (auto example : gen)
                examples.emplace_back(example);
            return examples;
        }

        utils::Generator<example_t> generate_test(std::string path) {
            auto loader = ParaphraseLoader();
            loader.sentence1_column = 2;
            loader.sentence2_column = 3;
            loader.similarity_column = 4;
            loader.similarity_score_extractor = [](const string& number_column) {
                auto num = from_string<int>(number_column);
                return ((double) num) / 5.0;
            };
            return generate_examples(loader, path);
        }

        paraphrase_full_dataset load_test(std::string path) {
            auto gen = STS_2015::generate_test(path);
            paraphrase_full_dataset examples;
            for (auto example : gen)
                examples.emplace_back(example);
            return examples;
        }

        utils::Generator<example_t> generate_dev(std::string path) {
            return generate_train(path);
        }

        paraphrase_full_dataset load_dev(std::string path) {
            return load_train(path);
        }
    }
    namespace STS_2014 {
        paraphrase_full_dataset load(std::string path) {
            auto loader = ParaphraseLoader();
            loader.sentence1_column  = 0;
            loader.sentence2_column  = 1;
            loader.similarity_column = 2;

            loader.similarity_score_extractor = [](const string& score_str) {
                return from_string<double>(score_str) / 5.0;
            };

            return load(loader, path);
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

    utils::Generator<vector<numeric_example_t>> convert_to_indexed_minibatches(
            const utils::Vocab& word_vocab,
            utils::Generator<example_t>& examples,
            int minibatch_size) {

        auto gen = utils::make_generator<vector<numeric_example_t>>(
                [examples, minibatch_size, word_vocab](utils::yield_t<vector<numeric_example_t>> yield) {
            int seen = 0;
            auto minibatch = vector<numeric_example_t>();
            minibatch.reserve(minibatch_size);
            auto ex_iter = examples;
            for (auto example : ex_iter) {
                minibatch.emplace_back(
                    word_vocab.encode(std::get<0>(example)),
                    word_vocab.encode(std::get<1>(example)),
                    std::get<2>(example)
                );
                if (seen++ % minibatch_size == 0) {
                    yield(minibatch);
                    minibatch.clear();
                }
            }
            if (minibatch.size() > 0) {
                yield(minibatch);
            }
        });
        return gen;
    }


    utils::Generator<vector<numeric_example_t>> convert_to_indexed_minibatches(
            const utils::CharacterVocab& character_vocab,
            utils::Generator<example_t>& examples,
            int minibatch_size) {

        auto gen = utils::make_generator<vector<numeric_example_t>>(
                [examples, minibatch_size, character_vocab](utils::yield_t<vector<numeric_example_t>> yield) {
            int seen = 0;
            auto minibatch = vector<numeric_example_t>();
            minibatch.reserve(minibatch_size);
            auto ex_iter = examples;
            for (auto example : ex_iter) {
                seen++;
                minibatch.emplace_back(
                    character_vocab.encode(std::get<0>(example)),
                    character_vocab.encode(std::get<1>(example)),
                    std::get<2>(example)
                );
                if (seen % minibatch_size == 0) {
                    yield(minibatch);
                    minibatch.clear();
                }
            }
            if (minibatch.size() > 0) {
                yield(minibatch);
            }
        });
        return gen;
    }


    paraphrase_minibatch_dataset convert_to_indexed_minibatches(
            const utils::CharacterVocab& character_vocab,
            paraphrase_full_dataset& examples,
            int minibatch_size) {
        paraphrase_minibatch_dataset dataset;

        auto to_index_pair = [&character_vocab](example_t& example) {
            return numeric_example_t(
                character_vocab.encode(std::get<0>(example)),
                character_vocab.encode(std::get<1>(example)),
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

    template<typename T>
    vector<T> collect_predictions(
            paraphrase_minibatch_dataset& dataset,
            std::function<T(std::vector<uint>&, std::vector<uint>&)> predict,
            int num_threads
            ) {

        int total;
        for (auto& minibatch : dataset) total += minibatch.size();
        vector<T> predictions(total);

        ThreadPool pool(num_threads);
        int predictions_idx = 0;
        for (int batch_id = 0; batch_id < dataset.size(); ++batch_id) {
            pool.run([batch_id, predictions_idx, &predict, &predictions, &dataset]() {
                auto& minibatch = dataset[batch_id];
                for (int example_idx = 0; example_idx < minibatch.size(); example_idx++) {
                    predictions[example_idx + predictions_idx] = predict(
                        std::get<0>(minibatch[example_idx]),
                        std::get<1>(minibatch[example_idx])
                    );
                }
            });
            predictions_idx += dataset[batch_id].size();
        }
        pool.wait_until_idle();
        return predictions;
    }

    double pearson_correlation(
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads) {

        vector<double> gold_labels;
        for (auto& minibatch : dataset) {
            for (auto& example : minibatch) gold_labels.emplace_back(std::get<2>(example));
        }

        auto predictions = collect_predictions<double>(dataset, predict, num_threads);

        return utils::pearson_correlation(gold_labels, predictions);
    }

    utils::Accuracy binary_accuracy(
        paraphrase_minibatch_dataset& dataset,
        std::function<Label(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads) {

        auto predictions = collect_predictions<Label>(dataset, predict, num_threads);

        int example_idx    = 0;
        int true_positive  = 0;
        int false_positive = 0;
        int true_negative  = 0;
        int false_negative = 0;

        Label correct_label;
        for (auto& minibatch : dataset) {
            for (auto& example : minibatch) {
                correct_label = std::get<2>(example) >= 0.65 ? PARAPHRASE : (std::get<2>(example) <= 0.55 ? NOT_PARAPHRASE : UNDECIDED);
                if (correct_label != UNDECIDED) {
                    if        (predictions[example_idx] == PARAPHRASE     && correct_label == PARAPHRASE) {
                        true_positive++;
                    } else if (predictions[example_idx] == NOT_PARAPHRASE && correct_label == NOT_PARAPHRASE) {
                        true_negative++;
                    } else if (predictions[example_idx] == PARAPHRASE     && correct_label == NOT_PARAPHRASE) {
                        false_positive++;
                    } else if (predictions[example_idx] == NOT_PARAPHRASE && correct_label == PARAPHRASE) {
                        false_negative++;
                    }
                }
                example_idx++;
            }
        }

        return utils::Accuracy()
            .true_positive(true_positive)
            .true_negative(true_negative)
            .false_positive(false_positive)
            .false_negative(false_negative);
    }

    utils::Accuracy binary_accuracy(
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
