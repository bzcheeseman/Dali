#ifndef PARAPHRASE_DALI_H
#define PARAPHRASE_DALI_H

#include <string>
#include <vector>
#include <atomic>
#include <set>
#include <memory>
#include <algorithm>
#include <map>

#include "dali/utils/tsv_utils.h"
#include "dali/utils/Reporting.h"
#include "dali/visualizer/visualizer.h"
#include "dali/utils/scoring_utils.h"

namespace paraphrase {
    /* example_t is a tuple of the form is sentence 1, sentence 2,
    and similarity as a double (e.g. 0.0 => different, 1.0 => identical) */
    typedef std::tuple< std::vector<std::string>, std::vector<std::string>, double > example_t;
    typedef std::tuple<std::vector<uint>, std::vector<uint>, double>         numeric_example_t;
    typedef std::function<double(const std::string&)> similarity_score_extractor_t;
    class ParaphraseLoader {
        public:
            int expected_columns;
            int sentence1_column;
            int sentence2_column;
            int similarity_column;
            similarity_score_extractor_t similarity_score_extractor;
            ParaphraseLoader();
            example_t tsv_row_to_example(utils::row_t& row) const;
            std::vector<example_t> convert_tsv(const utils::tokenized_labeled_dataset& tsv_data) const;
    };
    typedef std::vector<example_t> paraphrase_full_dataset;


    utils::Generator<example_t> generate_examples(std::string path, similarity_score_extractor_t similarity_score_extractor);
    utils::Generator<example_t> generate_examples(ParaphraseLoader&, std::string path);

    paraphrase_full_dataset load(std::string path, similarity_score_extractor_t similarity_score_extractor);
    paraphrase_full_dataset load(ParaphraseLoader&, std::string path);

    namespace STS_2015 {
        utils::Generator<example_t> generate_train(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/train.tsv");
        paraphrase_full_dataset load_train(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/train.tsv");
        utils::Generator<example_t> generate_test(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/test.tsv");
        paraphrase_full_dataset load_test(std::string =  STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/test.tsv");
        utils::Generator<example_t> generate_dev(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/dev.tsv");
        paraphrase_full_dataset load_dev(std::string =   STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/dev.tsv");
    }

    namespace STS_2014 {
        paraphrase_full_dataset     load_test(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2014/test.tokenized.tsv");
        utils::Generator<example_t> generate_test(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2014/test.tokenized.tsv");
        paraphrase_full_dataset     load_train(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2014/train.tokenized.tsv");
        utils::Generator<example_t> generate_train(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2014/train.tokenized.tsv");
    }

    namespace wikianswers {
        paraphrase_full_dataset     load(std::string = STR(DALI_DATA_DIR) "/paraphrase_wikianswers/wikianswers.paraphrases.tsv.gz");
        utils::Generator<example_t> generate(std::string = STR(DALI_DATA_DIR) "/paraphrase_wikianswers/wikianswers.paraphrases.tsv.gz");
    }

    std::vector<std::string> get_vocabulary(const paraphrase_full_dataset& examples, int min_occurence, int max_examples_seen = -1);
    std::vector<std::string> get_vocabulary(utils::Generator<example_t>& examples, int min_occurence, int max_examples_seen = -1);

    typedef std::vector<std::vector<numeric_example_t>> paraphrase_minibatch_dataset;

    paraphrase_minibatch_dataset convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        paraphrase_full_dataset& dataset,
        int minibatch_size);

    paraphrase_minibatch_dataset convert_to_indexed_minibatches(
        const utils::CharacterVocab& character_vocab,
        paraphrase_full_dataset& dataset,
        int minibatch_size);

    utils::Generator<std::vector<numeric_example_t>> convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        utils::Generator<example_t>& dataset,
        int minibatch_size);

    utils::Generator<std::vector<numeric_example_t>> convert_to_indexed_minibatches(
        const utils::CharacterVocab& character_vocab,
        utils::Generator<example_t>& dataset,
        int minibatch_size);

    /**
    Pearson Correlation
    -------------------

    Obtain pearson correlation between true similarity of
    sentences, and model's predicted similarities.

    Inputs
    ------
    std::vector<std::vector<std::tuple<std::vector<uint>, std::vector<uint>, double>>>& dataset
    std::function<std::vector<uint>(std::vector<uint>&,std::vector<uint>&))> predict : lambda returning
        predicted entity predictions at each timesteps
    int num_threads (optional) : how many threads to use for parallelizing prediction

    **/

    double pearson_correlation(
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads = 9);

    enum Label {
        PARAPHRASE,
        NOT_PARAPHRASE,
        UNDECIDED
    };

    /**
    Share among threads the computation of predictions and return a vector
    of all predictions.
    **/
    template<typename T>
    std::vector<T> collect_predictions(
        paraphrase_minibatch_dataset& dataset,
        std::function<T(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads = 9);

    /*
    Binary Accuracy
    ---------------

    returns F1 score, recall, and precision for the predict method

    Note:

    */
    utils::Accuracy binary_accuracy(
        paraphrase_minibatch_dataset& dataset,
        std::function<Label(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads = 9
        );

    // This is a convenience method for evaluation real valued paraphrase
    // prediction, based on guidelines from STS 2015 dataset when evaluation.
    utils::Accuracy binary_accuracy(
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads = 9
        );

    json11::Json nearest_neighbors(
        const utils::CharacterVocab& character_vocab,
        std::vector<uint>& original,
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> distance,
        int max_number_of_comparisons
    );

    json11::Json nearest_neighbors(
        const utils::Vocab& word_vocab,
        std::vector<uint>& original,
        paraphrase_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> distance,
        int max_number_of_comparisons
    );

};

#endif
