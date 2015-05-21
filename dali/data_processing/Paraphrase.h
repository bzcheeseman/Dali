#ifndef PARAPHRASE_DALI_H
#define PARAPHRASE_DALI_H

#include <string>
#include <vector>
#include <atomic>
#include <set>
#include <map>

#include "dali/utils/core_utils.h"
#include "dali/utils/Reporting.h"
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
            std::vector<example_t> convert_tsv(const utils::tokenized_labeled_dataset& tsv_data);
    };
    typedef std::vector<example_t> paraphrase_full_dataset;

    paraphrase_full_dataset load(std::string path, similarity_score_extractor_t similarity_score_extractor);

    namespace STS_2015 {
        paraphrase_full_dataset load_train(std::string = STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/train.tsv");
        paraphrase_full_dataset load_test(std::string =  STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/test.tsv");
        paraphrase_full_dataset load_dev(std::string =   STR(DALI_DATA_DIR) "/paraphrase_STS_2015/secret/dev.tsv");
    }

    std::vector<std::string> get_vocabulary(const paraphrase_full_dataset& examples, int min_occurence);

    typedef std::vector<std::vector<numeric_example_t>> paraphrase_minibatch_dataset;

    paraphrase_minibatch_dataset convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        paraphrase_full_dataset& dataset,
        int minibatch_size);

    paraphrase_minibatch_dataset convert_to_character_minibatches(
        const utils::CharacterVocab& word_vocab,
        paraphrase_full_dataset& dataset,
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
};

#endif
