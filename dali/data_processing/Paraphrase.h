#ifndef PARAPHRASE_DALI_H
#define PARAPHRASE_DALI_H

#include <string>
#include <vector>
#include <atomic>
#include <set>
#include "dali/utils/core_utils.h"
#include "dali/utils/Reporting.h"

namespace paraphrase {
    /*
    example_t is a tuple of the form is sentence 1, sentence 2,
    and similarity as a double (e.g. 0.0 => different, 1.0 => identical)
    */
    typedef std::tuple< std::vector<std::string>, std::vector<std::string>, double > example_t;
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
    typedef std::vector<example_t> ner_full_dataset;

    ner_full_dataset load(std::string path, similarity_score_extractor_t similarity_score_extractor);

    std::vector<std::string> get_vocabulary(const ner_full_dataset& examples, int min_occurence);

    typedef std::vector<std::vector<std::tuple<std::vector<uint>, std::vector<uint>, double>>> ner_minibatch_dataset;

    ner_minibatch_dataset convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        ner_full_dataset& dataset,
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

    Outputs
    -------

    double recall : returns tuple of {total correct, root correct}

    **/
    double pearson_correlation(
        ner_minibatch_dataset& dataset,
        std::function<double(std::vector<uint>&, std::vector<uint>&)> predict,
        int num_threads = 9);
};

#endif
