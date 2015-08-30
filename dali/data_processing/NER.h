#ifndef NER_DALI_H
#define NER_DALI_H

#include <string>
#include <vector>
#include <atomic>
#include <set>
#include "dali/utils/tsv_utils.h"
#include "dali/utils/Reporting.h"
#include "dali/utils/vocab.h"

namespace NER {
    typedef std::pair< std::vector<std::string>, std::vector<std::string> > example_t;
    class NER_Loader {
        public:
            std::string start_symbol;
            int expected_columns;
            int data_column;
            int label_column;
            NER_Loader();
            std::vector<example_t> convert_tsv(const utils::tokenized_labeled_dataset& tsv_data);
    };
    typedef std::vector<example_t> ner_full_dataset;

    ner_full_dataset load(std::string path, std::string start_symbol = "-DOCSTART-");

    std::vector<std::string> get_vocabulary(const ner_full_dataset& examples, int min_occurence);
    std::vector<std::string> get_label_vocabulary(const ner_full_dataset& examples);

    typedef std::vector<std::vector<std::tuple<std::vector<uint>, std::vector<uint>>>> ner_minibatch_dataset;

    ner_minibatch_dataset convert_to_indexed_minibatches(
        const utils::Vocab& word_vocab,
        const utils::Vocab& label_vocab,
        ner_full_dataset& dataset,
        int minibatch_size);

    /**
    Average Recall
    --------------
    Obtain average recall using minibatches by parallelizing a
    predict function that takes a string of indices (words) as
    input, and a lambda returning an integer as output.

    Inputs
    ------
    dataset
    predict : lambda returning predicted entity predictions at each timesteps
    num_threads (optional) : how many threads to use for parallelizing prediction

    Outputs
    -------
    double recall : returns tuple of {total correct, root correct}

    **/
    double average_recall(
        ner_minibatch_dataset& dataset,
        std::function<std::vector<uint>(std::vector<uint>&)> predict,
        int num_threads = 9);
}

#endif
