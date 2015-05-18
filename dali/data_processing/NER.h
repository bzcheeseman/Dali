#ifndef NER_DALI_H
#define NER_DALI_H

#include <string>
#include <vector>
#include "dali/utils.h"

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
    std::vector<example_t> load(std::string path, std::string start_symbol = "-DOCSTART-");
}

#endif
