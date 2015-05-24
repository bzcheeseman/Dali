#ifndef TSV_UTILS_DALI_H
#define TSV_UTILS_DALI_H

#include "dali/utils/core_utils.h"
#include "dali/utils/generator.h"
#include <string>
#include <vector>
#include <fstream>
#include <ostream>
#include <sstream>
#include <memory>

namespace utils {
    typedef std::vector<std::vector<std::string>> row_t;

    Generator<row_t> generate_tsv_rows(const std::string& fname, const char& delimiter = '\t');

    template<typename T>
    Generator<row_t> generate_tsv_rows_from_stream(std::shared_ptr<T> stream, const char& delimiter = '\t');

    tokenized_labeled_dataset load_tsv(const std::string&, int number_of_columns = -1, const char& delimiter = '\t');
}

#endif
