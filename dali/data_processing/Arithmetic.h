#ifndef DATA_PROCESSING_ARITHMETIC_H
#define DATA_PROCESSING_ARITHMETIC_H

#include <string>
#include <vector>
#include "dali/utils/core_utils.h"

namespace arithmetic {
    typedef std::pair<std::vector<std::string>, std::vector<std::string>> Example;
    typedef std::pair<std::vector<uint>, std::vector<uint>> NumericalExample;

    std::vector<Example> generate(int num, int expression_length);
    std::vector<NumericalExample> generate_numerical(int num, int expression_length, bool with_end_symbol=true);

    extern std::vector<std::string> symbols;
    extern utils::Vocab vocabulary;
}

#endif
