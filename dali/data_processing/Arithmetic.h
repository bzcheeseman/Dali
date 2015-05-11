#ifndef DATA_PROCESSING_ARITHMETIC_H
#define DATA_PROCESSING_ARITHMETIC_H

#include <string>
#include <vector>
#include "dali/utils/core_utils.h"

namespace arithmetic {
    typedef std::pair<std::vector<std::string>, std::vector<std::string>> Example;
    std::vector<Example> generate(int num, int expression_length);
    extern std::vector<std::string> symbols;
    utils::Vocab vocabulary();
}

#endif
