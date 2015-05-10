#ifndef DATA_PROCESSING_ARITHMETIC_H
#define DATA_PROCESSING_ARITHMETIC_H

#include <string>
#include <vector>
#include "dali/utils/core_utils.h"

namespace arithmetic {
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> generate(int num, int expression_length);
    extern std::vector<std::string> symbols;
}

#endif
