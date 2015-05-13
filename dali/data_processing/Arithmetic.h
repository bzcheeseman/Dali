#ifndef DATA_PROCESSING_ARITHMETIC_H
#define DATA_PROCESSING_ARITHMETIC_H

#include <string>
#include <vector>
#include "dali/utils/core_utils.h"

namespace arithmetic {
    typedef std::pair<std::vector<std::string>, std::vector<std::string>> Example;
    typedef std::pair<std::vector<uint>, std::vector<uint>> NumericalExample;

    std::vector<Example> generate(int num, int expression_length, int min=0, int max=9);
    std::vector<NumericalExample> generate_numerical(int num, int expression_length, int min=0, int max=9, bool with_end_symbol=true);


    std::tuple<std::vector<int>, std::vector<std::string>> remove_multiplies(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    std::tuple<std::vector<int>, std::vector<std::string>> generate_example(int expression_length, int& min, int& max);
    std::vector<std::string> convert_to_chars(const std::vector<int>& numbers, const std::vector<std::string>& ops);
    int compute_result(const std::vector<int>& numbers, const std::vector<std::string>& ops);

    extern std::vector<std::string> symbols;
    extern utils::Vocab vocabulary;
}

#endif
