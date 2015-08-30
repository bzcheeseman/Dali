#ifndef DATA_PROCESSING_ARITHMETIC_H
#define DATA_PROCESSING_ARITHMETIC_H

#include <string>
#include <vector>
#include <atomic>
#include "dali/utils/vocab.h"
#include "dali/utils/random.h"
#include "dali/utils/ThreadPool.h"

namespace arithmetic {
    typedef std::pair<std::vector<std::string>, std::vector<std::string>> example_t;
    typedef std::pair<std::vector<uint>, std::vector<uint>> numeric_example_t;

    std::vector<example_t> generate(int num, int expression_length, int min=0, int max=9);
    std::vector<numeric_example_t> generate_numerical(int num, int expression_length, int min=0, int max=9, bool with_end_symbol=true);

    extern std::vector<std::string> symbols;
    extern utils::Vocab vocabulary;

    double average_recall(
        std::vector<numeric_example_t>& dataset,
        std::function<std::vector<uint>(std::vector<uint>&)> predict,
        int num_threads = 9);
}

#endif
