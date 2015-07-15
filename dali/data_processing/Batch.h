#ifndef DALI_DATA_PROCESSING_BATCH_H
#define DALI_DATA_PROCESSING_BATCH_H

#include <vector>
#include "dali/tensor/Mat.h"
#include "dali/utils/assert2.h"
#include "dali/utils/core_utils.h"

// Batch of input-target pairs.
template<typename R>
struct Batch {
    // inputs
    Mat<int> data;
    // labels
    Mat<int> target;
    // when the labels must be used
    Mat<R> mask;
    // length of each example in batch
    std::vector<int> code_lengths;
    // number of unique target-example pairs
    int total_codes;

    Batch() = default;

    void insert_example(const std::vector<std::string>& example, const utils::Vocab& vocab, size_t example_idx, int offset = 0);

};

#endif
