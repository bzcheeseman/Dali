#ifndef NLP_UTILS_H
#define NLP_UTILS_H

#include "gflags/gflags.h"
#include "dali/utils/core_utils.h"

DEFINE_int32(subsets,            10,
        "Break up dataset into how many minibatches ? \n"
        "(Note: reduces batch sparsity)");
DEFINE_int32(min_occurence,      2,
        "How often a word must appear to be included in the Vocabulary \n"
        "(Note: other words replaced by special **UNKNONW** word)");
DEFINE_int32(epochs,             2000,
        "How many training loops through the full dataset ?");
DEFINE_string(train,             "",
        "Training dataset . ");
DEFINE_int32(j,                  1,
        "How many threads should be used ?");
DEFINE_string(validation,        "",
        "Location of the validation dataset");


#endif
