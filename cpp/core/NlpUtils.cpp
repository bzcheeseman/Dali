#include "NlpUtils.h"

#include "utils.h"

DEFINE_int32(subsets,            10,
        "Break up dataset into how many minibatches ? \n"
        "(Note: reduces batch sparsity)");
DEFINE_int32(min_occurence,      2,
        "How often a word must appear to be included in the Vocabulary \n"
        "(Note: other words replaced by special **UNKNONW** word)");
DEFINE_int32(epochs,             5,
        "How many training loops through the full dataset ?");
DEFINE_int32(report_frequency,   1,
        "How often (in epochs) to print the error to standard out during training.");
DEFINE_string(train,             "",
        "Training dataset . ");
DEFINE_int32(j,                  1,
        "How many threads should be used ?");
DEFINE_string(validation,        "",
        "Location of the validation dataset");

static bool dummy = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_train,
                                                            &utils::validate_flag_nonempty);
static bool dummy1 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_validation,
                                                            &utils::validate_flag_nonempty);
