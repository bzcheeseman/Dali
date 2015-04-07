#ifndef NLP_UTILS_H
#define NLP_UTILS_H

#include "gflags/gflags.h"
#include "dali/utils/core_utils.h"

DECLARE_int32(subsets);
DECLARE_int32(min_occurence);
DECLARE_int32(epochs);
DECLARE_string(train);
DECLARE_int32(j);
DECLARE_string(validation);

static bool dummy = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_train,
                                                            &utils::validate_flag_nonempty);
static bool dummy1 = GFLAGS_NAMESPACE::RegisterFlagValidator(&FLAGS_validation,
                                                            &utils::validate_flag_nonempty);
#endif
