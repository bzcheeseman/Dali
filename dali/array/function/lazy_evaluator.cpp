#include "lazy_evaluator.h"

#include "dali/utils/assert2.h"

namespace internal {
    int requires_reduction(const Array& output, const std::vector<int>& in_bshape) {
        if (output.is_stateless()) {
            return -1;
        }

        if (output.ndim() != in_bshape.size()) {
            return -1;
        }

        auto out_bshape = output.bshape();
        int reduction_dimension = -1;

        for (int i = 0; i < out_bshape.size(); i++) {
            if (out_bshape[i] < 0) {
                ASSERT2(out_bshape[i] == -1,
                        "Assigning to broadcast_reshaped Array is not supported at the time.");
                if (std::abs(in_bshape[i]) > 1) {
                    ASSERT2(reduction_dimension == -1,
                            "Can only assign to an array with at most 1 broadcasted dimension.");
                    reduction_dimension = i;
                }
            }
        }

        return reduction_dimension;
    }
}
