#include "lazy_eval.h"
#include "dali/utils/assert2.h"

namespace internal {
    ReductionInstruction requires_reduction(const Array& output, const std::vector<int>& in_bshape) {
        ReductionInstruction instruction;

        if (output.is_stateless()) {
            return instruction;
        }

        if (output.ndim() != in_bshape.size()) {
            return instruction;
        }

        auto out_bshape = output.bshape();
        int num_reductions = 0;

        for (int i = 0; i < out_bshape.size(); i++) {
            if (out_bshape[i] < 0) {
                ASSERT2(out_bshape[i] == -1,
                        "Assigning to broadcast_reshaped Array is not supported at the time.");
                if (std::abs(in_bshape[i]) > 1) {
                    // see if number of reductions is greater than 1 or equal to size of output.
                    num_reductions += 1;
                    instruction.axis = i;
                }
            }
        }

        if (num_reductions > 1 && num_reductions == out_bshape.size()) {
            instruction.all_reduce = true;
            instruction.axis = -1;
        }
        ASSERT2(num_reductions == out_bshape.size() || num_reductions <= 1,
            "Can (currently) only assign to an array with at most 1 broadcasted dimension, or to fully broadcasted scalar");
        return instruction;
    }
}
