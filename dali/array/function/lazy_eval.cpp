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

        std::vector<int> reduction_axes;

        for (int i = 0; i < out_bshape.size(); ++i) {
            if (out_bshape[i] < 0) {
                ASSERT2(out_bshape[i] == -1,
                        "Assigning to broadcast_reshaped Array is not supported at the time.");
                if (std::abs(in_bshape[i]) > 1) {
                    // see if number of reductions is greater than 1 or equal to size of output.
                    reduction_axes.emplace_back(i);
                }
            }
        }

        if (reduction_axes.size() == 0) {
            return instruction;
        }

        if (reduction_axes.size() == out_bshape.size()) {
            instruction.type = REDUCTION_INSTR_T_ALL;
            return instruction;
        }

        bool contiguous = true;
        for (int i = 0; i + 1 < reduction_axes.size(); ++i) {
            if (reduction_axes[i] + 1 != reduction_axes[i + 1]) {
                contiguous = false;
                break;
            }
        }

        if (contiguous) {
            instruction.reduce_start = reduction_axes[0];
            instruction.reduce_end   = reduction_axes[0] + reduction_axes.size();
            instruction.type = REDUCTION_INSTR_T_CONTIG;
        } else {
            instruction.noncontiguous_axes = reduction_axes;
            instruction.type = REDUCTION_INSTR_T_NONCONTIG;
        }
        return instruction;
    }
}
