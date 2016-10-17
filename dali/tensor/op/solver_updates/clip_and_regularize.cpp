#include "clip_and_regularize.h"

#include "dali/array/op2/binary.h"
#include "dali/array/op2/unary.h"
#include "dali/array/op_overload/common.h"
#include "dali/array/op_overload/nonlazy.h"

namespace tensor_ops {
    void clip_and_regularize(const Tensor& param,
                             const double& clip_abs,
                             const double& clip_norm,
                             const double& regc) {
        bool use_regc = regc > 0;
        bool use_abs_clip = clip_abs > 0;

        bool use_norm_clip = clip_norm > 0;

        double norm;
        if (use_norm_clip) {
            Array res = (Array)param.dw.L2_norm();
            // compute norm conditionally
            norm = (double)(Array)param.dw.L2_norm();
            // cancel normalization if norm is below threshold
            if (norm <= clip_norm) {
                use_norm_clip = false;
            }
        }

        if (use_regc) {
            if (!use_abs_clip && !use_norm_clip) {
                param.dw = param.dw + regc * param.w;
            } else if (use_abs_clip && !use_norm_clip) {
                param.dw = op::clip(param.dw, clip_abs) + (regc * param.w);
            } else if (!use_abs_clip && use_norm_clip) {
                param.dw = (clip_norm / norm) * param.dw + (regc * param.w);
            } else if (use_abs_clip && use_norm_clip) {
                param.dw = op::clip((clip_norm / norm) * param.dw, clip_abs) + (regc * param.w);
            }
        } else {
            if (use_abs_clip && !use_norm_clip) {
                param.dw = op::clip(param.dw, clip_abs);
            } else if (!use_abs_clip && use_norm_clip) {
                param.dw = (clip_norm / norm) * param.dw;
            } else if (use_abs_clip && use_norm_clip) {
                param.dw = op::clip(param.dw, clip_abs);
            }
        }
    }

    void regularize(const Tensor& param,
                    const double& regc) {
        if (regc > 0) {
            param.dw += (regc * param.w);
        }
    }

    void normalize_gradient(const Tensor& param,
                            const double& norm_threshold) {
        double norm = (double)(Array)param.dw.L2_norm();
        if (norm > norm_threshold) {
            param.dw = (norm_threshold / norm) * param.dw;
        }
    }
}  // namespace tensor_ops
