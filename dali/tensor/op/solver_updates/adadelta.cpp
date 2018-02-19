#include "adadelta.h"

#include "dali/array/op/binary.h"
#include "dali/array/op/unary.h"
#include "dali/utils/assert2.h"
#include "dali/utils/make_message.h"

namespace tensor_ops {
    void adadelta_update(Tensor& param,
                         Array& gsum,
                         Array& xsum,
                         const double& rho,
                         const double& smooth_eps) {
        ASSERT2(gsum.number_of_elements() == param.number_of_elements(), utils::make_message(
            "adadelta_update's gsum parameter has a different size than parameter (got ",
            gsum.number_of_elements(), " and expected ", param.number_of_elements(), ")."));
        ASSERT2(xsum.number_of_elements() == param.number_of_elements(), utils::make_message(
            "adadelta_update's xsum parameter has a different size than parameter (got ",
            xsum.number_of_elements(), " and expected ", param.number_of_elements(), ")."));
        // update gradient cache using decay rule:
        gsum = (rho * gsum + (1.0 - rho) * op::square(param.dw));
        // DEBUG_ASSERT_POSITIVE((MAT(gsum).array()  + this->smooth_eps).matrix());
        // DEBUG_ASSERT_POSITIVE(((xsum.array() + this->smooth_eps) / (MAT(gsum).array() + this->smooth_eps)).matrix());
        Array dparam(
            param.dw * (op::sqrt(xsum + smooth_eps) / op::sqrt(gsum + smooth_eps))
        );

        xsum = rho * xsum + (1.0 - rho) * op::square(dparam);
        // update gradient using AdaDelta rule
        param.w -= dparam;
    }
}  // namespace tensor_ops
