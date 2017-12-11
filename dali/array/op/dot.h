#ifndef DALI_ARRAY_OP_DOT_H
#define DALI_ARRAY_OP_DOT_H

#include "dali/array/array.h"
#include <vector>

namespace op {
    struct MatMul : public Expression {
        Array left_;
        Array right_;
        MatMul(Array left, Array right);
        std::vector<Array> arguments() const;
        virtual std::shared_ptr<Expression> copy() const;
        memory::Device preferred_device() const;
    };

    Array dot(Array a, Array b);
    Array tensordot_as_dot(Array a, Array b, const std::vector<int>& a_reduce_axes,
                                             const std::vector<int>& b_reduce_axes);
}

#endif
