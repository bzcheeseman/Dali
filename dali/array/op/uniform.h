#ifndef DALI_ARRAY_EXPRESSION_UNIFORM_H
#define DALI_ARRAY_EXPRESSION_UNIFORM_H

#include "dali/array/array.h"

namespace op {
	struct Uniform : public Expression {
        Array low_;
        Array high_;
        Uniform(Array left, Array right, const std::vector<int>& shape);
        std::vector<Array> arguments() const;
        using Expression::copy;
        virtual expression_ptr copy() const;
        memory::Device preferred_device() const;
        virtual bool supports_operator(OPERATOR_T operator_t) const;
    };
	Array uniform(Array low, Array high, const std::vector<int>& shape);
}

#endif  // DALI_ARRAY_EXPRESSION_UNIFORM_H
