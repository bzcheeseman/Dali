#ifndef DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
#define DALI_ARRAY_EXPRESSION_JIT_RUNNER_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"

struct JITNode : public Expression {
    using Expression::Expression;
};

struct JITRunner : public JITNode {
    Array root_;
    std::vector<Array> leaves_;

    virtual expression_ptr copy() const;
    JITRunner(Array root, const std::vector<Array>& leaves);
    virtual std::vector<Array> arguments() const;
    virtual memory::Device preferred_device() const;

};

#endif // DALI_ARRAY_EXPRESSION_JIT_RUNNER_H
