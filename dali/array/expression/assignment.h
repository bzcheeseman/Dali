#ifndef DALI_ARRAY_EXPRESSION_ASSIGNMENT_H
#define DALI_ARRAY_EXPRESSION_ASSIGNMENT_H

#include "dali/array/array.h"
#include "dali/array/expression/expression.h"

struct Assignment : public Expression {
    Array left_;
    OPERATOR_T operator_t_;
    Array right_;

    virtual expression_ptr copy() const;
    Assignment(Array left, OPERATOR_T operator_t, Array right);
    Assignment(const Assignment& other);
    virtual memory::Device preferred_device() const;
    virtual std::vector<Array> arguments() const;

};

#endif  // DALI_ARRAY_EXPRESSION_ASSIGNMENT_H
