#ifndef DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H
#define DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H

#include <memory>
#include <vector>

#include "dali/array/expression/expression.h"
#include "dali/array/memory/synchronized_memory.h"

struct ScalarView : public Expression {
    double value_;

    ScalarView(double value);
    ScalarView(const ScalarView& other);

    virtual expression_ptr copy() const;
    virtual std::vector<Array> arguments() const;
    virtual bool spans_entire_memory() const;
    virtual memory::Device preferred_device() const;

};

#endif  // DALI_ARRAY_EXPRESSION_SCALAR_VIEW_H
