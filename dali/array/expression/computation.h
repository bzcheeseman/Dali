#ifndef DALI_ARRAY_EXPRESSION_COMPUTATION_H
#define DALI_ARRAY_EXPRESSION_COMPUTATION_H

#include "dali/array/expression/operator.h"
#include "dali/array/array.h"

struct Computation {
    Array left_;
    OPERATOR_T operator_t_;
    Array right_;
    Array assignment_;

    Computation(Array left, OPERATOR_T operator_t, Array right, Array assignment);
    virtual void run() = 0;
    void run_and_cleanup();
};

typedef std::function<std::shared_ptr<Computation>(Array, OPERATOR_T, Array, Array) > to_computation_t;

int register_implementation(const char*, to_computation_t impl);
std::vector<std::shared_ptr<Computation>> convert_to_ops(Array root);


#endif // DALI_ARRAY_EXPRESSION_COMPUTATION_H
