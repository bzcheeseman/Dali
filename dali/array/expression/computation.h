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
    void* left_data(memory::Device);
    const void* argument_data(memory::Device device, int idx) const;
};

typedef std::function<std::shared_ptr<Computation>(Array, OPERATOR_T, Array, Array) > to_computation_t;
typedef std::function<bool(const Array&)> implementation_test_t;

int register_implementation(const char* opname, to_computation_t impl);
// register an implementation for opname, depending on the implementation_test_t. If implementation_test_t is true,
// then use to_computation_t to pick an implementation.
int register_implementation(const std::string& opname, implementation_test_t test, to_computation_t impl);

std::vector<std::shared_ptr<Computation>> convert_to_ops(Array root);

template<typename Expression, typename Comp>
int register_implementation_default() {
	return register_implementation(
        typeid(Expression).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            return std::make_shared<Comp>(dest, operator_t, x, assignment);
        }
    );
}

#endif // DALI_ARRAY_EXPRESSION_COMPUTATION_H
