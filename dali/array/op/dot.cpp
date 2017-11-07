#include "dot.h"
#include "dali/utils/make_message.h"
#include "dali/array/expression/expression.h"
#include "dali/array/expression/computation.h"


/* Expression Graph utilities */
// TODO(jonathan): move to generic location

memory::Device device_promotion(const Array& a, const Array& b) {
    auto apref = a.preferred_device();
    auto bpref = b.preferred_device();
    if (apref == bpref) {return apref;}
    return memory::default_preferred_device;
}

DType type_promotion(const Array& a, const Array& b) {
    // TODO(jonathan,szymon) speed up this function
    bool a_scalar = a.is_scalar();
    bool b_scalar = b.is_scalar();

    if ((a_scalar ^ b_scalar) == 0) {
        // if they are both scalars or both arrays
        if (a.dtype() == DTYPE_DOUBLE || b.dtype() == DTYPE_DOUBLE) {
            return DTYPE_DOUBLE;
        } else if (a.dtype() == DTYPE_FLOAT || b.dtype() == DTYPE_FLOAT) {
            return DTYPE_FLOAT;
        } else {
            return DTYPE_INT32;
        }
    } else if (a_scalar) {
        // if a is scalar and b is array.
        return b.dtype();
    } else {
        // if a is array and b is scalar.
        return a.dtype();
    }
}

Array ascontiguousarray_or_simple_transpose(Array node) {
    auto buff = node.buffer_arg();
    if (!buff.is_stateless() && (buff.contiguous_memory() or buff.is_transpose())) {
        return node;
    }
    return node.ascontiguousarray();
}

// DOT SPECIFIC CLASSES

struct MatMul : public Expression {
    Array left_;
    Array right_;

    MatMul(Array left, Array right) :
        Expression({left.shape()[0], right.shape()[1]},
                   type_promotion(left, right)),
                   left_(left), right_(right) {}

    std::vector<Array> arguments() const {
        return {left_, right_};
    }

    virtual std::shared_ptr<Expression> copy() const {
        return std::make_shared<MatMul>(*this);
    }

    memory::Device preferred_device() const {
        return device_promotion(left_, right_);
    }
};


struct MatMulImpl : public Computation {
    using Computation::Computation;
    void run() {
        std::cout << "MatMulImpl is running " << std::endl;
    }
};


struct IMatMulImpl : public Computation {
    using Computation::Computation;
    void run() {
        std::cout << "IMatMulImpl is running " << std::endl;
    }
};

int impl = register_implementation(
    typeid(MatMul).name(),
    [](Array dest, OPERATOR_T operator_t, Array x) -> std::shared_ptr<Computation> {
        if (x.dtype() == DTYPE_FLOAT || x.dtype() == DTYPE_DOUBLE) {
            return std::make_shared<MatMulImpl>(dest, operator_t, x);
        } else if (x.dtype() == DTYPE_INT32) {
            return std::make_shared<IMatMulImpl>(dest, operator_t, x);
        } else {
            throw std::runtime_error("no implementation found.");
        }
    }
);

namespace op {
    Array tensordot_as_dot(Array a, Array b,
                           const std::vector<int>& a_reduce_axes,
                           const std::vector<int>& b_reduce_axes) {
        throw std::runtime_error("not implemented yet");
    }

    Array dot(Array a, Array b) {
        int a_ndim = a.ndim();
        int b_ndim = b.ndim();
        if (a_ndim == 2 && b_ndim == 2) {
            a = ascontiguousarray_or_simple_transpose(a);
            b = ascontiguousarray_or_simple_transpose(b);
            return Array(std::make_shared<MatMul>(a, b));
        } else if (a_ndim > 2 or b_ndim > 2) {
            return tensordot_as_dot(a, b,
                                    /*a_reduce_axes=*/{a_ndim - 1,},
                                    /*b_reduce_axes=*/{b_ndim - 1,});
        } else {
            throw std::runtime_error(utils::make_message(
                "dot not implemented yet for a.ndim = ", a_ndim,
                ", b.ndim = ", b_ndim));
        }
    }
}
