#include "tensor.h"

#include "dali/array/op.h"
#include "dali/tensor/op.h"
#include "dali/tensor/op/other.h"
#include "dali/tensor/op/dot.h"

// #include "dali/tensor/Index.h"

using std::vector;
using std::string;
using std::stringstream;
using utils::assert2;
using std::make_shared;

Tensor::Tensor(const Array& w_, const Array& dw_, bool constant_) :
        w(w_), dw(dw_), constant(constant_), name(nullptr) {
}

// this does not need to initialize anything once we get rid of w and dw.
Tensor::Tensor() {
}

Tensor::Tensor(const std::initializer_list<int>& shape,
               DType dtype,
               memory::Device preferred_device) :
    Tensor(Array::zeros(shape, dtype, preferred_device),
           Array::zeros(shape, dtype, preferred_device), false) {
}

Tensor::Tensor(const Array& other, bool copy) {
    if (copy && !other.is_stateless()) {
        w = Array(other, true);
    } else {
        w = other;
    }
    dw = Array::zeros_like(w);
}

Tensor::Tensor(const Tensor& other, bool copy_w, bool copy_dw) :
        name(other.name),
        constant(other.constant) {
    if (copy_w && !other.w.is_stateless()) {
        // This copies memory using copy constructor
        // The copy is only executed if matrix was actually initialized
        // hence the && other.m part.
        w = Array(other.w, true);
    } else {
        // This does not.
        w = other.w;
    }

    if (copy_dw && !other.dw.is_stateless()) {
        // see comment for copy_w.
        dw = Array(other.dw, true);
    } else {
        dw = other.dw;
    }
}

Tensor::~Tensor() {}

void Tensor::copy_from(const Tensor& source) {
    w = Array(source.w, true);
}

void Tensor::copy_grad_from(const Tensor& source) {
    dw = Array(source.dw, true);
}

void Tensor::print(std::basic_ostream<char>& stream) const {
    if (!w.is_stateless()) {
        w.print(stream);
    } else {
        stream << "<stateless tensor>";
    }
}

void Tensor::grad() const {
    dw += 1;
}

void Tensor::clear_grad() const {
    dw.clear();
}

void Tensor::clear() {
    w.clear();
    dw.clear();
}

const std::vector<int>& Tensor::shape() const {
    return w.shape();
}
const std::vector<int>& Tensor::strides() const {
    return w.strides();
}

DType Tensor::dtype() const {
    return w.dtype();
}

memory::Device Tensor::preferred_device() const {
    return w.preferred_device();
}

int Tensor::ndim() const {
    return w.ndim();
}
int Tensor::number_of_elements() const {
    return w.number_of_elements();
}

bool Tensor::is_stateless() const {
    return w.is_stateless();
}

bool Tensor::is_scalar() const {
    return w.is_scalar();
}

bool Tensor::is_vector() const {
    return w.is_vector();
}

bool Tensor::is_matrix() const {
    return w.is_matrix();
}

void Tensor::set_name(string& _name) {
    if (name != nullptr) {
        *name = _name;
    } else {
        name = std::make_shared<string>(_name);
    }
}

void Tensor::set_name(char * _name) {
    if (name != nullptr) {
        *name = _name;
    } else {
        name = std::make_shared<string>(_name);
    }
}

void Tensor::set_name(const char * _name) {
    if (name != nullptr) {
        *name = _name;
    } else {
        name = std::make_shared<string>(_name);
    }
}

Tensor Tensor::shallow_copy() {
    return Tensor(*this, false, true);
}

#define TENSOR_UNARY_OP( opname, RTYPE) \
    RTYPE Tensor::opname() const {return tensor_ops::opname(*this);}\

#define TENSOR_UNARY_OP_WITH_ARG( opname, ARGTYPE, RTYPE) \
    RTYPE Tensor::opname(const ARGTYPE& argname) const {return tensor_ops::opname(*this, argname);}\

#define TENSOR_UNARY_OP_WITH_VECINT_BOOL_ARG( opname, RTYPE) \
    RTYPE Tensor::opname(const std::vector<int>& axes, bool keepdims) const {return tensor_ops::opname(*this, axes, keepdims);}\

TENSOR_UNARY_OP_WITH_ARG(dot, Tensor, Tensor);
TENSOR_UNARY_OP_WITH_ARG(astype, DType, Tensor);
TENSOR_UNARY_OP(square, Tensor);
TENSOR_UNARY_OP(cube, Tensor);
TENSOR_UNARY_OP(is_grad_nan, bool);
TENSOR_UNARY_OP(is_nan, bool);
TENSOR_UNARY_OP(sqrt, Tensor);
TENSOR_UNARY_OP(rsqrt, Tensor);
TENSOR_UNARY_OP(eltinv, Tensor);
TENSOR_UNARY_OP(tanh, Tensor);
TENSOR_UNARY_OP(softplus, Tensor);
TENSOR_UNARY_OP(sigmoid, Tensor);
TENSOR_UNARY_OP(sum, Tensor);
TENSOR_UNARY_OP_WITH_VECINT_BOOL_ARG(sum, Tensor);
TENSOR_UNARY_OP(mean, Tensor);
TENSOR_UNARY_OP_WITH_VECINT_BOOL_ARG(mean, Tensor);
TENSOR_UNARY_OP(min, Tensor);
TENSOR_UNARY_OP_WITH_VECINT_BOOL_ARG(min, Tensor);
TENSOR_UNARY_OP(max, Tensor);
TENSOR_UNARY_OP_WITH_VECINT_BOOL_ARG(max, Tensor);
TENSOR_UNARY_OP(L2_norm, Tensor);
TENSOR_UNARY_OP_WITH_VECINT_BOOL_ARG(L2_norm, Tensor);
TENSOR_UNARY_OP(argsort, Tensor);
TENSOR_UNARY_OP_WITH_ARG(argsort, int, Tensor);
TENSOR_UNARY_OP(argmax, Tensor);
TENSOR_UNARY_OP_WITH_ARG(argmax, int, Tensor);
TENSOR_UNARY_OP(argmin, Tensor);
TENSOR_UNARY_OP_WITH_ARG(argmin, int, Tensor);
TENSOR_UNARY_OP(log, Tensor);
TENSOR_UNARY_OP(exp, Tensor);
TENSOR_UNARY_OP(abs, Tensor);
TENSOR_UNARY_OP(relu, Tensor);
TENSOR_UNARY_OP_WITH_ARG(reshape, std::vector<int>, Tensor);
TENSOR_UNARY_OP_WITH_ARG(right_fit_ndim, int, Tensor);

Tensor Tensor::operator[](int idx) const {
    return pluck_axis(0, idx);
}

Tensor Tensor::operator[](const Tensor& indices) const {
    return tensor_ops::gather(*this, indices);
}

Tensor Tensor::operator[](const std::vector<int>& indices) const {
    Array indices_arr({(int)indices.size()}, DTYPE_INT32);
    indices_arr = indices;
    return tensor_ops::gather(*this, indices_arr);
}

Tensor Tensor::operator[](const std::initializer_list<int>& indices) const {
    Array indices_arr({(int)indices.size()}, DTYPE_INT32);
    indices_arr = indices;
    return tensor_ops::gather(*this, indices_arr);
}

SlicingInProgress<Tensor> Tensor::operator[](const Slice& s) const {
    auto ret = SlicingInProgress<Tensor>(*this);
    return ret[s];
}

SlicingInProgress<Tensor> Tensor::operator[](const Broadcast& b) const {
    auto ret = SlicingInProgress<Tensor>(*this);
    return ret[b];
}

Tensor Tensor::pluck_axis(int axis, const Slice& slice) const {
    return Tensor::from_w_and_dw(w.pluck_axis(axis, slice), dw.pluck_axis(axis, slice), constant);
}

Tensor Tensor::pluck_axis(int axis, int idx) const {
    return Tensor::from_w_and_dw(w.pluck_axis(axis, idx), dw.pluck_axis(axis, idx), constant);
}

Tensor Tensor::squeeze(int axis) const {
    return Tensor::from_w_and_dw(w.squeeze(axis), dw.squeeze(axis), constant);
}

Tensor Tensor::expand_dims(int new_axis) const {
    return Tensor::from_w_and_dw(w.expand_dims(new_axis), dw.expand_dims(new_axis), constant);
}

Tensor Tensor::dimshuffle(const std::vector<int>& axes) const {
    return Tensor::from_w_and_dw(w.dimshuffle(axes), dw.dimshuffle(axes), constant);
}

Tensor Tensor::transpose() const {
    return Tensor::from_w_and_dw(w.transpose(), dw.transpose(), constant);
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
    return Tensor::from_w_and_dw(w.transpose(axes), dw.transpose(axes), constant);
}

Tensor Tensor::swapaxes(const int& axis1, const int& axis2) const {
    return Tensor::from_w_and_dw(w.swapaxes(axis1, axis2), dw.swapaxes(axis1, axis2), constant);
}

Tensor Tensor::broadcast_scalar_to_ndim(int ndim) const {
    return Tensor::from_w_and_dw(w.broadcast_scalar_to_ndim(ndim),
                                 dw.broadcast_scalar_to_ndim(ndim),
                                 constant);
}

Tensor Tensor::ravel() const {
    return tensor_ops::ravel(*this);
}

// TODO(jonathan): recover the gather operations for Tensor:
// Tensor Tensor::col(int col) {
//     return TensorOps::col_pluck(*this, col);
// }
//
// Tensor Tensor::operator[](
//         Indexing::Index indices) const {
//     return TensorOps::rows_pluck(*this, indices);
// }
//
// Tensor Tensor::operator[](
//         Tensor<int> indices) const {
//     return TensorOps::rows_pluck(*this, indices);
// }
//
// Tensor Tensor::operator()(
//         Indexing::Index indices) const {
//     return TensorOps::rows_pluck(*this, indices);
// }
//
// Tensor Tensor::operator()(
//         Indexing::Index row_indices,
//         Indexing::Index col_indices) const {
//     return TensorOps::rows_cols_pluck(*this, row_indices, col_indices);
// }
//
// Tensor Tensor::operator[](
//         int row) const {
//     return TensorOps::row_pluck(*this, row);
// }
// Tensor Tensor::operator()(
//         int row) const {
//     return TensorOps::row_pluck(*this, row);
// }
//
// Tensor Tensor::operator()(
//         void* nothing,
//         int col) const {
//     return TensorOps::col_pluck(*this, col);
// }
// Tensor Tensor::operator^(Tensor other) const {
//     return TensorOps::pow(*this, other);
// }
//

Tensor Tensor::zeros_like(const Tensor& other) {
    return Tensor(Array::zeros_like(other.w));
}

Tensor Tensor::ones_like(const Tensor& other) {
    return Tensor(Array::ones_like(other.w));
}

Tensor Tensor::empty_like(const Tensor& other) {
    return Tensor(Array::empty_like(other.w));
}

Tensor Tensor::fill_like(const double& scalar, const Tensor& other) {
    Array out = Array::empty_like(other.w);
    out = scalar;
    return Tensor(out);
}

Tensor Tensor::zeros(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    return Tensor(Array::zeros(shape, dtype, preferred_device));
}

Tensor Tensor::ones(const std::vector<int>& shape,
                    const DType& dtype,
                    const memory::Device& preferred_device) {
    return Tensor(Array::ones(shape, dtype, preferred_device));
}

Tensor Tensor::empty(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Tensor(Array(shape, dtype, preferred_device));
}

Tensor Tensor::normal(const Array& loc,
                      const Array& scale,
                      const std::vector<int>& shape) {
    return Tensor(op::normal(loc, scale, shape), false);
}

Tensor Tensor::uniform(const Array& low,
                       const Array& high,
                       const std::vector<int>& shape) {
    return Tensor(op::uniform(low, high, shape), false);
}

Tensor Tensor::bernoulli(const Array& prob,
                         const std::vector<int>& shape) {
    return Tensor(op::bernoulli(prob, shape), false);
}

Tensor Tensor::bernoulli_normalized(const Array& prob,
                                    const std::vector<int>& shape) {
    return Tensor(op::bernoulli_normalized(prob, shape), false);
}

Tensor Tensor::fill(const double& scalar,
                    const std::vector<int>& shape,
                    const DType& dtype,
                    const memory::Device& preferred_device) {
    Array out(shape, dtype, preferred_device);
    out = scalar;
    return Tensor(out);
}

Tensor Tensor::load(FILE * fp) {
    auto loaded = Array::load(fp);
    return Tensor(loaded, Array::zeros_like(loaded), false);
}

Tensor Tensor::load(const std::string& fname) {
    auto loaded = Array::load(fname);
    return Tensor(loaded, Array::zeros_like(loaded), false);
}

void Tensor::save(const std::string& fname, const Tensor& arr, const std::ios_base::openmode& mode) {
    Array::save(fname, arr.w, mode);
}

void Tensor::save(std::basic_ostream<char>& stream, const Tensor& arr) {
    Array::save(stream, arr.w);
}


Tensor Tensor::from_w_and_dw(const Array& w, const Array& dw, bool constant) {
    return Tensor(w, dw, constant);
}

void Tensor::to_device(memory::Device device) const {
    w.to_device(device);
    dw.to_device(device);
}

void Tensor::to_cpu() const {
    to_device(memory::Device::cpu());
}

void Tensor::to_gpu(int number) const {
#ifdef DALI_USE_CUDA
    to_device(memory::Device::gpu(number));
#else
    ASSERT2(false, "Dali compiled without CUDA support, cannot move Tensor to gpu.");
#endif
}


std::ostream &operator <<(std::ostream &os, const Tensor& tensor) {
    os << "Tensor(" << tensor.shape() << ", dtype=" << tensor.dtype() << ")";
    return os;
}


#define DALI_DEFINE_TENSOR_INTERACTION_INPLACE(SYMBOL, OPERATION_NAME)\
    Tensor& operator SYMBOL (Tensor& left, const Tensor& right) {\
        auto assignment = OPERATION_NAME(left, right);\
        left = assignment;\
        return left;\
    }\
    void operator SYMBOL (Tensor&& left, const Tensor& right) {\
        auto assignment = OPERATION_NAME(left, right);\
        left = assignment;\
    }\

DALI_DEFINE_TENSOR_INTERACTION_INPLACE(+=, tensor_ops::add);
DALI_DEFINE_TENSOR_INTERACTION_INPLACE(-=, tensor_ops::subtract);
DALI_DEFINE_TENSOR_INTERACTION_INPLACE(*=, tensor_ops::eltmul);
DALI_DEFINE_TENSOR_INTERACTION_INPLACE(/=, tensor_ops::eltdiv);
DALI_DEFINE_TENSOR_INTERACTION_INPLACE(^=, tensor_ops::pow);

#define DALI_DEFINE_TENSOR_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, DTYPE)\
    Tensor operator SYMBOL (const Tensor& left, DTYPE right) {\
        return OPERATION_NAME(left, Array(right));\
    }\
    Tensor operator SYMBOL (DTYPE left, const Tensor& right) {\
        return OPERATION_NAME(Array(left), right);\
    }

#define DALI_DEFINE_TENSOR_INTERACTION(SYMBOL, OPERATION_NAME)\
    Tensor operator SYMBOL (const Tensor& left, const Tensor& right) {\
        return OPERATION_NAME(left, right);\
    }\
    DALI_DEFINE_TENSOR_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, double);\
    DALI_DEFINE_TENSOR_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, float);\
    DALI_DEFINE_TENSOR_SCALAR_INTERACTION(SYMBOL, OPERATION_NAME, int);\

DALI_DEFINE_TENSOR_INTERACTION(+, tensor_ops::add);
DALI_DEFINE_TENSOR_INTERACTION(-, tensor_ops::subtract);
DALI_DEFINE_TENSOR_INTERACTION(*, tensor_ops::eltmul);
DALI_DEFINE_TENSOR_INTERACTION(/, tensor_ops::eltdiv);
DALI_DEFINE_TENSOR_INTERACTION(^, tensor_ops::pow);
