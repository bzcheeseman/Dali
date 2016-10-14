#include "tensor.h"

#include "dali/array/op.h"
#include "dali/array/op/initializer.h"
#include "dali/tensor/op.h"
#include "dali/tensor/op/other.h"
#include "dali/tensor/op/dot.h"

// #include "dali/tensor/Index.h"

using std::vector;
using std::string;
using std::stringstream;
using utils::assert2;
using std::make_shared;

////////////////////////////////////////////////////////////////////////////////
//                             TENSOR                                         //
////////////////////////////////////////////////////////////////////////////////
Tensor::Tensor(const std::vector<int>& shape,
               Assignable<Array> weights_initialization,
               DType dtype_,
               memory::Device preferred_device) :
        w(shape,dtype_,preferred_device),
        dw(shape,dtype_,preferred_device) {
    w = weights_initialization;
    dw.clear();
}


Tensor::Tensor(const Array& w_, const Array& dw_, bool constant_) :
        w(w_), dw(dw_), constant(constant_), name(nullptr) {
}

// this does not need to initialize anything once we get rid of w and dw.
Tensor::Tensor() {
}

Tensor::Tensor(const std::initializer_list<int>& shape,
       DType dtype_,
       memory::Device preferred_device) :
   Tensor(shape,initializer::zeros(),dtype_,preferred_device) {
}

Tensor::Tensor(const std::vector<int>& shape,
       DType dtype_,
       memory::Device preferred_device) :
   Tensor(shape,initializer::zeros(),dtype_,preferred_device) {
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

DType Tensor::dtype() const {
    return w.dtype();
}

Tensor Tensor::astype(const DType& dtype) const {
    return tensor_ops::astype(*this, dtype);
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
Tensor Tensor::vectorlike_to_vector() const {
    return Tensor(w.vectorlike_to_vector(),
                  dw.vectorlike_to_vector(),
                  constant);
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





// Tensor Tensor::eltmul(R alpha) const {
//     return TensorOps::eltmul(*this, alpha);
// }
//
// #define MAT_OP_SPECIALIZATION(fname, opname, R, ScalarType) \
//         template<>                                          \
//         template<>                                          \
//         Tensor Tensor::fname(ScalarType power) const {      \
//             return TensorOps::opname(*this, (R)power);      \
//         }
//
// #define MAT_OP_SPECIALIZATIONS(fname, opname)              \
//         MAT_OP_SPECIALIZATION(fname,opname,float,float);   \
//         MAT_OP_SPECIALIZATION(fname,opname,float,double);  \
//         MAT_OP_SPECIALIZATION(fname,opname,float,int);     \
//         MAT_OP_SPECIALIZATION(fname,opname,double,float);  \
//         MAT_OP_SPECIALIZATION(fname,opname,double,double); \
//         MAT_OP_SPECIALIZATION(fname,opname,double,int);    \
//         MAT_OP_SPECIALIZATION(fname,opname,int,float);     \
//         MAT_OP_SPECIALIZATION(fname,opname,int,double);    \
//         MAT_OP_SPECIALIZATION(fname,opname,int,int);
//
// MAT_OP_SPECIALIZATIONS(pow,pow);
// MAT_OP_SPECIALIZATIONS(operator^,pow);
//
// Tensor Tensor::steep_sigmoid(R aggressiveness) const {
//     return TensorOps::steep_sigmoid(*this, aggressiveness);
// }
//
bool Tensor::is_nan() const {
    return tensor_ops::is_nan(*this);
}

bool Tensor::is_grad_nan() const {
    return tensor_ops::is_grad_nan(*this);
}
//
//
// #define MAT_BINARY_OP( opname ) \
//  \
//     Tensor Tensor::opname(Tensor matrix) const {\
//         return TensorOps::opname(*this, matrix);\
//     }
//
// MAT_BINARY_OP( eltmul_broadcast_colwise )
// MAT_BINARY_OP( eltmul )
// MAT_BINARY_OP( eltmul_broadcast_rowwise )
// MAT_BINARY_OP( eltmul_rowwise )
// MAT_BINARY_OP( add )
// MAT_BINARY_OP( sub )
// MAT_BINARY_OP( add_broadcast_rowwise )
// MAT_BINARY_OP( add_broadcast_colwise )
// MAT_BINARY_OP( sub_broadcast )
// MAT_BINARY_OP( sub_broadcast_reversed )
// MAT_BINARY_OP( mul )
//

Tensor Tensor::dot(const Tensor& other) const {
    return tensor_ops::dot(*this, other);
}

#define TENSOR_UNARY_OP( opname ) \
    Tensor Tensor::opname() const {\
        return tensor_ops::opname(*this);\
    }\

#define TENSOR_UNARY_OP_WITH_INT_ARG( opname ) \
    Tensor Tensor::opname(const int& argname) const {\
        return tensor_ops::opname(*this, argname);\
    }\

#define TENSOR_UNARY_OP_WITH_DOUBLE_ARG( opname ) \
    Tensor Tensor::opname(const double& argname) const {\
        return tensor_ops::opname(*this, argname);\
    }\

TENSOR_UNARY_OP(square);
TENSOR_UNARY_OP(cube);
TENSOR_UNARY_OP(sqrt);
TENSOR_UNARY_OP(rsqrt);
TENSOR_UNARY_OP(eltinv);
TENSOR_UNARY_OP(tanh);
TENSOR_UNARY_OP(softplus);
TENSOR_UNARY_OP(sigmoid);
TENSOR_UNARY_OP_WITH_DOUBLE_ARG(steep_sigmoid);
TENSOR_UNARY_OP(sum);
TENSOR_UNARY_OP_WITH_INT_ARG(sum);
TENSOR_UNARY_OP(mean);
TENSOR_UNARY_OP_WITH_INT_ARG(mean);
TENSOR_UNARY_OP(min);
TENSOR_UNARY_OP_WITH_INT_ARG(min);
TENSOR_UNARY_OP(max);
TENSOR_UNARY_OP_WITH_INT_ARG(max);
TENSOR_UNARY_OP(L2_norm);
TENSOR_UNARY_OP_WITH_INT_ARG(L2_norm);
TENSOR_UNARY_OP(argsort);
TENSOR_UNARY_OP_WITH_INT_ARG(argsort);
TENSOR_UNARY_OP(argmax);
TENSOR_UNARY_OP_WITH_INT_ARG(argmax);
TENSOR_UNARY_OP(argmin);
TENSOR_UNARY_OP_WITH_INT_ARG(argmin);
TENSOR_UNARY_OP(log);
TENSOR_UNARY_OP(exp);
TENSOR_UNARY_OP(abs);
TENSOR_UNARY_OP(relu);

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

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    return tensor_ops::reshape(*this, new_shape);
}

Tensor Tensor::copyless_reshape(const std::vector<int>& new_shape) const {
    return Tensor::from_w_and_dw(w.copyless_reshape(new_shape), dw.copyless_reshape(new_shape), constant);
}

Tensor Tensor::right_fit_ndim(const int& dimensionality) const {
    return tensor_ops::right_fit_ndim(*this, dimensionality);
}

Tensor Tensor::copyless_right_fit_ndim(const int& dimensionality) const {
    return Tensor::from_w_and_dw(
        w.copyless_right_fit_ndim(dimensionality),
        dw.copyless_right_fit_ndim(dimensionality),
        constant
    );
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

Tensor Tensor::broadcast_axis(int axis) const {
    return Tensor::from_w_and_dw(w.broadcast_axis(axis), dw.broadcast_axis(axis), constant);
}

Tensor Tensor::insert_broadcast_axis(int new_axis) const {
    return Tensor::from_w_and_dw(w.insert_broadcast_axis(new_axis), dw.insert_broadcast_axis(new_axis), constant);
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
    return Tensor::from_w_and_dw(
        w.swapaxes(axis1, axis2),
        dw.swapaxes(axis1, axis2),
        constant
    );
}

Tensor Tensor::broadcast_scalar_to_ndim(int ndim) const {
    return Tensor::from_w_and_dw(w.broadcast_scalar_to_ndim(ndim),
                  dw.broadcast_scalar_to_ndim(ndim),
                  constant);
}

Tensor Tensor::copyless_ravel() const {
    return Tensor::from_w_and_dw(w.copyless_ravel(), dw.copyless_ravel(), constant);
}

Tensor Tensor::ravel() const {
    return tensor_ops::ravel(*this);
}

//
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


//
//
// Tensor Tensor::operator+(Tensor other) const {
//     return TensorOps::add(*this, other);
// }
//
// Tensor Tensor::operator+(R other) const {
//     return TensorOps::add(*this, other);
// }
//
// Tensor& Tensor::operator+=(Tensor other) {
//     auto sum = TensorOps::add(*this, other);
//     this->m = sum.m;
//     this->g = sum.g;
//     return *this;
// }
//
// Tensor& Tensor::operator+=(R other) {
//     auto sum = TensorOps::add(*this, other);
//     this->m = sum.m;
//     this->g = sum.g;
//     return *this;
// }
//
// Tensor Tensor::operator-(Tensor other) const {
//     return TensorOps::sub(*this, other);
// }
//
// Tensor Tensor::operator-(R other) const {
//     return TensorOps::add(*this, -other);
// }
//
// Tensor& Tensor::operator-=(Tensor other) {
//     auto diff = TensorOps::sub(*this, other);
//     this->m = diff.m;
//     this->g = diff.g;
//     return *this;
// }
//
// Tensor& Tensor::operator-=(R other) {
//     auto diff = TensorOps::add(*this, -other);
//     this->m = diff.m;
//     this->g = diff.g;
//     return *this;
// }
//
// Tensor Tensor::operator*(Tensor other) const {
//     return TensorOps::eltmul(*this, other);
// }
//
// Tensor Tensor::operator*(R alpha) const {
//     return TensorOps::eltmul(*this, alpha);
// }
//
// Tensor& Tensor::operator*=(Tensor other) {
//     auto prod = TensorOps::eltmul(*this, other);
//     this->m = prod.m;
//     this->g = prod.g;
//     return *this;
// }
//
// Tensor& Tensor::operator*=(R other) {
//     auto prod = TensorOps::eltmul(*this, other);
//     this->m = prod.m;
//     this->g = prod.g;
//     return *this;
// }
//
//
// Tensor Tensor::operator-() const {
//     return (*this) * -1;
// }
//
// Tensor Tensor::operator/(Tensor other) const {
//     return TensorOps::eltdivide(*this, other);
// }
//
// Tensor Tensor::operator/(R alpha) const {
//     return TensorOps::eltdivide(*this, alpha);
// }
//
// Tensor& Tensor::operator/=(Tensor other) {
//     auto divided = TensorOps::eltdivide(*this, other);
//     this->m = divided.m;
//     this->g = divided.g;
//     return *this;
// }
//
// Tensor& Tensor::operator/=(R other) {
//     auto divided = TensorOps::eltdivide(*this, other);
//     this->m = divided.m;
//     this->g = divided.g;
//     return *this;
// }
//
// Tensor Tensor::operator^(Tensor other) const {
//     return TensorOps::pow(*this, other);
// }
//

Tensor Tensor::zeros_like(const Tensor& other) {
    return Tensor(other.shape(), initializer::zeros(), other.dtype(), other.preferred_device());
}

Tensor Tensor::ones_like(const Tensor& other) {
    return Tensor(other.shape(), initializer::ones(), other.dtype(), other.preferred_device());
}

Tensor Tensor::empty_like(const Tensor& other) {
    return Tensor(other.shape(), initializer::empty(), other.dtype(), other.preferred_device());
}

Tensor Tensor::fill_like(const double& scalar, const Tensor& other) {
    return Tensor(other.shape(), initializer::fill(scalar), other.dtype(), other.preferred_device());
}

Tensor Tensor::zeros(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    return Tensor(shape, initializer::zeros(), dtype, preferred_device);
}

Tensor Tensor::ones(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    return Tensor(shape, initializer::ones(), dtype, preferred_device);
}

Tensor Tensor::empty(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Tensor(shape, initializer::empty(), dtype, preferred_device);
}

Tensor Tensor::arange(const std::vector<int>& shape,
                      const DType& dtype,
                      const memory::Device& preferred_device) {
    return Tensor(shape, initializer::arange(0, 1.0), dtype, preferred_device);
}

Tensor Tensor::arange(const double& start, const double& stop, const double& step,
                      const DType& dtype,
                      const memory::Device& preferred_device) {

    int length = ((stop - start) + step - 1) / (step);
    ASSERT2(length > 0,
        utils::MS() << "Tensor length must be non-zero (got start="
                    << start
                    << ", stop="
                    << stop
                    << ", step="
                    << step << ").");
    return Tensor({length}, initializer::arange(start, step), dtype, preferred_device);
}

Tensor Tensor::gaussian(const double& mean,
                        const double& std,
                        const std::vector<int>& shape,
                        const DType& dtype,
                        const memory::Device& preferred_device) {
    return Tensor(shape, initializer::gaussian(mean, std), dtype, preferred_device);
}

Tensor Tensor::uniform(const double& lower,
                       const double& upper,
                       const std::vector<int>& shape,
                       const DType& dtype,
                       const memory::Device& preferred_device) {
    return Tensor(shape, initializer::uniform(lower, upper), dtype, preferred_device);
}

Tensor Tensor::uniform(const double& limit,
                       const std::vector<int>& shape,
                       const DType& dtype,
                       const memory::Device& preferred_device) {
    return Tensor(shape, initializer::uniform(-limit, limit), dtype, preferred_device);
}

Tensor Tensor::bernoulli(const double& prob,
                         const std::vector<int>& shape,
                         const DType& dtype,
                         const memory::Device& preferred_device) {
    return Tensor(shape, initializer::bernoulli(prob), dtype, preferred_device);
}

Tensor Tensor::bernoulli_normalized(const double& prob,
                                    const std::vector<int>& shape,
                                    const DType& dtype,
                                    const memory::Device& preferred_device) {
    return Tensor(shape, initializer::bernoulli_normalized(prob), dtype, preferred_device);
}

Tensor Tensor::fill(const double& scalar,
                    const std::vector<int>& shape,
                    const DType& dtype,
                    const memory::Device& preferred_device) {
    return Tensor(shape, initializer::fill(scalar), dtype, preferred_device);
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

#ifdef DALI_USE_CUDA
    void Tensor::to_gpu(int number) const {
        to_device(memory::Device::gpu(number));
    }
#endif

// /* External operators */
// Tensor operator+(int other, Tensor mat) {
//     return TensorOps::add(mat, (R) other);
// }
// Tensor operator+(float other, Tensor mat) {
//     return TensorOps::add(mat, other);
// }
// Tensor operator+(double other, Tensor mat) {
//     return TensorOps::add(mat, other);
// }
//
//
// Tensor operator-(int other, Tensor mat) {
//     return TensorOps::sub_broadcast_reversed(mat, (R) other);
// }
// Tensor operator-(float other, Tensor mat) {
//     return TensorOps::sub_broadcast_reversed(mat, other);
// }
// Tensor operator-(double other, Tensor mat) {
//     return TensorOps::sub_broadcast_reversed(mat, other);
// }
//
//
// Tensor operator*(int other, Tensor mat) {
//     return TensorOps::eltmul(mat, (R)other);
// }
// Tensor operator*(float other, Tensor mat) {
//     return TensorOps::eltmul(mat, other);
// }
// Tensor operator*(double other, Tensor mat) {
//     return TensorOps::eltmul(mat, other);
// }
//
// template Tensor<float> operator+(int, Tensor<float>);
// template Tensor<float> operator+(float, Tensor<float>);
// template Tensor<float> operator+(double, Tensor<float>);
//
// template Tensor<double> operator+(int, Tensor<double>);
// template Tensor<double> operator+(float, Tensor<double>);
// template Tensor<double> operator+(double, Tensor<double>);
//
//
// template Tensor<float> operator-(int, Tensor<float>);
// template Tensor<float> operator-(float, Tensor<float>);
// template Tensor<float> operator-(double, Tensor<float>);
//
// template Tensor<double> operator-(int, Tensor<double>);
// template Tensor<double> operator-(float, Tensor<double>);
// template Tensor<double> operator-(double, Tensor<double>);
//
//
// template Tensor<float> operator*(int, Tensor<float>);
// template Tensor<float> operator*(float, Tensor<float>);
// template Tensor<float> operator*(double, Tensor<float>);
//
// template Tensor<double> operator*(int, Tensor<double>);
// template Tensor<double> operator*(float, Tensor<double>);
// template Tensor<double> operator*(double, Tensor<double>);
//
//
// std::ostream& operator<<(std::ostream& strm, const Tensor& a) {
//     if (a.name != nullptr) {
//         return strm << "<#Tensor name=\"" << *a.name<< "\" n=" << a.dims(0) << ", d=" << a.dims(1) << ">";
//     } else {
//         return strm << "<#Tensor n=" << a.dims(0) << ", d=" << a.dims(1) << ">";
//     }
// }
//
// template std::ostream& operator<< <float>(std::ostream& strm, const Tensor<float>& a);
// template std::ostream& operator<< <double>(std::ostream& strm, const Tensor<double>& a);
// template std::ostream& operator<< <int>(std::ostream& strm, const Tensor<int>& a);
//
// template <typename R>
// std::size_t std::hash<Tensor>::operator()(const Tensor& k) const {
//     auto ptr = &(k.w());
//     auto hasher = std::hash<decltype(ptr)>();
//     return hasher(ptr);
// }
//
// template std::size_t std::hash<Tensor<float>>::operator()(const Tensor<float>& k)   const;
// template std::size_t std::hash<Tensor<double>>::operator()(const Tensor<double>& k) const;
// template std::size_t std::hash<Tensor<int>>::operator()(const Tensor<int>& k) const;
//
// template <typename R>
// bool operator!=(const Tensor& A, const Tensor& B) {
//     return &(A.w()) != &(B.w());
// }
//
// template bool operator!=(const Tensor<float>&, const Tensor<float>&);
// template bool operator!=(const Tensor<double>&, const Tensor<double>&);
//
// template <typename R>
// bool operator==(const Tensor& A, const Tensor& B) {
//     return &(A.w()) == &(B.w());
// }
//
// template bool operator==<float>(const Tensor<float>&, const Tensor<float>&);
// template bool operator==<double>(const Tensor<double>&, const Tensor<double>&);
//
// namespace utils {
//
//     void save_matrices(vector<Tensor> parameters, string dirname) {
//         utils::ensure_directory(dirname);
//         const char * c_dirname = dirname.c_str();
//         utils::makedirs(c_dirname);
//         int i = 0;
//         for (auto& param : parameters) {
//             stringstream param_location;
//             param_location << dirname << "/param_" << i << ".npy";
//             param.npy_save(param_location.str());
//             i++;
//         }
//     }
//
//
//     void load_matrices(vector<Tensor> parameters, string dirname) {
//         utils::ensure_directory(dirname);
//         int i = 0;
//         for (auto& param : parameters) {
//             stringstream param_location;
//             param_location << dirname << "/param_" << i << ".npy";
//             param.npy_load(param_location.str());
//             i++;
//         }
//     }

std::ostream &operator <<(std::ostream &os, const Tensor& tensor) {
    os << "Tensor(" << tensor.shape() << ", dtype=" << tensor.dtype() << ")";
    return os;
}
