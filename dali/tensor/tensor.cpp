#include "dali/tensor/tensor.h"
#include "dali/array/op.h"
#include "dali/array/op/initializer.h"
// #include "dali/tensor/Index.h"

using std::vector;
using std::string;
using std::stringstream;
using utils::assert2;
using std::make_shared;

/* Tensor */
// this does not need to initialize anything once we get rid of w and dw.
Tensor::Tensor() {
}

Tensor::Tensor(const std::vector<int>& shape,
               AssignableArray weights_initialization,
               DType dtype_,
               memory::Device preferred_device) :
        w(shape,dtype_,preferred_device),
        dw(shape,dtype_,preferred_device) {
    w = weights_initialization;
}

Tensor::Tensor(const std::vector<int>& shape,
       DType dtype_,
       memory::Device preferred_device) :
   Tensor(shape,initializer::zeros(),dtype_,preferred_device) {
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

void Tensor::grad() {
    dw += 1;
}

void Tensor::clear_grad() {
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





// void Tensor::npy_save (string fname, string mode) {
//     auto dimensions = dims();
//     cnpy::npy_save(
//         fname,
//         w().data(),
//         dimensions.data(),
//         dimensions.size(),
//         mode
//     );
// }


// void Tensor::npy_save (FILE * fp) {
//     std::vector<char> header = cnpy::create_npy_header(w().data(),dims().data(),dims().size());
//     fwrite(&header[0],sizeof(char),header.size(),fp);
//     fwrite(w().data(),sizeof(R), number_of_elements(), fp);
// }
//
// void Tensor::npy_load(cnpy::NpyArray& arr) {
//     int n = arr.shape[0];
//     int d = arr.shape.size() > 1 ? arr.shape[1] : 1;
//
//     g = make_shared<storage_t>(mshadow::Shape2(n,d));
//     g->clear();
//
//     m = make_shared<storage_t>(mshadow::Shape2(n,d));
//     auto mut_data = w().mutable_cpu_data();
//     R* data_ptr = mut_data.dptr_;
//
//     if (arr.word_size == sizeof(double)) {
//         double* loaded_data = reinterpret_cast<double*>(arr.data);
//         if (arr.fortran_order) {
//             for (int i = 0; i < dims(0); i++) {
//                 for (int j = 0; j < dims(1); j++) {
//                     mut_data[i][j] = loaded_data[j * dims(0) + i];
//                 }
//             }
//         } else {
//             for (int i = 0; i < dims(0); i++) {
//                 for (int j = 0; j < dims(1); j++) {
//                     mut_data[i][j] = loaded_data[i * dims(1) + j];
//                 }
//             }
//         }
//     } else if (arr.word_size == sizeof(float)) {
//         float* loaded_data = reinterpret_cast<float*>(arr.data);
//         if (arr.fortran_order) {
//             for (int i = 0; i < dims(0); i++) {
//                 for (int j = 0; j < dims(1); j++) {
//                     mut_data[i][j] = loaded_data[j * dims(0) + i];
//                 }
//             }
//         } else {
//             for (int i = 0; i < dims(0); i++) {
//                 for (int j = 0; j < dims(1); j++) {
//                     mut_data[i][j] = loaded_data[i * dims(1) + j];
//                 }
//             }
//         }
//     } else {
//         ASSERT2(arr.word_size == sizeof(double) || arr.word_size == sizeof(float),
//             "Could not load numpy matrix : not recognized as float or double.");
//     }
// }
//
// void Tensor::npy_load(FILE * fp) {
//     auto arr = cnpy::load_the_npy_file(fp);
//     npy_load(arr);
//     arr.destruct();
// }
//
// void Tensor::npy_load(string fname) {
//     auto arr = cnpy::npy_load(fname);
//     npy_load(arr);
//     arr.destruct();
// }





Tensor Tensor::empty(const std::vector<int>& shape,
                     DType dtype_,
                     memory::Device preferred_device) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Tensor(shape, initializer::empty(), dtype_, preferred_device);
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
// bool Tensor::is_nan() const {
//     return TensorOps::is_nan(*this);
// }
//
// bool Tensor::is_grad_nan() const {
//     return TensorOps::is_grad_nan(*this);
// }
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
// // syntactic sugar
// Tensor Tensor::dot(Tensor other) const {
//     return TensorOps::mul(*this, other);
// }
//
// #define MAT_UNARY_OP( opname ) \
//  \
//     Tensor Tensor::opname() const {\
//         return TensorOps::opname(*this);\
//     }\
//
// MAT_UNARY_OP( square )
// MAT_UNARY_OP( L2_norm )
// MAT_UNARY_OP( sqrt )
// MAT_UNARY_OP( elt_inv )
// MAT_UNARY_OP( tanh )
// MAT_UNARY_OP( softplus )
// MAT_UNARY_OP( sigmoid )
// MAT_UNARY_OP( sum )
// MAT_UNARY_OP( mean )
// MAT_UNARY_OP( max )
// MAT_UNARY_OP( min )
// MAT_UNARY_OP( log )
// MAT_UNARY_OP( exp )
// MAT_UNARY_OP( abs )
// MAT_UNARY_OP( relu )
//
// Tensor Tensor::T() const {
//     return TensorOps::transpose(*this);
// }
//
// Tensor Tensor::slice(int rowstart, int rowwend) const {
//     return TensorOps::slice(*this, rowstart, rowwend);
// }
//
// Tensor Tensor::reshape(int rows, int cols) const {
//     return TensorOps::reshape(*this, rows, cols);
// }
//
// Tensor Tensor::ravel() const {
//     return TensorOps::reshape(*this, number_of_elements(), 1);
// }
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

Tensor Tensor::empty_like(const Tensor& other) {
    return Tensor(other.shape(), initializer::empty(), other.dtype(), other.preferred_device());
}

Tensor Tensor::zeros(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    return Tensor(shape, initializer::zeros(), dtype, preferred_device);
}

Tensor Tensor::empty(const std::vector<int>& shape,
                     const DType& dtype,
                     const memory::Device& preferred_device) {
    return Tensor(shape, initializer::empty(), dtype, preferred_device);
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
// int Tensor::argmax() const {
//     return TensorOps::argmax(*this);
// }
//
// int Tensor::argmin() const {
//     return TensorOps::argmin(*this);
// }
//
// vector<int> Tensor::argmin(int dimension) const {
//     return TensorOps::argmin(*this, dimension);
// }
//
// vector<int> Tensor::argmax(int dimension) const {
//     return TensorOps::argmax(*this, dimension);
// }
//
// template <typename R>
// vector<int> Tensor::argsort() const {
//     return TensorOps::argsort(*this);
// }
//
// int Tensor::argmax_slice(int lower, int upper) const {
//     return TensorOps::argmax_slice(*this, lower, upper);
// }
//
// int Tensor::argmin_slice(int lower, int upper) const {
//     return TensorOps::argmin_slice(*this, lower, upper);
// }
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
//
//
//     vector<size_t> argsort(const vector<Tensor<float>> &v) {
//         return TensorOps<float>::argsort(v);
//     }
//
//
//     vector<size_t> argsort(const vector<Tensor<double>> &v) {
//         return TensorOps<double>::argsort(v);
//     }
// }

std::ostream &operator <<(std::ostream &os, const Tensor& tensor) {
    os << "Tensor(" << tensor.shape() << ", dtype=" << dtype_to_name(tensor.dtype()) << ")";
    return os;
}
