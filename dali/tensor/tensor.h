#ifndef CORE_MAT_H
#define CORE_MAT_H

#include <atomic>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <ostream>
#include <unordered_map>

#include "dali/array/array.h"
// #include "dali/tensor/MatOps.h"
// #include "dali/tensor/Weights.h"
// #include "dali/tensor/Tape.h"
#include "dali/utils.h"

// template<typename R>
// struct weights;
/**
Tensor
---

Main tensor class for this library. The Tensor
class contains two pieces of memory, `w`
and `dw`. The first is the actual weights or
values associated with this tensor, and the
second is the local contribution to the
objective function (or ∂E/∂Tensor). This local
contribution can then be used in
backpropagation.

Tensor is a fundamental building block of
Automatic Differentiation in Dali.

**/

class Tensor {
    public:
        typedef Array storage_t;

        storage_t w;
        mutable storage_t dw;
        std::shared_ptr<std::string> name = nullptr;
        bool constant;


        Tensor();

        Tensor(const std::vector<int>& shape,
               weights::initializer_t wi,
               DType dtype_=DTYPE_FLOAT,
               memory::Device preferred_device=memory::default_preferred_device);

        Tensor(const std::vector<int>& shape,
               DType dtype_=DTYPE_FLOAT,
               memory::Device preferred_device=memory::default_preferred_device);

        /*
        A copy constructor that perform shallow and deep
        copies of a Tensor.
        By default shallow copy is performed.

        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads.
        */
        Tensor(const Tensor& other, bool copy_w=false, bool copy_d=false);

        ~Tensor();

        void copy_from(const Tensor<R>& source);
        void copy_grad_from(const Tensor<R>& source);

        void print(std::basic_ostream<char>& stream = std::cout) const;

        /**
        Adds 1 to the gradient (`dw`) of every element in this Tensor.

        Equivalent to adding the sum of entries in this Tensor to the
        objective function.
        **/

        void add_to_objective();

        void clear_grad();
        void clear();

        const std::vector<int>& shape() const;
        DType dtype() const;
        DType preferred_device() const;

        int ndim() const;
        int number_of_elements() const;

        bool is_stateless() const;

        void set_name(std::string& newname);
        void set_name(char* newname);
        void set_name(const char* newname);

        // void npy_save(std::string fname, std::string mode = "w");
        // void npy_save(FILE*);
        // IMPLEMENT IN ARRAY // void npy_load(std::string fname);
        // IMPLEMENT IN ARRAY // void npy_load(FILE*);
        //
        // static Tensor npy_load(File*);
        // static Tensor npy_load(const std::string&);

        static Tensor empty(const std::vector<int>& shape,
                         DType dtype_=DTYPE_FLOAT,
                         memory::Device preferred_device=memory::default_preferred_device);

        /* A copy constructor that perform shallow copies of a Mat.
        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads. */
        Tensor shallow_copy();

        // void resize(dim_t rows, dim_t cols);

        // Various operations on matrix.
        // Soon to be replaced by legitimate operators
        // See MatOps for documentation.

        // bool is_nan() const;
        // bool is_grad_nan() const;
        // Mat<R> eltmul_broadcast_colwise(Mat<R>) const;
        // Mat<R> eltmul(Mat<R>) const;
        // Mat<R> eltmul(R) const;
        // Mat<R> eltmul_broadcast_rowwise(Mat<R>) const;
        // Mat<R> eltmul_rowwise(Mat<R>) const;
        // Mat<R> add_broadcast_rowwise(Mat<R>) const;
        // Mat<R> add_broadcast_colwise(Mat<R>) const;
        // Mat<R> add(Mat<R>) const;
        // Mat<R> sub(Mat<R>) const;
        // Mat<R> sub_broadcast(Mat<R>) const;
        // Mat<R> sub_broadcast_reversed(Mat<R>) const;
        // Mat<R> square() const;
        // Mat<R> L2_norm() const;
        // Mat<R> sum() const;
        // Mat<R> mean() const;
        // Mat<R> max() const;
        // Mat<R> min() const;
        // Mat<R> log() const;
        // Mat<R> exp() const;
        // Mat<R> abs() const;
        // Mat<R> sigmoid() const;
        // Mat<R> steep_sigmoid(R aggressiveness = 3.75) const;
        // // Warning: transpose makes a copy, uses extra memory
        // Mat<R> T() const;
        // Mat<R> tanh() const;
        // Mat<R> softplus() const;
        // Mat<R> relu() const;
        // Mat<R> mul(Mat<R>) const;
        // Mat<R> dot(Mat<R>) const;
        // template<typename ScalarType>
        // Mat<R> pow(ScalarType) const;
        // Mat<R> sqrt() const;
        // Mat<R> elt_inv() const;
        // Mat<R> slice(int rowstart, int rowend) const;
        // Mat<R> reshape(int rows, int cols) const;
        // Mat<R> ravel() const;
        //
        // int argmax() const;
        // int argmin() const;
        //
        // std::vector<int> argmin(int dimension) const;
        // std::vector<int> argmax(int dimension) const;
        // std::vector<int> argsort() const;
        // /*
        // Restricted range argmax: returns the index of the
        // highest value between two indices, lower and upper
        // (useful if a range of predictions is inadmissible,
        // so we are only considering a subset of predictions)
        // */
        // int argmax_slice(int lower, int upper) const;
        // int argmin_slice(int lower, int upper) const;
        //
        // Mat<R> operator-() const;
        //
        // Mat<R> operator+(Mat<R>) const;
        // Mat<R> operator+(R) const;
        // Mat<R>& operator+=(Mat<R>);
        // Mat<R>& operator+=(R);
        //
        // Mat<R> operator-(Mat<R>) const;
        // Mat<R> operator-(R) const;
        // Mat<R>& operator-=(Mat<R>);
        // Mat<R>& operator-=(R);
        //
        // Mat<R> operator*(Mat<R> other) const;
        // Mat<R> operator*(R alpha) const;
        // Mat<R>& operator*=(Mat<R>);
        // Mat<R>& operator*=(R);
        //
        // Mat<R> operator/(Mat<R> other) const;
        // Mat<R> operator/(R alpha) const;
        // Mat<R>& operator/=(Mat<R>);
        // Mat<R>& operator/=(R);
        //
        // template<typename ScalarType>
        // Mat<R> operator^(ScalarType) const;
        //
        // Mat<R> operator^(Mat<R>) const;
        //
        //
        // // Plucking rows and columns:
        // Mat<R> col(int col);
        // Mat<R> operator[](int) const;
        // Mat<R> operator[](Mat<int>) const;
        // Mat<R> operator()(int) const;
        // Mat<R> operator[](Indexing::Index) const;
        // Mat<R> operator()(Indexing::Index) const;
        // Mat<R> operator()(Indexing::Index, Indexing::Index) const;
        // // Mat<R> operator()(void*, Indexing::Index) const;
        // Mat<R> operator()(void*, int) const;
        static Mat<R> zeros_like(Mat<R> shape);
        static Mat<R> empty_like(Mat<R> shape);


        // forcing memory location
        void to_device(memory::Device) const;
        void to_cpu() const;

        #ifdef DALI_USE_CUDA
            void to_gpu(int number) const;
        #endif
};
//
// template<typename R>
// Mat<R> operator+(int other, Mat<R> mat);
// template<typename R>
// Mat<R> operator+(float other, Mat<R> mat);
// template<typename R>
// Mat<R> operator+(double other, Mat<R> mat);
//
// template<typename R>
// Mat<R> operator-(int other, Mat<R> mat);
// template<typename R>
// Mat<R> operator-(float other, Mat<R> mat);
// template<typename R>
// Mat<R> operator-(double other, Mat<R> mat);
//
// template<typename R>
// Mat<R> operator*(int other, Mat<R> mat);
// template<typename R>
// Mat<R> operator*(float other, Mat<R> mat);
// template<typename R>
// Mat<R> operator*(double other, Mat<R> mat);
//
// template<typename R>
// std::ostream& operator<<(std::ostream&, const Mat<R>&);
//
// // define hash code for matrices:
// namespace std {
//     template <typename R> struct hash<Mat<R>> {
//         std::size_t operator()(const Mat<R>&) const;
//     };
// }
//
// namespace utils {
//
//     template<typename R>
//     void save_matrices(std::vector<Mat<R>>, std::string);
//
//     template<typename R>
//     void load_matrices(std::vector<Mat<R>>, std::string);
//
// }
//
// template <typename R>
// bool operator!=(const Mat<R>&, const Mat<R>&);
//
// template <typename R>
// bool operator==(const Mat<R>&, const Mat<R>&);

#endif
