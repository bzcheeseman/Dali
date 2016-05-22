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
#include "dali/array/function/lazy_evaluator.h"
#include "dali/array/slice.h"
#include "dali/utils.h"

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
    private:
        Tensor(const Array& w, const Array& dw, bool constant);
    public:
        typedef Array storage_t;

        storage_t w;
        mutable storage_t dw;
        std::shared_ptr<std::string> name = nullptr;
        bool constant = false;


        Tensor();

        Tensor(const std::vector<int>& shape,
               AssignableArray weights_initialization,
               DType dtype_=DTYPE_FLOAT,
               memory::Device preferred_device=memory::default_preferred_device);

        Tensor(const std::vector<int>& shape,
               DType dtype_=DTYPE_FLOAT,
               memory::Device preferred_device=memory::default_preferred_device);

        Tensor(const Array& other, bool copy=false);

        template<typename ExprT>
        Tensor(const LazyExp<ExprT>& expr) :
                Tensor(lazy::eval(expr.self())) {
        }

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

        void copy_from(const Tensor& source);
        void copy_grad_from(const Tensor& source);

        void print(std::basic_ostream<char>& stream = std::cout) const;

        /**
        Adds 1 to the gradient (`dw`) of every element in this Tensor.

        Equivalent to adding the sum of entries in this Tensor to the
        objective function.
        **/
        void grad() const;

        void clear_grad() const;
        void clear();

        const std::vector<int>& shape() const;
        DType dtype() const;
        memory::Device preferred_device() const;

        int ndim() const;
        int number_of_elements() const;

        bool is_stateless() const;
        bool is_scalar() const;
        bool is_vector() const;
        bool is_matrix() const;
        Tensor vectorlike_to_vector() const;

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

        /* A copy constructor that perform shallow copies of a Tensor.
        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads. */
        Tensor shallow_copy();

        // void resize(dim_t rows, dim_t cols);

        // Various operations on matrix.
        // Soon to be replaced by legitimate operators
        // See TensorOps for documentation.

        bool is_nan() const;
        bool is_grad_nan() const;
        // Tensor eltmul_broadcast_colwise(Tensor) const;
        // Tensor eltmul(Tensor) const;
        // Tensor eltmul(R) const;
        // Tensor eltmul_broadcast_rowwise(Tensor) const;
        // Tensor eltmul_rowwise(Tensor) const;
        // Tensor add_broadcast_rowwise(Tensor) const;
        // Tensor add_broadcast_colwise(Tensor) const;
        // Tensor add(Tensor) const;
        // Tensor sub(Tensor) const;
        // Tensor sub_broadcast(Tensor) const;
        // Tensor sub_broadcast_reversed(Tensor) const;
        // Tensor square() const;
        // Tensor L2_norm() const;
        Tensor sum() const;
        Tensor sum(const int& axis) const;
        Tensor mean() const;
        Tensor mean(const int& axis) const;
        Tensor max() const;
        Tensor max(const int& axis) const;
        Tensor min() const;
        Tensor min(const int& axis) const;
        Tensor L2_norm() const;
        Tensor L2_norm(const int& axis) const;

        // Tensor log() const;
        // Tensor exp() const;
        // Tensor abs() const;
        // Tensor sigmoid() const;
        // Tensor steep_sigmoid(R aggressiveness = 3.75) const;
        // // Warning: transpose makes a copy, uses extra memory
        // Tensor T() const;
        // Tensor tanh() const;
        // Tensor softplus() const;
        // Tensor relu() const;
        Tensor dot(const Tensor&) const;
        // template<typename ScalarType>
        // Tensor pow(ScalarType) const;
        // Tensor sqrt() const;
        // Tensor elt_inv() const;
        // Tensor slice(int rowstart, int rowend) const;

        Tensor operator[](int idx) const;
        SlicingInProgress<Tensor> operator[](const Slice& s) const;
        SlicingInProgress<Tensor> operator[](const Broadcast& b) const;

        Tensor reshape(const std::vector<int>&) const;
        Tensor copyless_reshape(const std::vector<int>&) const;

        Tensor pluck_axis(int axis, const Slice& slice) const;
        Tensor pluck_axis(int axis, int idx) const;
        Tensor squeeze(int axis) const;
        Tensor expand_dims(int new_axis) const;
        Tensor broadcast_axis(int axis) const;
        Tensor insert_broadcast_axis(int new_axis) const;

        Tensor broadcast_scalar_to_ndim(int ndim) const;
        Tensor dimshuffle(const std::vector<int>& axes) const;
        Tensor transpose(const std::vector<int>& axes) const;
        Tensor transpose() const;
        Tensor ravel() const;
        Tensor copyless_ravel() const;
        // Returns the indices of the maximum values along an axis.
        Tensor argmax() const;
        Tensor argmax(const int& axis) const;
        // Returns the indices of the minimum values along an axis.
        Tensor argmin() const;
        Tensor argmin(const int& axis) const;
        // std::vector<int> argsort() const;
        // Tensor operator-() const;
        //
        // Tensor operator+(Tensor) const;
        // Tensor operator+(R) const;
        // Tensor& operator+=(Tensor);
        // Tensor& operator+=(R);
        //
        // Tensor operator-(Tensor) const;
        // Tensor operator-(R) const;
        // Tensor& operator-=(Tensor);
        // Tensor& operator-=(R);
        //
        // Tensor operator*(Tensor other) const;
        // Tensor operator*(R alpha) const;
        // Tensor& operator*=(Tensor);
        // Tensor& operator*=(R);
        //
        // Tensor operator/(Tensor other) const;
        // Tensor operator/(R alpha) const;
        // Tensor& operator/=(Tensor);
        // Tensor& operator/=(R);
        //
        // template<typename ScalarType>
        // Tensor operator^(ScalarType) const;
        //
        // Tensor operator^(Tensor) const;
        //
        //
        // // Plucking rows and columns:
        // Tensor col(int col);
        // Tensor operator[](Indexing::Index) const;
        // Tensor operator()(Indexing::Index) const;
        // Tensor operator()(Indexing::Index, Indexing::Index) const;
        static Tensor zeros_like(const Tensor& other);
        static Tensor empty_like(const Tensor& other);
        static Tensor zeros(const std::vector<int>& shape,
                            const DType& dtype=DTYPE_FLOAT,
                            const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor empty(const std::vector<int>& shape,
                            const DType& dtype=DTYPE_FLOAT,
                            const memory::Device& preferred_device=memory::default_preferred_device);


        static Tensor gaussian(const double& mean,
                               const double& std,
                               const std::vector<int>& shape,
                               const DType& dtype=DTYPE_FLOAT,
                               const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor uniform(const double& lower,
                              const double& upper,
                              const std::vector<int>& shape,
                              const DType& dtype=DTYPE_FLOAT,
                              const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor bernoulli(const double& prob,
                                const std::vector<int>& shape,
                                const DType& dtype=DTYPE_FLOAT,
                                const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor bernoulli_normalized(const double& prob,
                                           const std::vector<int>& shape,
                                           const DType& dtype=DTYPE_FLOAT,
                                           const memory::Device& preferred_device=memory::default_preferred_device);
        template<typename T>
        static Tensor fill(const T& scalar,
                           const std::vector<int>& shape,
                           const DType& dtype=DTYPE_FLOAT,
                           const memory::Device& preferred_device=memory::default_preferred_device);



        static Tensor from_w_and_dw(const Array& w, const Array& dw, bool constant);

        // forcing memory location
        void to_device(memory::Device) const;
        void to_cpu() const;

        #ifdef DALI_USE_CUDA
            void to_gpu(int number) const;
        #endif
};

std::ostream &operator <<(std::ostream &os, const Tensor& tensor);
//
// template<typename R>
// Tensor operator+(int other, Tensor mat);
// template<typename R>
// Tensor operator+(float other, Tensor mat);
// template<typename R>
// Tensor operator+(double other, Tensor mat);
//
// template<typename R>
// Tensor operator-(int other, Tensor mat);
// template<typename R>
// Tensor operator-(float other, Tensor mat);
// template<typename R>
// Tensor operator-(double other, Tensor mat);
//
// template<typename R>
// Tensor operator*(int other, Tensor mat);
// template<typename R>
// Tensor operator*(float other, Tensor mat);
// template<typename R>
// Tensor operator*(double other, Tensor mat);
//
// template<typename R>
// std::ostream& operator<<(std::ostream&, const Tensor&);
//
// // define hash code for matrices:
// namespace std {
//     template <typename R> struct hash<Tensor> {
//         std::size_t operator()(const Tensor&) const;
//     };
// }
//
// namespace utils {
//
//     template<typename R>
//     void save_matrices(std::vector<Tensor>, std::string);
//
//     template<typename R>
//     void load_matrices(std::vector<Tensor>, std::string);
//
// }
//
// template <typename R>
// bool operator!=(const Tensor&, const Tensor&);
//
// template <typename R>
// bool operator==(const Tensor&, const Tensor&);

#endif
