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
#include "dali/array/slice.h"

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

        Tensor(const std::initializer_list<int>& shape,
               DType dtype_=DTYPE_FLOAT,
               memory::Device preferred_device=memory::default_preferred_device);

        Tensor(const std::vector<int>& shape,
               DType dtype_=DTYPE_FLOAT,
               memory::Device preferred_device=memory::default_preferred_device);

        Tensor(const Array& other, bool copy=false);
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
        const std::vector<int>& strides() const;
        DType dtype() const;
        Tensor astype(const DType& dtype) const;
        memory::Device preferred_device() const;

        int ndim() const;
        int number_of_elements() const;

        bool is_stateless() const;
        bool is_scalar() const;
        bool is_vector() const;
        bool is_matrix() const;

        void set_name(std::string& newname);
        void set_name(char* newname);
        void set_name(const char* newname);

        /* A copy constructor that perform shallow copies of a Tensor.
        Key usage is for Hogwild style training of parameters
        where different computation threads share memory for
        the parameters but each compute their own gradients.
        The gradients are kept in separate `dw` memory buffers
        but `w` buffers are shared amongst threads. */
        Tensor shallow_copy();
        
        bool is_nan() const;
        bool is_grad_nan() const;

        // Reducers
        Tensor sum() const;
        Tensor sum(const std::vector<int>& axes, bool keepdims=false) const;
        Tensor mean() const;
        Tensor mean(const std::vector<int>& axes, bool keepdims=false) const;
        Tensor max() const;
        Tensor max(const std::vector<int>& axes, bool keepdims=false) const;
        Tensor min() const;
        Tensor min(const std::vector<int>& axes, bool keepdims=false) const;
        Tensor L2_norm() const;
        Tensor L2_norm(const std::vector<int>& axes, bool keepdims=false) const;
        // Returns the indices of the maximum values along an axis.
        Tensor argmax() const;
        Tensor argmax(const int& axis) const;
        // Returns the indices of the minimum values along an axis.
        Tensor argmin() const;
        Tensor argmin(const int& axis) const;
        // Returns the indices of a sort performed on each axis.
        Tensor argsort() const;
        Tensor argsort(const int& axis) const;

        // Unary methods
        Tensor log() const;
        Tensor exp() const;
        Tensor abs() const;
        Tensor tanh() const;
        Tensor softplus() const;
        Tensor relu() const;
        Tensor dot(const Tensor&) const;
        Tensor operator[](int idx) const;
        Tensor operator[](const Tensor& indices) const;
        Tensor operator[](const std::vector<int>& indices) const;
        Tensor operator[](const std::initializer_list<int>& indices) const;
        SlicingInProgress<Tensor> operator[](const Slice& s) const;
        SlicingInProgress<Tensor> operator[](const Broadcast& b) const;
        Tensor sqrt() const;
        Tensor rsqrt() const;
        Tensor square() const;
        Tensor cube() const;
        Tensor eltinv() const;
        Tensor sigmoid() const;

        // Shape transformations
        Tensor reshape(const std::vector<int>&) const;
        Tensor right_fit_ndim(const int& dimensionality) const;
        Tensor pluck_axis(int axis, const Slice& slice) const;
        Tensor pluck_axis(int axis, int idx) const;
        Tensor squeeze(int axis) const;
        Tensor expand_dims(int new_axis) const;
        Tensor broadcast_axis(int axis) const;
        Tensor insert_broadcast_axis(int new_axis) const;

        Tensor broadcast_scalar_to_ndim(int ndim) const;
        Tensor dimshuffle(const std::vector<int>& axes) const;
        Tensor swapaxes(const int& axis1, const int& axis2) const;
        Tensor transpose(const std::vector<int>& axes) const;
        Tensor transpose() const;
        Tensor ravel() const;
        Tensor copyless_ravel() const;
        // Tensor operator-() const;
        static Tensor zeros_like(const Tensor& other);
        static Tensor ones_like(const Tensor& other);
        static Tensor empty_like(const Tensor& other);
        static Tensor fill_like(const double& prob, const Tensor& other);

        static Tensor zeros(const std::vector<int>& shape,
                            const DType& dtype=DTYPE_FLOAT,
                            const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor ones(const std::vector<int>& shape,
                           const DType& dtype=DTYPE_FLOAT,
                           const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor empty(const std::vector<int>& shape,
                            const DType& dtype=DTYPE_FLOAT,
                            const memory::Device& preferred_device=memory::default_preferred_device);
        static Tensor normal(const Array& loc, const Array& scale, const std::vector<int>& shape);
        static Tensor uniform(const Array& low, const Array& high, const std::vector<int>& shape);
        static Tensor bernoulli(const Array& prob, const std::vector<int>& shape);
        static Tensor bernoulli_normalized(const Array& prob, const std::vector<int>& shape);
        static Tensor fill(const double& scalar,
                           const std::vector<int>& shape,
                           const DType& dtype=DTYPE_FLOAT,
                           const memory::Device& preferred_device=memory::default_preferred_device);

        template<typename T>
        Tensor& operator=(const std::vector<T>& values) = delete;
        template<typename T>
        Tensor& operator=(const std::initializer_list<T>& values) = delete;

        static Tensor load(const std::string& fname);
        static Tensor load(FILE * fp);
        static void save(const std::string& fname, const Tensor& arr, const std::ios_base::openmode& mode=std::ios_base::out);
        static void save(std::basic_ostream<char>& stream, const Tensor& arr);

        static Tensor from_w_and_dw(const Array& w, const Array& dw, bool constant);

        // forcing memory location
        void to_device(memory::Device) const;
        void to_cpu() const;
        void to_gpu(int number) const;
};

std::ostream &operator <<(std::ostream &os, const Tensor& tensor);

#define DALI_DECLARE_TENSOR_INTERACTION_INPLACE(SYMBOL)\
    Tensor& operator SYMBOL (Tensor&  left, const Tensor& right);\
    void operator SYMBOL (Tensor&& left, const Tensor& right);\

DALI_DECLARE_TENSOR_INTERACTION_INPLACE(+= );
DALI_DECLARE_TENSOR_INTERACTION_INPLACE(-= );
DALI_DECLARE_TENSOR_INTERACTION_INPLACE(*= );
DALI_DECLARE_TENSOR_INTERACTION_INPLACE(/= );
DALI_DECLARE_TENSOR_INTERACTION_INPLACE(^= );

#define DALI_DECLARE_TENSOR_INTERACTION(SYMBOL)\
    Tensor operator SYMBOL (const Tensor& left, const Tensor& right);\
    Tensor operator SYMBOL (const Tensor& left, double right);\
    Tensor operator SYMBOL (const Tensor& left, float right);\
    Tensor operator SYMBOL (const Tensor& left, int right);\
    Tensor operator SYMBOL (double left, const Tensor& right);\
    Tensor operator SYMBOL (float left, const Tensor& right);\
    Tensor operator SYMBOL (int left, const Tensor& right)

DALI_DECLARE_TENSOR_INTERACTION(+);
DALI_DECLARE_TENSOR_INTERACTION(-);
DALI_DECLARE_TENSOR_INTERACTION(*);
DALI_DECLARE_TENSOR_INTERACTION(/);
DALI_DECLARE_TENSOR_INTERACTION(^);

#endif
