#ifndef DALI_MAT_MATH_TENSOROPS_H
#define DALI_MAT_MATH_TENSOROPS_H

#include <mshadow/tensor.h>
#include <random>
#include <functional>
#include <math.h>


#ifdef DALI_USE_CUDA

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
// contains thrust::max_element & thrust::min_element
#include <thrust/extrema.h>

#define STR(x) __THIS_IS_VERY_ABNOXIOUS(x)
#define __THIS_IS_VERY_ABNOXIOUS(tok) #tok

/* CUDA UTILS START HERE */
namespace TensorOps {
    template<typename R, int ndims>
    thrust::device_ptr<R> to_thrust(const mshadow::Tensor<mshadow::gpu, ndims, R>& tg) {
        auto dev_ptr = thrust::device_pointer_cast(tg.dptr_);
        return dev_ptr;
    }

    template<typename T>
    struct near_equal {
        T tol;
        near_equal(T _tol) : tol(_tol) {}
        __host__ __device__ bool operator()(const T& lhs, const T& rhs) const {
            return std::fabs(lhs - rhs) < tol;
        }
    };
}
#endif

enum OptionalTranspose {
    TRANSPOSE,
    NO_TRANSPOSE
};

enum OperationType {
    OVERWRITE,
    ADD_TO_EXISTING
};

/* CUDA UTILS END HERE */

#define DALI_ASSIGN(op, out, expr) if ((op) == OVERWRITE) { out = (expr); } else {  out += (expr);  }

namespace TensorOps {
    using mshadow::gpu;
    using mshadow::cpu;

    namespace comparison {
        #ifdef DALI_USE_CUDA
        template<int ndims, typename R>
        bool equals(const mshadow::Tensor<gpu, ndims, R>& a, const mshadow::Tensor<gpu, ndims, R>& b, int num_elts) {
            return thrust::equal(to_thrust(a), to_thrust(a) + num_elts, to_thrust(b));
        }
        #endif
        template<int ndims, typename R>
        bool equals(const mshadow::Tensor<cpu, ndims, R>& a, const mshadow::Tensor<cpu, ndims, R>& b, int num_elts) {
            return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_);
        }


        #ifdef DALI_USE_CUDA
        template<int ndims, typename R>
        bool allclose(const mshadow::Tensor<gpu, ndims, R>& a, const mshadow::Tensor<gpu, ndims, R>& b, int num_elts, R tol) {
            return thrust::equal(to_thrust(a),
                                 to_thrust(a) + num_elts,
                                 to_thrust(b),
                                 near_equal<R>(tol));
        }
        #endif
        template<int ndims, typename R>
        bool allclose(const mshadow::Tensor<cpu, ndims, R>& a, const mshadow::Tensor<cpu, ndims, R>& b, int num_elts, R tol) {
            return std::equal(a.dptr_, a.dptr_ + num_elts, b.dptr_,
            [tol](const R & lhs, const R & rhs) {
                return std::fabs(lhs - rhs) < tol;
            });
        }
    }

    #ifdef DALI_USE_CUDA
    template<int ndims, typename R>
    R sum(const mshadow::Tensor<gpu, ndims, R>& a, int num_elts) {
        return thrust::reduce(to_thrust(a), to_thrust(a) + num_elts, 0.0, thrust::plus<R>());
    }
    #endif

    template<int ndims, typename R>
    R sum(const mshadow::Tensor<cpu, ndims, R>& a, int num_elts) {
        return std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0);
    }


    #ifdef DALI_USE_CUDA
    template <typename T>
    struct thrust_square {
        __host__ __device__
        T operator()(const T& x) const {
            return x * x;
        }
    };

    template<int ndims, typename R>
    R L2_norm(const mshadow::Tensor<gpu, ndims, R>& a, int num_elts) {
        return std::sqrt(thrust::transform_reduce(to_thrust(a), to_thrust(a) + num_elts, thrust_square<R>(), 0.0, thrust::plus<R>()));
    }
    #endif


    namespace arg {
        #ifdef DALI_USE_CUDA
        // Convert a linear index to a row index
        template <typename R>
        struct linear_index_to_row_index : public thrust::unary_function<R,R> {
            R cols; // number of columns
            __host__ __device__
            linear_index_to_row_index(R _cols) : cols(_cols) {}
            __host__ __device__
            R operator()(R i) {
                return i / cols;
            }
        };

        // convert a linear index to a row index
        template <typename T>
        struct linear_index_to_col_index: public thrust::unary_function<T, T> {
            T rows; // number of rows
            __host__ __device__ linear_index_to_col_index(T rows) : rows(rows) {}

            __host__ __device__ T operator() (T i) {
                return i % rows;
            }
        };

        #define THRUST_COMP_KERNEL_RESULT_T thrust::tuple<int, R>

        #define THRUST_COMP_KERNEL(compsymbol, kernel_name) \
            template<typename R> \
            struct kernel_name : public thrust::binary_function<THRUST_COMP_KERNEL_RESULT_T ,THRUST_COMP_KERNEL_RESULT_T , THRUST_COMP_KERNEL_RESULT_T > {\
                __host__ __device__ \
                THRUST_COMP_KERNEL_RESULT_T operator()(const THRUST_COMP_KERNEL_RESULT_T & a, const THRUST_COMP_KERNEL_RESULT_T & b) const {\
                    if (thrust::get<1>(a) compsymbol thrust::get<1>(b)){\
                        return a;\
                    } else {\
                        return b;\
                    }\
                }\
            }\

        template <int extraction, typename TA, typename TB>
        struct thrust_extract_arg : public thrust::unary_function<TA,TB> {
            __host__ __device__
            auto operator()(const thrust::tuple<TA, TB>& x) -> decltype(thrust::get<extraction>(x)) const {
                return thrust::get<extraction>(x);
            }
        };

        template <int extraction, typename TA, typename TB, typename T>
        struct thrust_extract_arg_divide : public thrust::unary_function<TA,TB> {
            T divisor;
            thrust_extract_arg_divide(T _divisor) : divisor(_divisor) {}

            __host__ __device__
            auto operator()(const thrust::tuple<TA, TB>& x) -> typename thrust::tuple_element<extraction,thrust::tuple<TA, TB>>::type const {
                return thrust::get<extraction>(x) % divisor;
            }
        };

        THRUST_COMP_KERNEL(<=, argmin_op);
        THRUST_COMP_KERNEL(>=, argmax_op);

        // declare this operation exists for every dimension (then specialize)
        template <typename R, int dimension>
        std::vector<int> argmin(const mshadow::Tensor<gpu, dimension, R>& A);
        // declare this operation exists for every dimension (then specialize)
        template <typename R, int dimension>
        std::vector<int> argmax(const mshadow::Tensor<gpu, dimension, R>& A);

        // specialize for kernel
        #define THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( kernel_name, fname ) \
            template <typename R> \
            std::vector<int> fname (const mshadow::Tensor<gpu, 2, R>& A, int reduce_dim) { \
                int num_rows = A.shape_[0]; \
                int num_cols = A.shape_[1]; \
                if (reduce_dim == 0) {\
                    /* reduce_by_key for thrust only works for contiguous portions, since */\
                    /* row-major memory breaks this assumption, we cannot do this without */\
                    /* many intermediate steps; we instead transpose & switch reduce dim  */\
                    mshadow::Tensor<gpu, 2, R> A_T( mshadow::Shape2(num_cols, num_rows));\
                    mshadow::AllocSpace(&A_T, false);\
                    A_T = A.T();\
                    auto sorted = fname ( A_T , 1 );\
                    mshadow::FreeSpace(&A_T);\
                    return sorted;\
                } else if (reduce_dim == 1) {\
                    /* allocate storage for row arguments and indices */ \
                    thrust::device_vector<THRUST_COMP_KERNEL_RESULT_T> row_arguments(num_rows); \
                    thrust::device_vector<int> row_indices(num_rows); \
                    /* compute row arguments by finding argmin values with equal row indices */ \
                    thrust::reduce_by_key(\
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(num_cols)), \
                        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(num_cols)) + (num_rows * num_cols), \
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), to_thrust(A))), \
                        row_indices.begin(),                                \
                        row_arguments.begin(),                              \
                        thrust::equal_to<int>(),                            \
                        kernel_name<R>()                                    \
                    );                                                      \
                    std::vector<int> host_arguments(num_rows);              \
                    thrust::device_vector<int> row_indices2(num_rows);      \
                    thrust::transform(                                      \
                        row_arguments.begin(),                              \
                        row_arguments.end(),                                \
                        row_indices2.begin(),                               \
                        thrust_extract_arg_divide<0, int, R, int>(num_cols) \
                    );\
                    thrust::copy(row_indices2.begin(), row_indices2.end(), host_arguments.begin());\
                    return host_arguments; \
                } else {\
                    throw std::runtime_error(STR(fname) " can only be used with axis equal to 0 or 1.");\
                }\
            }

        THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmin_op, argmin )
        THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmax_op, argmax )

        // following the advice from this Stackoverflow:
        // http://stackoverflow.com/questions/7709181/finding-the-maximum-element-value-and-its-position-using-cuda-thrust
        #define THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( thrust_op_name, fname ) \
            template <typename R> \
            std::vector<int> fname (const mshadow::Tensor<gpu, 1, R>& A, int reduce_dim) { \
                auto start = to_thrust(A);\
                auto idx = thrust_op_name (start, start + A.shape_[0]);\
                return { (int) (&idx[0] - &start[0]) };\
            }

        THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( thrust::min_element , argmin )
        THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( thrust::max_element , argmax )

        #endif

        // declare this operation exists for every dimension (then specialize)
        template <typename R, int dimension>
        std::vector<int> argmin(const mshadow::Tensor<cpu, dimension, R>& A);

        // declare this operation exists for every dimension (then specialize)
        template <typename R, int dimension>
        std::vector<int> argmax(const mshadow::Tensor<cpu, dimension, R>& A);

        #define CPU_KERNEL_ROWWISE_FROM_2D_MSHADOW( fname , opsymbol ) \
            template <typename R> \
            std::vector<int> fname (const mshadow::Tensor<cpu, 2, R>& A, int reduce_dim) {\
                const int num_rows = A.shape_[0];\
                const int num_cols = A.shape_[1];\
                if (reduce_dim == 1) {\
                    std::vector<int> arguments(num_rows);\
                    for (int i = 0; i < num_rows; i++) {\
                        arguments[i] = 0;\
                        for (int j = 0; j < num_cols; j++) {\
                            if (  *(A.dptr_ + (A.stride_ * i) + j) opsymbol *(A.dptr_ + (A.stride_ * i) + arguments[i]) ) {\
                                arguments[i] = j;\
                            }\
                        }\
                    }\
                    return arguments;\
                } else if (reduce_dim == 0) {\
                    std::vector<int> arguments(num_cols);\
                    for (int j = 0; j < num_cols; j++) {\
                        arguments[j] = 0;\
                        for (int i = 0; i < num_rows; i++) {\
                            if ( *(A.dptr_ + (A.stride_ * i) + j) opsymbol *(A.dptr_ + (A.stride_ * arguments[j]) + j) ) {\
                                arguments[j] = i;\
                            }\
                        }\
                    }\
                    return arguments;\
                } else {\
                    throw std::runtime_error("argmax can only be used with axis equal to 0 or 1.");\
                }\
            }

        #define CPU_KERNEL_ROWWISE_FROM_1D_MSHADOW( fname, opsymbol ) \
            template <typename R>\
            std::vector<int> fname (const mshadow::Tensor<cpu, 1, R>& A, int reduce_dim) { \
                std::vector<int> offset_row(1, 0);\
                int& offset = offset_row[0];\
                const int num_rows = A.shape_[0];\
                for (int i = 0; i < A.shape_[0]; i++) {\
                    if (*(A.dptr_ + i) opsymbol *(A.dptr_ + offset)) {\
                        offset = i;\
                    }\
                }\
                return offset_row;\
            }

        CPU_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmax, > )
        CPU_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmin, < )

        CPU_KERNEL_ROWWISE_FROM_1D_MSHADOW( argmax, > )
        CPU_KERNEL_ROWWISE_FROM_1D_MSHADOW( argmin, < )
    }

    template <typename T>
    struct thrust_square_reduce {
        T operator()(const T& x, const T& y) const {
            return x + (y * y);
        }
    };

    template<int ndims, typename R>
    R L2_norm(const mshadow::Tensor<cpu, ndims, R>& a, int num_elts) {
        return std::sqrt(std::accumulate(a.dptr_, a.dptr_ + num_elts, 0.0, thrust_square_reduce<R>()));
    }

    template<int ndims, typename R>
    void eye(mshadow::Tensor<cpu,ndims,R>& tc, R diag) {
        if (tc.shape_[0] != tc.shape_[1]) {
            throw std::runtime_error("Identity initialization must be called on a square matrix.");
        }
        tc = 0.0;
        for (int i = 0; i < tc.shape_[0]; i++)
            tc[i][i] = diag;
    }

    #ifdef DALI_USE_CUDA
    template<typename R>
    struct IdentityKernel : public thrust::binary_function<int,R, R> {
        R diag;
        int rows;
        IdentityKernel(R _diag, int _rows) : diag(_diag), rows(_rows) {}
        __host__ __device__ R operator()(const int& offset, const R& b) const {
            return (offset / rows) == (offset % rows) ? diag : 0.0;
        }
    };

    template<int ndims, typename R>
    void eye(mshadow::Tensor<gpu,ndims,R>& tg, R diag) {
        if (tg.shape_[0] != tg.shape_[1]) {
            throw std::runtime_error("Identity initialization must be called on a square matrix.");
        }
        int num_elts = tg.shape_[0] * tg.shape_[1];
        int num_rows = tg.shape_[0];
        // counting iterators define a sequence [0, 8)
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + num_elts;

        thrust::transform(
            first,
            last,
            to_thrust(tg),
            to_thrust(tg),
            IdentityKernel<R>(diag, num_rows)
        );
    }
    #endif

    namespace random {
        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<Device, ndims, R>& t, R lower, R upper) {
            std::random_device rd;
            mshadow::Random<Device, R> generator((int)rd());
            generator.SampleUniform(&t, lower, upper);
        }

        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<Device, ndims, R>& t, R mean, R std) {
            std::random_device rd;
            mshadow::Random<Device, R> generator((int)rd());
            generator.SampleGaussian(&t, mean, std);
        }
    };

    namespace op {
        template<typename R>
        struct square {
            MSHADOW_XINLINE static R Map(const R& a) {
                return a * a;
            }
        };

        template<typename R>
        struct sqrt_f {
            MSHADOW_XINLINE static R Map(const R& a) {
                return sqrt(a);
            }
        };

        template<typename R>
        struct inv {
            MSHADOW_XINLINE static R Map(const R& a) {
                return ((R)1.0) / a;
            }
        };

        template<typename R>
        struct sigmoid {
            MSHADOW_XINLINE static R Map(const R& a) {
                #ifdef DALI_USE_CUDA
                    return 1.0 / (1.0 + expf(-a));
                #else
                    return 1.0 / (1.0 + std::exp(-a));
                #endif
            }
        };

        template<typename R>
        struct log {
            MSHADOW_XINLINE static R Map(const R& a) {
                #ifdef DALI_USE_CUDA
                    return logf(a);
                #else
                    return std::log(a);
                #endif
            }
        };

        template<typename R>
        struct exp {
            MSHADOW_XINLINE static R Map(const R& a) {
                #ifdef DALI_USE_CUDA
                    return expf(a);
                #else
                    return std::exp(a);
                #endif
            }
        };

        template<typename R>
        struct div_grad {
            MSHADOW_XINLINE static R Map(const R& a, const R& b) {
                return a / (b * b);
            }
        };

        template<typename R>
        struct dsigmoid {
            MSHADOW_XINLINE static R Map(const R& a) {
                return a * (((R)1.0) - a);
            }
        };

        template<typename R>
        struct tanh {
            MSHADOW_XINLINE static R Map(const R& a) {
                #ifdef DALI_USE_CUDA
                    return tanhf(a);
                #else
                    return std::tanh(a);
                #endif
            }
        };

        template<typename R>
        struct dtanh {
            MSHADOW_XINLINE static R Map(const R& a) {
                return 1.0 - a * a;
            }
        };

        template<typename R>
        struct power {
            MSHADOW_XINLINE static R Map(const R& a, const R& b) {
                #ifdef DALI_USE_CUDA
                    return powf(a, b);
                #else
                    return pow(a, b);
                #endif
            }
        };

        template<typename R>
        struct abs {
            MSHADOW_XINLINE static R Map(const R& a) {
                return std::abs(a);
            }
        };

        template<typename R>
        struct log_or_zero {
            MSHADOW_XINLINE static R Map(const R& a) {
                #ifdef DALI_USE_CUDA
                    return a > 0 ? logf(a)      : 0;
                #else
                    return a > 0 ? std::log(a) : 0;
                #endif
            }
        };

        template<typename R>
        struct sign {
            MSHADOW_XINLINE static R Map(const R& x) {
                return x > 0.0 ? 1.0 : -1.0;
            }
        };

        template<typename R>
        struct max_scalar {
            MSHADOW_XINLINE static R Map(const R& x, const R& y) {
                return x > y ? x : y;
            }
        };

        template<typename R>
        struct  max_scalar_mask {
            MSHADOW_XINLINE static R Map(const R& m, const R& lower_bound) {
                return (m >= lower_bound) ? 1.0 : 0.0;
            }
        };

        template<typename R>
        struct  steep_sigmoid {
            MSHADOW_XINLINE static R Map(const R& x, const R& aggressiveness) {
                #ifdef DALI_USE_CUDA
                    return 1.0 / (1.0 + expf( - aggressiveness * x));
                #else
                    return 1.0 / (1.0 + std::exp( - aggressiveness * x));
                #endif
            }
        };

        template<typename R>
        struct  steep_sigmoid_backward {
            MSHADOW_XINLINE static R Map(const R& x, const R& aggressiveness) {
                return aggressiveness * (x - x * x);
            }
        };

    }
};

#endif
