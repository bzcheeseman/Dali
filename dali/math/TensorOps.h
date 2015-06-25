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
            R C; // number of columns

            __host__ __device__
            linear_index_to_row_index(R C) : C(C) {}

            __host__ __device__
            R operator()(R i) {
                return i / C;
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
                return thrust::get<extraction>(x) / divisor;
            }
        };

        THRUST_COMP_KERNEL(<, argmin_op);
        THRUST_COMP_KERNEL(>, argmax_op);

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
                int nRows    = A.shape_[0]; \
                int nColumns = A.shape_[1]; \
                /* allocate storage for row argmins and indices */ \
                thrust::device_vector<THRUST_COMP_KERNEL_RESULT_T> row_arguments(nRows); \
                thrust::device_vector<int> row_indices(nRows); \
                /* compute row arguments by finding argmin values with equal row indices */ \
                thrust::reduce_by_key \
                  (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)), \
                   thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns), \
                   thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), to_thrust(A))), \
                   row_indices.begin(), \
                   row_arguments.begin(), \
                   thrust::equal_to<int>(),\
                   kernel_name<R>()); \
                std::vector<int> host_arguments(nRows); \
                thrust::device_vector<int> row_indices2(nRows); \
                \
                thrust::transform(                  \
                    row_arguments.begin(),          \
                    row_arguments.end(),            \
                    row_indices2.begin(),            \
                    thrust_extract_arg_divide<0, int, R, int>(nColumns) \
                );\
                thrust::copy(row_indices2.begin(), row_indices2.end(), host_arguments.begin());\
                return host_arguments; \
            }

        THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmin_op, argmin )
        THRUST_KERNEL_ROWWISE_FROM_2D_MSHADOW( argmax_op, argmax )

        #define THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( kernel_name, fname ) \
            template <typename R> \
            std::vector<int> fname (const mshadow::Tensor<gpu, 1, R>& A, int reduce_dim) { \
                assert(false); \
                std::vector<int> host_arguments; \
                return host_arguments; \
            }

        THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( argmin_op, argmin )
        THRUST_KERNEL_ROWWISE_FROM_1D_MSHADOW( argmax_op, argmax )

        #endif

        // declare this operation exists for every dimension (then specialize)
        template <typename R, int dimension>
        std::vector<int> argmin(const mshadow::Tensor<cpu, dimension, R>& A);


        template <typename R>
        std::vector<int> argmin (const mshadow::Tensor<cpu, 2, R>& A, int reduce_dim) {

        }

        template <typename R>
        std::vector<int> argmin (const mshadow::Tensor<cpu, 1, R>& A, int reduce_dim) {

        }

        // declare this operation exists for every dimension (then specialize)
        template <typename R, int dimension>
        std::vector<int> argmax(const mshadow::Tensor<cpu, dimension, R>& A);


        template <typename R>
        std::vector<int> argmax (const mshadow::Tensor<cpu, 2, R>& A, int reduce_dim) {

        }

        template <typename R>
        std::vector<int> argmax (const mshadow::Tensor<cpu, 1, R>& A, int reduce_dim) {

        }


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
