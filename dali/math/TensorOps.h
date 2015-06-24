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
        struct log_or_zero {
            MSHADOW_XINLINE static R Map(const R& a) {
                #ifdef DALI_USE_CUDA
                    return a > 0 ? logf(a)      : 0;
                #else
                    return a > 0 ? std::log(a) : 0;
                #endif
            }
        };

    }
};

#endif
