#include "uniform.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/utils/print_utils.h"
#include "dali/utils/random.h"
#include "dali/array/expression/computation.h"
#include "dali/array/op/elementwise_operation.h"
#include "dali/array/op/unary.h"
#include "dali/array/op/binary.h"

#ifdef DALI_USE_CUDA
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>
// contains thrust::max_element & thrust::min_element
#include <thrust/extrema.h>
#endif

// TODO(jonathan): improve rng usage using http://docs.nvidia.com/cuda/curand/device-api-overview.html#pseudorandom-sequences
namespace op {
    struct Uniform : public Expression {
        Uniform(Array low, Array high, const std::vector<int>& shape) :
            Expression(shape, low.dtype(), {low, high}) {}
        using Expression::copy;
        virtual expression_ptr copy() const {return std::make_shared<Uniform>(arguments_[0], arguments_[1], shape_);}
        memory::Device preferred_device() const {return memory::default_preferred_device;}
        virtual bool supports_operator(OPERATOR_T operator_t) const {return operator_t == OPERATOR_T_EQL;}
        const Array& low() const {return arguments_[0];}
        const Array& high() const {return arguments_[1];}
        virtual expression_ptr _reshape(const std::vector<int>& new_shape,
                                        const Array* owner) const override {
            return std::make_shared<Uniform>(arguments_[0], arguments_[1], new_shape);
        }
    };
    struct Normal : public Expression {
        Normal(Array loc, Array scale, const std::vector<int>& shape) :
            Expression(shape, loc.dtype(), {loc, scale}) {}
        using Expression::copy;
        virtual expression_ptr copy() const {return std::make_shared<Normal>(arguments_[0], arguments_[1], shape_);}
        memory::Device preferred_device() const {return memory::default_preferred_device;}
        virtual bool supports_operator(OPERATOR_T operator_t) const {return operator_t == OPERATOR_T_EQL;}
        const Array& loc() const {return arguments_[0];}
        const Array& scale() const {return arguments_[1];}
        virtual expression_ptr _reshape(const std::vector<int>& new_shape,
                                        const Array* owner) const override {
            return std::make_shared<Uniform>(arguments_[0], arguments_[1], new_shape);
        }
    };

    namespace {
        template<typename T>
        void cpu_uniform(T* dst, const T* low, const T* high, int size) {
            auto& gen = utils::random::generator();
            std::uniform_real_distribution<T> dist(*low, *high);
            for (int i = 0; i < size; ++i) {
                *(dst + i) = dist(gen);
            }
        }

        template<>
        void cpu_uniform(int* dst, const int* low, const int* high, int size) {
            auto& gen = utils::random::generator();
            std::uniform_int_distribution<int> dist(*low, *high);
            for (int i = 0; i < size; ++i) {
                *(dst + i) = dist(gen);
            }
        }

        template<typename T>
        void cpu_normal(T* dst, const T* loc, const T* scale, int size) {
            auto& gen = utils::random::generator();
            std::normal_distribution<T> dist(*loc, *scale);
            for (int i = 0; i < size; ++i) {
                *(dst + i) = dist(gen);
            }
        }
    }
    struct CpuUniformImpl : public Computation {
        using Computation::Computation;
        virtual void run() {
            Array dst = left_;
            Uniform* uni = static_cast<Uniform*>(right_.expression().get());
            auto op_dtype = dst.dtype();
            auto device = memory::Device::cpu();
            void* dst_ptr = left_data(device);
            const void* arg0_ptr = argument_data(device, 0);
            const void* arg1_ptr = argument_data(device, 1);
            if (dst.dtype() == DTYPE_FLOAT) {
                cpu_uniform(static_cast<float*>(dst_ptr), static_cast<const float*>(arg0_ptr), static_cast<const float*>(arg1_ptr), dst.number_of_elements());
            } else if (dst.dtype() == DTYPE_DOUBLE) {
                cpu_uniform(static_cast<double*>(dst_ptr), static_cast<const double*>(arg0_ptr), static_cast<const double*>(arg1_ptr), dst.number_of_elements());
            } else if (dst.dtype() == DTYPE_INT32) {
                cpu_uniform(static_cast<int*>(dst_ptr), static_cast<const int*>(arg0_ptr), static_cast<const int*>(arg1_ptr), dst.number_of_elements());
            } else {
                ASSERT2(false, utils::make_message(
                    "Uniform is not implemented for dtype ", dtype_to_name(dst.dtype()), "."));
            }
        }
    };
    int cpu_uniform_impl = register_implementation(
        typeid(Uniform).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            if (dest.preferred_device().is_cpu()) {
                return std::make_shared<CpuUniformImpl>(dest, operator_t, x, assignment);
            } else {
                return nullptr;
            }
        }
    );
    struct CpuNormalImpl : public Computation {
        using Computation::Computation;
        virtual void run() {
            Array dst = left_;
            Normal* uni = static_cast<Normal*>(right_.expression().get());
            auto op_dtype = dst.dtype();
            auto device = memory::Device::cpu();
            void* dst_ptr = left_data(device);
            const void* arg0_ptr = argument_data(device, 0);
            const void* arg1_ptr = argument_data(device, 1);
            if (dst.dtype() == DTYPE_FLOAT) {
                cpu_normal(static_cast<float*>(dst_ptr), static_cast<const float*>(arg0_ptr), static_cast<const float*>(arg1_ptr), dst.number_of_elements());
            } else if (dst.dtype() == DTYPE_DOUBLE) {
                cpu_normal(static_cast<double*>(dst_ptr), static_cast<const double*>(arg0_ptr), static_cast<const double*>(arg1_ptr), dst.number_of_elements());
            } else {
                ASSERT2(false, utils::make_message(
                    "Normal is not implemented for dtype ", dtype_to_name(dst.dtype()), "."));
            }
        }
    };
    int cpu_normal_impl = register_implementation(
        typeid(Normal).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            if (dest.preferred_device().is_cpu()) {
                return std::make_shared<CpuNormalImpl>(dest, operator_t, x, assignment);
            } else {
                return nullptr;
            }
        }
    );

#ifdef DALI_USE_CUDA
    namespace {
        // from thrust Monte Carlo experiment
        // here: https://github.com/thrust/thrust/blob/master/examples/monte_carlo.cu
        template<typename R>
        struct hashable_operator {
            __host__ __device__
            static unsigned int hash_operator(unsigned int a) {
                a = (a+0x7ed55d16) + (a<<12);
                a = (a^0xc761c23c) ^ (a>>19);
                a = (a+0x165667b1) + (a<<5);
                a = (a+0xd3a2646c) ^ (a<<9);
                a = (a+0xfd7046c5) + (a<<3);
                a = (a^0xb55a4f09) ^ (a>>16);
                return a;
            }
        };

        template<typename R>
        struct uniform_operator : public thrust::unary_function<unsigned int,R>,
                                         hashable_operator<R> {
            const double lower;
            const double upper;
            const unsigned int seed;
            uniform_operator(const double& lower_, const double& upper_, unsigned int seed_) : lower(lower_), upper(upper_), seed(seed_) {}
            __host__ __device__
            R operator () (unsigned int thread_id) {
                unsigned int local_seed = seed + this->hash_operator(thread_id);
                thrust::default_random_engine rng(local_seed);
                thrust::uniform_real_distribution<R> dist(lower, upper);
                return dist(rng);
            }
        };

        template<typename R>
        struct gaussian_operator : public thrust::unary_function<unsigned int,R>,
                                          hashable_operator<R> {
            const R loc;
            const R scale;
            const unsigned int seed;
            gaussian_operator(const R& loc_, const R& scale_, const unsigned int& seed_) : loc(loc_), scale(scale_), seed(seed_) {}
            __host__ __device__
            R operator () (unsigned int thread_id) {
                unsigned int local_seed = seed + this->hash_operator(thread_id);
                thrust::default_random_engine rng(local_seed);
                thrust::normal_distribution<R> dist(loc, scale);
                return dist(rng);
            }
        };

        template<>
        __host__ __device__
        int uniform_operator<int>::operator()(unsigned int thread_id) {
            unsigned int local_seed = seed + this->hash_operator(thread_id);
            thrust::default_random_engine rng(local_seed);
            thrust::uniform_int_distribution<int> dist(lower, upper);
            return dist(rng);
        }

        template<typename T>
        void thrust_uniform(T* dst_ptr, const T* low_ptr, const T* high_ptr, int size) {
            thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + size,
                thrust::device_pointer_cast(dst_ptr),
                uniform_operator<T>(
                    *low_ptr, *high_ptr,
                    utils::randinteger<unsigned int>(0,999999)
                )
            );
        }

        template<typename T>
        void thrust_normal(T* dst_ptr, const T* loc_ptr, const T* scale_ptr, int size) {
            thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + size,
                thrust::device_pointer_cast(dst_ptr),
                gaussian_operator<T>(
                    *loc_ptr, *scale_ptr,
                    utils::randinteger<unsigned int>(0,999999)
                )
            );
        }
    }
    #define ASSERT_SCALAR(name, array_name, array)\
        ASSERT2(array.ndim() == 0, utils::make_message(\
            name, "'s ", array_name, " argument must be a scalar but got ", array_name,\
            ".shape = ", array.shape(), "."));

    struct GpuUniformImpl : public Computation {
        using Computation::Computation;
        void run() {
            Array dst = left_;
            Uniform* uni = static_cast<Uniform*>(right_.expression().get());
            Array low = uni->low();
            Array high = uni->high();
            auto op_dtype = dst.dtype();
            auto device = dst.preferred_device();
            void* dst_ptr = left_data(device);
            const void* low_ptr = argument_data(memory::Device::cpu(), 0);
            const void* high_ptr = argument_data(memory::Device::cpu(), 1);
            if (dst.dtype() == DTYPE_FLOAT) {
                thrust_uniform(static_cast<float*>(dst_ptr), static_cast<const float*>(low_ptr), static_cast<const float*>(high_ptr), dst.number_of_elements());
            } else if (dst.dtype() == DTYPE_DOUBLE) {
                thrust_uniform(static_cast<double*>(dst_ptr), static_cast<const double*>(low_ptr), static_cast<const double*>(high_ptr), dst.number_of_elements());
            } else if (dst.dtype() == DTYPE_INT32) {
                thrust_uniform(static_cast<int*>(dst_ptr), static_cast<const int*>(low_ptr), static_cast<const int*>(high_ptr), dst.number_of_elements());
            } else {
                ASSERT2(false, utils::make_message(
                    "uniform is not implemented for dtype ", dtype_to_name(dst.dtype()), "."));
            }
        }
    };

    int gpu_uniform_impl = register_implementation(
        typeid(Uniform).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            if (dest.preferred_device().is_gpu()) {
                return std::make_shared<GpuUniformImpl>(dest, operator_t, x, assignment);
            } else {
                return nullptr;
            }
        }
    );

    struct GpuNormalImpl : public Computation {
        using Computation::Computation;
        void run() {
            Array dst = left_;
            Normal* uni = static_cast<Normal*>(right_.expression().get());
            auto op_dtype = dst.dtype();
            auto device = dst.preferred_device();
            void* dst_ptr = left_data(device);
            const void* arg0_ptr = argument_data(memory::Device::cpu(), 0);
            const void* arg1_ptr = argument_data(memory::Device::cpu(), 1);
            if (dst.dtype() == DTYPE_FLOAT) {
                thrust_normal(static_cast<float*>(dst_ptr), static_cast<const float*>(arg0_ptr), static_cast<const float*>(arg1_ptr), dst.number_of_elements());
            } else if (dst.dtype() == DTYPE_DOUBLE) {
                thrust_normal(static_cast<double*>(dst_ptr), static_cast<const double*>(arg0_ptr), static_cast<const double*>(arg1_ptr), dst.number_of_elements());
            } else {
                ASSERT2(false, utils::make_message(
                    "normal is not implemented for dtype ", dtype_to_name(dst.dtype()), "."));
            }
        }
    };

    int gpu_normal_impl = register_implementation(
        typeid(Normal).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            if (dest.preferred_device().is_gpu()) {
                return std::make_shared<GpuNormalImpl>(dest, operator_t, x, assignment);
            } else {
                return nullptr;
            }
        }
    );
#endif  // DALI_USE_CUDA
    Array uniform(Array low, Array high, const std::vector<int>& shape) {
        ASSERT_SCALAR("uniform", "low", low);
        ASSERT_SCALAR("uniform", "high", high);
        std::tie(low, high) = ensure_arguments_compatible(low, high, "uniform", false);
        return Array(std::make_shared<Uniform>(low, high, shape));
    }
    Array normal(Array loc, Array scale, const std::vector<int>& shape) {
        ASSERT_SCALAR("normal", "loc", loc);
        ASSERT_SCALAR("normal", "scale", scale);
        if (loc.dtype() == DTYPE_INT32) {
            loc = loc.astype(DTYPE_DOUBLE);
        }
        if (scale.dtype() == DTYPE_INT32) {
            scale = scale.astype(DTYPE_DOUBLE);
        }
        std::tie(loc, scale) = ensure_arguments_compatible(loc, scale, "normal", false);
        return Array(std::make_shared<Normal>(loc, scale, shape));
    }
    Array bernoulli(Array prob, const std::vector<int>& shape) {
        ASSERT_SCALAR("bernoulli", "prob", prob);
        auto samples = op::uniform(Array::zeros_like(prob), Array::ones_like(prob), shape);
        return op::lessthanequal(samples, prob);
    }
    Array bernoulli_normalized(Array prob, const std::vector<int>& shape) {
        return bernoulli(prob, shape) * op::eltinv(prob);
    }

// TODO(jonathan): resuscitate
//     Assignable<Array> svd() {
//         return svd(gaussian(0.0, 1.0));
//     }
//     Assignable<Array> svd(const Assignable<Array>& preinitializer) {
//         return Assignable<Array>([preinitializer](Array& tensor, const OPERATOR_T& operator_t) {
//             ASSERT2(false, "SVD INIT: Not implemented yet");
//             /* Eigen implementation */
//             // assert(tensor.dims().size() == 2);
//             // preinitializer(tensor);
//             // auto svd = GET_MAT(tensor).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
//             // int n = tensor.dims(0);
//             // int d = tensor.dims(1);
//             // if (n < d) {
//             //     GET_MAT(tensor) = svd.tensorV().block(0, 0, n, d);
//             // } else {
//             //     GET_MAT(tensor) = svd.tensorU().block(0, 0, n, d);
//             // }
//         });
//     }

}  // namespace op
