#include "uniform.h"
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
#include "dali/array/expression/computation.h"
#include "dali/utils/make_message.h"
#include "dali/utils/random.h"

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

    template<>
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
}

namespace op {
    struct GpuUniformImpl : public Computation {
        using Computation::Computation;
        void run() {
            Array dst = left_;
            op::Uniform* uni = static_cast<op::Uniform*>(right_.expression().get());
            Array low = uni->low_;
            Array high = uni->high_;
            auto op_dtype = dst.dtype();
            auto device = dst.preferred_device();
            void* dst_ptr = dst.memory()->overwrite_data(device);
            const void* low_ptr = low.memory()->readonly_data(memory::Device::cpu());
            const void* high_ptr = high.memory()->readonly_data(memory::Device::cpu());

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
        typeid(op::Uniform).name(),
        [](Array dest, OPERATOR_T operator_t, Array x, Array assignment) -> std::shared_ptr<Computation> {
            if (dest.preferred_device().is_gpu()) {
                return std::make_shared<GpuUniformImpl>(dest, operator_t, x, assignment);
            } else {
                return nullptr;
            }
        }
    );
}
