#ifndef DALI_MATH_TENSOR_RANDOM_H
#define DALI_MATH_TENSOR_RANDOM_H

#include "dali/config.h"
#include <mshadow/tensor.h>
#include <math.h>
#include <random>

#include "dali/utils/random.h"
#include "dali/math/TensorFunctions.h"
#include "dali/math/TensorOps.h"
#include "dali/math/ThrustUtils.h"


namespace TensorOps {
    namespace random {
        #ifdef DALI_USE_CUDA
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
            const R lower;
            const R upper;
            const unsigned int seed;
            uniform_operator(R _lower, R _upper, unsigned int _seed) : lower(_lower), upper(_upper), seed(_seed) {}
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

        template<typename R>
        struct gaussian_operator : public thrust::unary_function<unsigned int,R>,
                                          hashable_operator<R> {
            const R mean;
            const R std;
            const unsigned int seed;
            gaussian_operator(R _mean, R _std, unsigned int _seed) : mean(_mean), std(_std), seed(_seed) {}
            __host__ __device__
            R operator () (unsigned int thread_id) {
                unsigned int local_seed = seed + this->hash_operator(thread_id);
                thrust::default_random_engine rng(local_seed);
                thrust::normal_distribution<R> dist(mean, std);
                return dist(rng);
            }
        };

        template<>
        int gaussian_operator<int>::operator()(unsigned int thread_id) {
            unsigned int local_seed = seed + this->hash_operator(thread_id);
            thrust::default_random_engine rng(local_seed);
            thrust::normal_distribution<float> dist((float)mean, (float)std);
            return (int)dist(rng);
        }

        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<mshadow::gpu, ndims, R> A, R lower, R upper) {
            // about 63x faster than SampleUniform for gpu
            thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + A.shape_.Size(),
                    to_thrust(A),
                    uniform_operator<R>(lower, upper, utils::randinteger<unsigned int>(0,999999)));
        }
        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<mshadow::gpu, ndims, R> A, R mean, R std) {
            thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + A.shape_.Size(),
                    to_thrust(A),
                    gaussian_operator<R>(mean, std, utils::randinteger<unsigned int>(0,999999)));
        }
        #endif

        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<mshadow::cpu, ndims, R> t, R lower, R upper) {
            mshadow::Random<mshadow::cpu, R> generator(utils::randint(0,999999));
            generator.SampleUniform(&t, lower, upper);
        }

        template<int ndims, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<mshadow::cpu, ndims, int> t, int lower, int upper) {
            std::uniform_int_distribution<int> dist(lower, upper);
            auto& gen = utils::random::generator();
            for (int i = 0; i < t.shape_.Size(); ++i) {
                *(t.dptr_ + i) = dist(gen);
            }
        }

        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<mshadow::cpu, ndims, R> t, R mean, R std) {
            mshadow::Random<mshadow::cpu, R> generator(utils::randint(0,999999));
            generator.SampleGaussian(&t, mean, std);
        }

        template<int ndims, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<mshadow::cpu, ndims, int> t, int mean, int std) {
            std::normal_distribution<float> dist((float)mean, (float)std);
            auto& gen = utils::random::generator();
            for (int i = 0; i < t.shape_.Size(); ++i) {
                *(t.dptr_ + i) = (int)dist(gen);
            }
        }

        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void bernoulli(tensor_t<Device, ndims, R> t, R prob) {
            random::uniform(t, (R)0.0, (R)1.0);
            t = mshadow::expr::F<op::threshold<R>>(t, prob);
        }

        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void bernoulli_normalized(tensor_t<Device, ndims, R> t, R prob) {
            random::uniform(t, (R)0.0, (R)1.0);
            t = mshadow::expr::F<op::threshold<R>>(t, prob) * (1.0 / prob);
        }
    }
}

#endif
