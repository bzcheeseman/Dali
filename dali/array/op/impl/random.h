#ifndef DALI_ARRAY_TENSOR_RANDOM_H
#define DALI_ARRAY_TENSOR_RANDOM_H

#include "dali/config.h"
#include <mshadow/tensor.h>
#include <math.h>
#include <random>

#include "dali/utils/random.h"
#include "dali/array/TensorFunctions.h"
#include "dali/array/op/impl/ops.h"
#include "dali/array/op/impl/thrust_utils.h"

namespace tensor_ops {
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

        template<typename R>
        struct gaussian_operator : public thrust::unary_function<unsigned int,R>,
                                          hashable_operator<R> {
            const double mean;
            const double std;
            const unsigned int seed;
            gaussian_operator(const double& mean_, const double& std_, const unsigned int& seed_) : mean(mean_), std(std_), seed(seed_) {}
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
            thrust::normal_distribution<double> dist(mean, std);
            return (int)dist(rng);
        }

        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<mshadow::gpu, ndims, R> A, const double& lower, const double& upper) {
            // about 63x faster than SampleUniform for gpu
            thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + A.shape_.Size(),
                    to_thrust(A),
                    uniform_operator<R>(lower, upper, utils::randinteger<unsigned int>(0,999999)));
        }
        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<mshadow::gpu, ndims, R> A, const double& mean, const double& std) {
            thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + A.shape_.Size(),
                    to_thrust(A),
                    gaussian_operator<R>(mean, std, utils::randinteger<unsigned int>(0,999999)));
        }
#endif

        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<mshadow::cpu, ndims, R> t, const double& lower, const double& upper) {
            mshadow::Random<mshadow::cpu, R> generator(utils::randint(0,999999));
            generator.SampleUniform(&t, lower, upper);
        }

        template<int ndims, template <typename,int,typename> class tensor_t>
        void uniform(tensor_t<mshadow::cpu, ndims, int> t, const double& lower, const double& upper) {
            std::uniform_int_distribution<double> dist(lower, upper);
            auto& gen = utils::random::generator();
            for (int i = 0; i < t.shape_.Size(); ++i) {
                *(t.dptr_ + i) = (int)dist(gen);
            }
        }

        template<int ndims, typename R, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<mshadow::cpu, ndims, R> t, const double& mean, const double& std) {
            mshadow::Random<mshadow::cpu, R> generator(utils::randint(0,999999));
            generator.SampleGaussian(&t, mean, std);
        }

        template<int ndims, template <typename,int,typename> class tensor_t>
        void gaussian(tensor_t<mshadow::cpu, ndims, int> t, const double& mean, const double& std) {
            std::normal_distribution<double> dist(mean, std);
            auto& gen = utils::random::generator();
            for (int i = 0; i < t.shape_.Size(); ++i) {
                *(t.dptr_ + i) = (int)dist(gen);
            }
        }

        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void bernoulli(tensor_t<Device, ndims, R> t, const double& prob) {
            random::uniform(t, 0.0, 1.0);
            t = mshadow::expr::F<op::threshold<R>>(t, prob);
        }

        template<typename Device, int ndims, typename R, template <typename,int,typename> class tensor_t>
        void bernoulli_normalized(tensor_t<Device, ndims, R> t, const double& prob) {
            random::uniform(t, 0.0, 1.0);
            t = mshadow::expr::F<op::threshold<R>>(t, prob) * (1.0 / prob);
        }
    }
}
#endif
