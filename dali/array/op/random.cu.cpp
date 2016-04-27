#include "dali/array/op/random.h"

#include "dali/array/function/function.h"
#include "dali/array/op/impl/ops.h"
#include "dali/array/op/impl/random.h"
#include "dali/utils/random.h"
#include "dali/array/TensorFunctions.h"

using std::vector;

////////////////////////////////////////////////////////////////////////////////
//                         DEFINING THRUST OPERATORS                          //
//                                ---------                                   //
//  Those operators are later used in GPU initialization. This is a           //
//  surprisingly verbose way of doing this, but results in significant        //
//  performance boost over mshadow initialization on GPU                      //
////////////////////////////////////////////////////////////////////////////////

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

    }
#endif


////////////////////////////////////////////////////////////////////////////////
//                         THE MOTHER OF ALL INITIALIZERS                     //
//                                ---------                                   //
//  Small set of common sanity checks for all initializers.                   //
////////////////////////////////////////////////////////////////////////////////


// TODO(jonathan,szymon): merge this initializer with "fill" initializer in other.h
template<typename Class, typename... Args>
struct Initializer : public Function<Initializer<Class, Args...>, Array, Args...> {
    static void prepare_output(Array& out, const Args&... args) {
        ASSERT2(!out.is_stateless(),
                "Weight initializer must only be used for an array which is not stateless");
    }


    template<int devT, typename T>
    void typed_eval(MArray<devT, T> output, const Args&... args) {
        ASSERT2(output.array.spans_entire_memory(),
                "Currently array initialization is only supported for Arrays which own entire underlying memory (are not views)");
        auto self = static_cast<Class*>(this);
        self->template initialize(output, args...);
    }
};

////////////////////////////////////////////////////////////////////////////////
//                       INITIALIZER DEFINITIONS                              //
//                                ---------                                   //
//  Some of them are very case specific, but they also work.                  //
////////////////////////////////////////////////////////////////////////////////

struct GaussianInitializer : public Initializer<GaussianInitializer, const double&, const double&> {
#ifdef DALI_USE_CUDA
    template<typename T>
    void initialize(MArray<memory::DEVICE_T_GPU, T> out, const double& mean, const double& std) {
        thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + out.array.number_of_elements(),
                out.to_thrust(memory::AM_OVERWRITE),
                gaussian_operator<T>(mean, std, utils::randinteger<unsigned int>(0,999999)));
    }
#endif

    template<typename T>
    void initialize(MArray<memory::DEVICE_T_CPU, T> out, const double& mean, const double& std) {
        mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
        auto m_out = out.d1(memory::AM_OVERWRITE);
        generator.SampleGaussian(&m_out, mean, std);
    }

    void initialize(MArray<memory::DEVICE_T_CPU, int> out, const double& mean, const double& std) {
        std::normal_distribution<double> dist(mean, std);
        auto& gen = utils::random::generator();
        auto ptr = out.ptr(memory::AM_OVERWRITE);
        for (int i = 0; i < out.array.number_of_elements(); ++i) {
            *(ptr + i) = (int)dist(gen);
        }
    }
};

struct UniformInitializer : public Initializer<UniformInitializer, const double&, const double&> {

#ifdef DALI_USE_CUDA
    template<typename T>
    void initialize(MArray<memory::DEVICE_T_GPU, T> out, const double& lower, const double& upper) {
        // // about 63x faster than SampleUniform for gpu
        // thrust::transform(
        //         thrust::make_counting_iterator(0),
        //         thrust::make_counting_iterator(0) + A.shape_.Size(),
        //         to_thrust(A),
        //         uniform_operator<R>(lower, upper, utils::randinteger<unsigned int>(0,999999)));
    }
#endif

    template<typename T>
    void initialize(MArray<memory::DEVICE_T_CPU, T> out, const double& lower, const double& upper) {
        // mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
        // auto m_out = out.d1(memory::AM_OVERWRITE);
        // generator.SampleUniform(&m_out, lower, upper);
    }

    void initialize(MArray<memory::DEVICE_T_CPU, int> out, const double& lower, const double& upper) {
        // std::uniform_int_distribution<double> dist(lower, upper);
        // auto& gen = utils::random::generator();
        // auto ptr = out.ptr(memory::AM_OVERWRITE);
        // for (int i = 0; i < out.array.number_of_elements(); ++i) {
        //     *(ptr + i) = (int)dist(gen);
        // }
    }
};

struct BernoulliInitialzier : public Initializer<BernoulliInitialzier, const double&> {
    template<int devT, typename T>
    void initialize(MArray<devT,T> out, const double& prob) {
        UniformInitializer().initialize(out, 0.0, 1.0);
        out.d1(memory::AM_OVERWRITE) = mshadow::expr::F<tensor_ops::op::threshold<T>>(out.d1(), prob);
    }
};

struct BernoulliNormalizerInitializer : public Initializer<BernoulliNormalizerInitializer, const double&> {
    template<int devT, typename T>
    void initialize(MArray<devT,T> out, const double& prob) {
        UniformInitializer().initialize(out, 0.0, 1.0);
        out.d1(memory::AM_OVERWRITE) = mshadow::expr::F<tensor_ops::op::threshold<T>>(out.d1(), prob) * (1.0 / prob);
    }
};

////////////////////////////////////////////////////////////////////////////////
//                  WRAPPING STRUCTS INTO FUNCTIONS                           //
////////////////////////////////////////////////////////////////////////////////

namespace tensor_ops {
    namespace random {
        AssignableArray gaussian(const double& mean, const double& std) {
            return GaussianInitializer::run(mean, std);
        }

        AssignableArray uniform(const double& lower, const double& upper) {
            return UniformInitializer::run(lower, upper);
        }

        AssignableArray bernoulli(const double& prob) {
            return BernoulliInitialzier::run(prob);

        }

        AssignableArray bernoulli_normalized(const double& prob) {
            return BernoulliNormalizerInitializer::run(prob);
        }
    } // namespace random
} // namespace tensor_ops