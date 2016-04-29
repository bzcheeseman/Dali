#include "initializer.h"

#ifdef DALI_USE_CUDA
    #include <thrust/random.h>
#endif

#include "dali/array/function/function.h"
#include "dali/array/TensorFunctions.h"
#include "dali/utils/random.h"
#include "dali/runtime_config.h"

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
struct Initializer : public Function<Class, Array, Args...> {
    static bool disable_output_shape_check; // = true;
    static bool disable_output_dtype_check; // = true;

    static DType deduce_output_dtype(const Args&... args) {
        return DTYPE_FLOAT;
    }

    static vector<int> deduce_output_shape(const Args&... args) {
        return {};
    }

    static memory::Device deduce_output_device(const Args&... args) {
        return memory::default_preferred_device;
    }

    template<int devT, typename T>
    void assert_contiguous_memory(const TypedArray<devT,T>& out) {
        ASSERT2(out.array.contiguous_memory(),
                "Currently array initialization is only supported for Arrays are contiguous view of underlying memory (no striding)");
    }
};

template<typename Class, typename... Args>
bool Initializer<Class, Args...>::disable_output_shape_check = true;

template<typename Class, typename... Args>
bool Initializer<Class, Args...>::disable_output_dtype_check = true;

////////////////////////////////////////////////////////////////////////////////
//                       INITIALIZER DEFINITIONS                              //
//                                ---------                                   //
//  Some of them are very case specific, but they also work.                  //
////////////////////////////////////////////////////////////////////////////////

struct GaussianInitializer : public Initializer<GaussianInitializer, const double&, const double&> {
#ifdef DALI_USE_CUDA
    template<typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out, const double& mean, const double& std) {
        assert_contiguous_memory(out);
        thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + out.array.number_of_elements(),
                out.to_thrust(memory::AM_OVERWRITE),
                gaussian_operator<T>(mean, std, utils::randinteger<unsigned int>(0,999999)));
    }
#endif

    template<typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out, const double& mean, const double& std) {
        mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
        auto m_out = out.d1(memory::AM_OVERWRITE);
        generator.SampleGaussian(&m_out, mean, std);
    }

    void typed_eval(TypedArray<memory::DEVICE_T_CPU, int> out, const double& mean, const double& std) {
        assert_contiguous_memory(out);
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
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out, const double& lower, const double& upper) {
        assert_contiguous_memory(out);
        // about 63x faster than SampleUniform for gpu
        thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + out.array.number_of_elements(),
                out.to_thrust(memory::AM_OVERWRITE),
                uniform_operator<T>(lower, upper, utils::randinteger<unsigned int>(0,999999)));
    }
#endif

    template<typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out, const double& lower, const double& upper) {
        mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
        auto m_out = out.d1(memory::AM_OVERWRITE);
        generator.SampleUniform(&m_out, lower, upper);
    }

    void typed_eval(TypedArray<memory::DEVICE_T_CPU, int> out, const double& lower, const double& upper) {
        assert_contiguous_memory(out);
        // uniform_int_distribution can only tak ints as per standard
        // clang is more permissive here.
        std::uniform_int_distribution<int> dist(lower, upper);
        auto& gen = utils::random::generator();
        auto ptr = out.ptr(memory::AM_OVERWRITE);
        for (int i = 0; i < out.array.number_of_elements(); ++i) {
            *(ptr + i) = (int)dist(gen);
        }
    }
};

struct BernoulliInitialzier : public Initializer<BernoulliInitialzier, const double&> {
    template<int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const double& prob) {
        UniformInitializer().typed_eval(out, 0.0, 1.0);
        out.d1(memory::AM_OVERWRITE) = mshadow::expr::F<tensor_ops::op::threshold<T>>(out.d1(), prob);
    }
};

struct BernoulliNormalizerInitializer : public Initializer<BernoulliNormalizerInitializer, const double&> {
    template<int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const double& prob) {
        UniformInitializer().typed_eval(out, 0.0, 1.0);
        out.d1(memory::AM_OVERWRITE) = mshadow::expr::F<tensor_ops::op::threshold<T>>(out.d1(), prob) * (1.0 / prob);
    }
};

template<typename ConstT>
struct ConstantInitializer : public Initializer<ConstantInitializer<ConstT>, const double&> {
    static DType deduce_output_dtype(const ConstT& constant) {
        return template_to_dtype<ConstT>();
    }


    template<int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const ConstT& constant) {
        assert_dali_dtype<ConstT>();
        out.d1(memory::AM_OVERWRITE) = (T)constant;
    }
};

////////////////////////////////////////////////////////////////////////////////
//                  WRAPPING STRUCTS INTO FUNCTIONS                           //
////////////////////////////////////////////////////////////////////////////////

namespace initializer {

    AssignableArray empty() {
        return AssignableArray([](Array&){
            // do nothing
        });
    }
    AssignableArray zeros() {
        return AssignableArray([](Array& out){
            // efficient lazy clearing of memory.
            out.clear();
        });
    }

    AssignableArray ones() {
        return ConstantInitializer<float>::run(1.0);
    }

    template<typename ConstT>
    AssignableArray fill(const ConstT& constant) {
        return ConstantInitializer<ConstT>::run(constant);
    }

    template AssignableArray fill(const int&);
    template AssignableArray fill(const float&);
    template AssignableArray fill(const double&);

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

    AssignableArray eye(double diag) {
        return AssignableArray([diag](Array& tensor) {
            ASSERT2(false, "eye: Not implemented yet");

            // #ifdef DALI_USE_CUDA
            //     if (tensor.compute_me_on_gpu()) {
            //         tensor_ops::eye(tensor.mutable_gpu_data(), diag);
            //         return;
            //     }
            // #endif
            // tensor_ops::eye(tensor.mutable_cpu_data(), diag);
        });

    }
    AssignableArray svd(AssignableArray preinitializer) {
        return AssignableArray([preinitializer](Array& tensor) {
            ASSERT2(false, "SVD INIT: Not implemented yet");
            /* Eigen implementation */
            // assert(tensor.dims().size() == 2);
            // preinitializer(tensor);
            // auto svd = GET_MAT(tensor).jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            // int n = tensor.dims(0);
            // int d = tensor.dims(1);
            // if (n < d) {
            //     GET_MAT(tensor) = svd.tensorV().block(0, 0, n, d);
            // } else {
            //     GET_MAT(tensor) = svd.tensorU().block(0, 0, n, d);
            // }
        });
    }
} // namespace initializer
