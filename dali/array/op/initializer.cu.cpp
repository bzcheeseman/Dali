#include "initializer.h"

#include "dali/array/array.h"
#include "dali/array/dtype.h"
#include "dali/array/memory/device.h"
#include "dali/runtime_config.h"

#ifdef DALI_USE_CUDA
    #include <thrust/random.h>
#endif

#include "dali/array/function/function.h"
#include "dali/array/function/operator.h"
#include "dali/array/functor.h"
#include "dali/utils/random.h"

#include "dali/array/lazy/binary.h"

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
        ASSERT2(false, "Calling unspecialized uniform not allowed");
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
    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out, const double& mean, const double& std) {
        assert_contiguous_memory(out);
        thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(0) + out.array.number_of_elements(),
                out.to_thrust(memory::AM_OVERWRITE),
                gaussian_operator<T>(mean, std, utils::randinteger<unsigned int>(0,999999)));
    }
#endif

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out, const double& mean, const double& std) {
        assert_contiguous_memory(out);
        mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
        if (operator_t == OPERATOR_T_EQL) {
            auto m_out = out.contiguous_d1(memory::AM_OVERWRITE);
            generator.SampleGaussian(&m_out, mean, std);
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for GaussianInitializer");
        }
    }

    template<OPERATOR_T operator_t>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, int> out, const double& mean, const double& std) {
        assert_contiguous_memory(out);
        std::normal_distribution<double> dist(mean, std);
        auto& gen = utils::random::generator();
        if (operator_t == OPERATOR_T_EQL) {
            auto ptr = out.ptr(memory::AM_OVERWRITE);
            for (int i = 0; i < out.array.number_of_elements(); ++i) {
                *(ptr + i) = (int)dist(gen);
            }
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for GaussianInitializer");
        }
    }
};

struct UniformInitializer : public Initializer<UniformInitializer, const double&, const double&> {

    static void verify(const double& lower, const double& upper) {
        ASSERT2(lower < upper,
            utils::MS() << "Uniform initialzer must have nonempty interval, got ["
                        << lower << "," << upper <<"]");
    }


#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out, const double& lower, const double& upper) {
        assert_contiguous_memory(out);
        // about 63x faster than SampleUniform for gpu
        if (operator_t == OPERATOR_T_EQL) {
            thrust::transform(
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + out.array.number_of_elements(),
                    out.to_thrust(memory::AM_OVERWRITE),
                    uniform_operator<T>(lower, upper, utils::randinteger<unsigned int>(0,999999)));
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for UniformInitializer");
        }
    }
#endif

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out, const double& lower, const double& upper) {
        mshadow::Random<mshadow::cpu, T> generator(utils::randint(0,999999));
        auto m_out = out.contiguous_d1(memory::AM_OVERWRITE);
        generator.SampleUniform(&m_out, lower, upper);
    }

    template<OPERATOR_T operator_t>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, int> out, const double& lower, const double& upper) {
        assert_contiguous_memory(out);
        // uniform_int_distribution can only tak ints as per standard
        // clang is more permissive here.
        std::uniform_int_distribution<int> dist(lower, upper);
        auto& gen = utils::random::generator();
        if (operator_t == OPERATOR_T_EQL) {
            auto ptr = out.ptr(memory::AM_OVERWRITE);
            for (int i = 0; i < out.array.number_of_elements(); ++i) {
                *(ptr + i) = (int)dist(gen);
            }
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for UniformInitializer");
        }
    }
};

struct ArangeInitializer : public Initializer<ArangeInitializer> {

#ifdef DALI_USE_CUDA
    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_GPU, T> out) {
        if (operator_t == OPERATOR_T_EQL) {
            assert_contiguous_memory(out);
            auto cnt_iter = thrust::make_counting_iterator(0);
            thrust::copy(cnt_iter,
                         cnt_iter + out.array.number_of_elements(),
                         out.to_thrust(memory::AM_OVERWRITE));
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for ArangeInitializer");
        }
    }
#endif

    template<OPERATOR_T operator_t, typename T>
    void typed_eval(TypedArray<memory::DEVICE_T_CPU, T> out) {
        if (operator_t == OPERATOR_T_EQL) {
            auto ptr = out.ptr(memory::AM_OVERWRITE);
            for (int i = 0; i < out.array.number_of_elements(); ++i) {
                *(ptr + i) = (T)i;
            }
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for ArangeInitializer");
        }
    }
};

struct BernoulliInitializer : public Initializer<BernoulliInitializer, const double&> {
    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const double& prob) {
        if (operator_t == OPERATOR_T_EQL) {
            UniformInitializer().template typed_eval<operator_t>(out, 0.0, 1.0);
            operator_assign<operator_t, 1>(out, mshadow::expr::F<functor::threshold<T>>(out.contiguous_d1(), prob));
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for BernoulliInitializer");
        }
    }
};

struct BernoulliNormalizedInitializer : public Initializer<BernoulliNormalizedInitializer, const double&> {
    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const double& prob) {
        if (operator_t == OPERATOR_T_EQL) {
            UniformInitializer().template typed_eval<operator_t>(out, 0.0, 1.0);
            operator_assign<operator_t, 1>(out, mshadow::expr::F<functor::threshold<T>>(out.contiguous_d1(), prob) * (1.0 / prob));
        } else {
            ASSERT2(false,
                utils::MS() << operator_to_name(operator_t)
                            << " not yet implemented for BernoulliNormalizedInitializer");
        }
    }
};

template<typename ConstT>
struct ConstantInitializer : public Initializer<ConstantInitializer<ConstT>, const double&> {
    static DType deduce_output_dtype(const ConstT& constant) {
        return template_to_dtype<ConstT>();
    }

    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const ConstT& constant) {
        assert_dali_dtype<ConstT>();
        operator_assign<operator_t, 1>(out, (T)constant);
    }
};

struct EyeInitializer : public Initializer<EyeInitializer, const double&> {
    template<OPERATOR_T operator_t, int devT, typename T>
    void typed_eval(TypedArray<devT,T> out, const double& diag) {
        operator_assign<operator_t, 2>(
            out,
            mshadow::expr::FIndexed<functor::eye<T>>(
                out.d2(),
                diag
            )
        );
    }
};

///////////////////////////////////////AM_OVERWRITE/////////////////////////////////////////
//                  WRAPPING STRUCTS INTO FUNCTIONS                           //
////////////////////////////////////////////////////////////////////////////////

namespace initializer {

    Assignable<Array> empty() {
        return Assignable<Array>([](Array&, const OPERATOR_T& operator_t){
            // do nothing
        });
    }
    Assignable<Array> zeros() {
        return Assignable<Array>([](Array& out, const OPERATOR_T& operator_t){
            // efficient lazy clearing of memory.
            if (operator_t == OPERATOR_T_EQL || operator_t == OPERATOR_T_MUL) {
                out.clear();
            } else if (operator_t == OPERATOR_T_ADD || operator_t == OPERATOR_T_SUB) {
                // add 0 or remove 0 does nothing
            } else if (operator_t == OPERATOR_T_DIV) {
                // divide by zero...
                out /= ConstantInitializer<double>::run(0.0);
            }
        });
    }

    Assignable<Array> ones() {
        return ConstantInitializer<float>::run(1.0);
    }
    Assignable<Array> arange() {
        return ArangeInitializer::run();
    }

    template<typename ConstT>
    Assignable<Array> fill(const ConstT& constant) {
        return ConstantInitializer<ConstT>::run(constant);
    }

    template Assignable<Array> fill(const int&);
    template Assignable<Array> fill(const float&);
    template Assignable<Array> fill(const double&);

    Assignable<Array> gaussian(const double& mean, const double& std) {
        return GaussianInitializer::run(mean, std);
    }

    Assignable<Array> uniform(const double& lower, const double& upper) {
        return UniformInitializer::run(lower, upper);
    }

    Assignable<Array> bernoulli(const double& prob) {
        return BernoulliInitializer::run(prob);
    }

    Assignable<Array> bernoulli_normalized(const double& prob) {
        return BernoulliNormalizedInitializer::run(prob);
    }

    Assignable<Array> eye(const double& diag) {
        return EyeInitializer::run(diag);
    }

    Assignable<Array> svd() {
        return svd(gaussian(0.0, 1.0));
    }
    Assignable<Array> svd(const Assignable<Array>& preinitializer) {
        return Assignable<Array>([preinitializer](Array& tensor, const OPERATOR_T& operator_t) {
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
