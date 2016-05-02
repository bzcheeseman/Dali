#include "initializer.h"

#ifdef DALI_USE_CUDA
    #include <thrust/random.h>
#endif

#include "dali/array/function/function.h"
#include "dali/array/function/operator.h"
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
            operator_assign<operator_t, 1>(out, mshadow::expr::F<tensor_ops::op::threshold<T>>(out.contiguous_d1(), prob));
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
            operator_assign<operator_t, 1>(out, mshadow::expr::F<tensor_ops::op::threshold<T>>(out.contiguous_d1(), prob) * (1.0 / prob));
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

///////////////////////////////////////AM_OVERWRITE/////////////////////////////////////////
//                  WRAPPING STRUCTS INTO FUNCTIONS                           //
////////////////////////////////////////////////////////////////////////////////

namespace initializer {

    AssignableArray empty() {
        return AssignableArray([](Array&, const OPERATOR_T& operator_t){
            // do nothing
        });
    }
    AssignableArray zeros() {
        return AssignableArray([](Array& out, const OPERATOR_T& operator_t){
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

    AssignableArray ones() {
        return ConstantInitializer<float>::run(1.0);
    }
    AssignableArray arange() {
        return ArangeInitializer::run();
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
        return BernoulliInitializer::run(prob);
    }

    AssignableArray bernoulli_normalized(const double& prob) {
        return BernoulliNormalizedInitializer::run(prob);
    }

    AssignableArray eye(double diag) {
        return AssignableArray([diag](Array& tensor, const OPERATOR_T& operator_t) {
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
        return AssignableArray([preinitializer](Array& tensor, const OPERATOR_T& operator_t) {
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
