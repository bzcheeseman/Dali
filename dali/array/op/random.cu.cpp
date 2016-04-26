#include "dali/array/op/random.h"

#include "dali/array/function/function.h"
#include "dali/array/op/impl/ops.h"
#include "dali/array/op/impl/random.h"

// TODO(jonathan,szymon): merge this initializer with "fill" initializer in other.h
template<typename InitializerT, typename... Args>
struct RandomInitializer : public Function<RandomInitializer<InitializerT, Args...>, Array, Args...> {
    static const bool disable_output_shape_check = true;
    static const bool disable_output_dtype_check = true;

    static DType deduce_output_dtype(const Args&...) {
        return DTYPE_FLOAT;
    }

    static std::vector<int> deduce_output_shape(const Args&...) {
        return {};
    }

    static memory::Device deduce_output_device(const Args&...) {
        return memory::default_preferred_device;
    }

    static DType deduce_computation_dtype(const Array& out, const Args&...) {
        return ReduceOverArgs<CommonPropertyExtractor<DTypeProperty>>::reduce(out);
    }

    template<int devT, typename T>
    void typed_eval(MArray<devT, T> input, const Args&... args) {
        InitializerT::sample(input.d1(memory::AM_OVERWRITE), args...);
    }
};

struct gaussian_initializer {
    template<typename tensor_t>
    static void sample(tensor_t tensor, const double& mean, const double& std) {
        tensor_ops::random::gaussian(tensor, mean, std);
    }
};

struct uniform_initializer {
    template<typename tensor_t>
    static void sample(tensor_t tensor, const double& low, const double& high) {
        tensor_ops::random::uniform(tensor, low, high);
    }
};

struct bernoulli_initializer {
    template<typename tensor_t>
    static void sample(tensor_t tensor, const double& prob) {
        tensor_ops::random::bernoulli(tensor, prob);
    }
};

struct bernoulli_normalized_initializer {
    template<typename tensor_t>
    static void sample(tensor_t tensor, const double& prob) {
        tensor_ops::random::bernoulli_normalized(prob);
    }
};

namespace tensor_ops {
    namespace random {
        AssignableArray gaussian(const double& mean, const double& std) {
            return RandomInitializer<gaussian_initializer, double, double>::run(mean, std);
        }

        AssignableArray uniform(const double& low, const double& high) {
            return RandomInitializer<uniform_initializer, double, double>::run(low, high);
        }

        AssignableArray bernoulli(const double& prob) {
            return RandomInitializer<bernoulli_initializer, double>::run(prob);
        }

        AssignableArray bernoulli_normalized(const double& prob) {
            return RandomInitializer<bernoulli_normalized_initializer, double>::run(prob);
        }
    } // namespace random
} // namespace tensor_ops
