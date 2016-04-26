#include "dali/array/op/random.h"

#include "dali/array/op/impl/ops.h"

template<typename InitializerT>
struct RandomInitializer : public Function<RandomInitializer<InitializerT>, Array, InitializerT> {
    static const bool disable_output_shape_check = true;
    static const bool disable_output_dtype_check = true;

    static DType deduce_output_dtype(const InitializerT&) {
        return DTYPE_FLOAT;
    }

    static std::vector<int> deduce_output_shape(const InitializerT&) {
        return {};
    }

    static memory::Device deduce_output_device(const InitializerT&) {
        return memory::default_preferred_device;
    }

    template<int devT, typename T, typename... Args>
    void typed_eval(MArray<devT, T> input, const Args&... args) {
        InitializerT::sample(input.d1(memory::AM_OVERWRITE), ...args);
    }
};

struct gaussian_initializer {
	template<typename tensor_t>
	static sample(tensor_t& tensor, const double& mean, const double& std) {
		tensor_ops::random::gaussian(tensor, mean, std);
	}
};

struct uniform_initializer {
	template<typename tensor_t>
	static sample(tensor_t& tensor, const double& low, const double& high) {
		tensor_ops::random::uniform(tensor, low, high);
	}
};

AssignableArray gaussian(double mean, double std) {
    return RandomInitializer<gaussian_initializer>::run(mean, std);
}

AssignableArray uniform(double low, double high) {
	return RandomInitializer<uniform_initializer>::run(low, high);
}
