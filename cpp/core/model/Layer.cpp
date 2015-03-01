#include "core/model/Layer.h"

using std::make_shared;
using std::string;
using std::vector;

namespace model {
    template<typename REAL_t>
    Layer<REAL_t>::Layer(int input_size, int output_size, bool use_bias, double bound) :
            input_size(input_size),
            output_size(output_size),
            use_bias(use_bias) {
        mult = make_shared<MAT>(output_size, input_size, -bound/2.0, bound/2.0);
        if (use_bias)
            bias = make_shared<MAT>(output_size, 1, -bound/2.0, bound/2.0);
    }

    template<typename REAL_t>
    SHARED_MAT Layer<REAL_t>::activate(GRAPH& G, SHARED_MAT input) {
        if (use_bias) {
            return G.mul_with_bias(mult, input, bias);
        } else {
            return G.mul(mult, input);
        }
    }

    template<typename REAL_t>
    vector<SHARED_MAT> Layer<REAL_t>::parameters() const {
        if (use_bias) {
            return {mult, bias};
        } else {
            return {mult};
        }
    }

    template class Layer<float>;
    template class Layer<double>;
};
