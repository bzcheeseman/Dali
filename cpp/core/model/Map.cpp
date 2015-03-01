#include "core/model/Map.h"

using std::make_shared;
using std::string;
using std::vector;

namespace model {
    template<typename REAL_t>
    using shared_mat = typename Layer<REAL_t>::shared_mat;

    template<typename REAL_t>
    Layer<REAL_t>::Layer(int input_size, int output_size, bool use_bias, double bound) :
            input_size(input_size),
            output_size(output_size),
            use_bias(use_bias) {
        mult = make_shared<mat>(output_size, input_size, -bound/2.0, bound/2.0);
        if (use_bias)
            bias = make_shared<mat>(output_size, 1, -bound/2.0, bound/2.0);
    }
    template<typename REAL_t>
    shared_mat<REAL_t> Layer<REAL_t>::operator() (graph_t& G, shared_mat input) {
        return activate(G, input);
    }

    template<typename REAL_t>
    shared_mat<REAL_t> Layer<REAL_t>::activate(graph_t& G, shared_mat input) {
        if (use_bias) {
            return G.add(G.mul(mult, input), bias);
        } else {
            return G.mul(mult, input);
        }
    }

    template<typename REAL_t>
    vector<shared_mat<REAL_t> > Layer<REAL_t>::parameters() const {
        if (use_bias) {
            return {mult, bias};
        } else {
            return {mult};
        }
    }

    template class Layer<float>;
    template class Layer<double>;
};
