#include "core/Error.h"

using std::make_shared;


template<typename REAL_t>
SHARED_MAT Error<REAL_t>::cross_entropy(GRAPH& G,
                                        const Seq<SHARED_MAT>& expected,
                                        const Seq<SHARED_MAT>& prediction) {
    SHARED_MAT one = make_shared<MAT>(1, 1);
    one->w(0,0) = 1.0;
    SHARED_MAT zero = make_shared<MAT>(1, 1);
    zero->w(0,0) = 0.0;
    SHARED_MAT eps = make_shared<MAT>(1, 1);
    eps->w(0,0) = 1e-5;

    SHARED_MAT error = make_shared<MAT>(1,1);
    error->w(0,0) = 0;

    assert(expected.size() == prediction.size());
    for (int idx = 0; idx < expected.size(); ++idx) {
        SHARED_MAT part1 = G.mul(expected[idx],
                                 G.log(G.add(eps, prediction[idx])));
        SHARED_MAT oneminus_expected = G.sub(one, expected[idx]);
        SHARED_MAT oneminus_prediction = G.sub(one, prediction[idx]);

        SHARED_MAT part2 = G.mul(oneminus_expected,
                                 G.log(G.add(eps, oneminus_prediction)));

        error = G.add(error, G.sub(zero, G.add(part1, part2)));
    }
    return error;
}

template class Error<float>;
template class Error<double>;
