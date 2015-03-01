#include "core/model/Model.h"

using std::string;
using std::vector;

/* Model<REAL_t> */

namespace model {
    template<typename REAL_t>
    SHARED_MAT Model<REAL_t>::operator() (GRAPH& G, SHARED_MAT input) {
        return activate(G, input);
    }

    // Save to file.
    template<typename REAL_t>
    void Model<REAL_t>::save(string dirname) const {
        vector<SHARED_MAT> params = parameters();
        utils::save_matrices(params, dirname);
    };

    // Load from file.
    template<typename REAL_t>
    void Model<REAL_t>::load(string dirname) {
        vector<SHARED_MAT> params = parameters();
        utils::load_matrices(params, dirname);
    };

    template class Model<float>;
    template class Model<double>;
};

/* RecurrentModel<REAL_t> */

namespace model {
    template<typename REAL_t>
    Seq<SHARED_MAT> RecurrentModel<REAL_t>::operator() (
            GRAPH& G, const Seq<SHARED_MAT>& seq) {
        return activate_sequence(G, seq);
    }

    template<typename REAL_t>
    Seq<SHARED_MAT> RecurrentModel<REAL_t>::activate_sequence(
            GRAPH& G, const Seq<SHARED_MAT>& seq) {
        reset();
        Seq<SHARED_MAT> result;
        for (const SHARED_MAT& input: seq) {
            result.push_back(this->activate(G, input));
        }
        return result;
    }

    template class RecurrentModel<float>;
    template class RecurrentModel<double>;
};
