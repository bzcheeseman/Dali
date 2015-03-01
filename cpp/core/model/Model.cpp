#include "core/model/Model.h"

using std::string;

namespace model {
    // Save to file.
    template<typename REAL_t>
    void Model<REAL_t>::save(string dirname) const {
        utils::save_matrices(parameters(), dirname);
    };

    // Load from file.
    template<typename REAL_t>
    void Model<REAL_t>::load(string dirname) {
        utils::load_matrices(parameters(), dirname);
    };
};
