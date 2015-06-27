#include "dali/utils/random.h"
using std::vector;

namespace utils {
    int randint(int lower, int upper) {
        assert2 (lower <= upper, "Lower bound must be smaller than upper bound.");
        std::uniform_int_distribution<int> distribution(lower, upper);
        return distribution(utils::random::generator());
    }

    double randdouble(double lower, double upper) {
        assert2 (lower <= upper, "Lower bound must be smaller than upper bound.");
        std::uniform_real_distribution<double> distribution(lower, upper);
        return distribution(utils::random::generator());
    }

    vector<size_t> random_arange(size_t size) {
        vector<size_t> indices(size);
        for (size_t i=0; i < size;i++) indices[i] = i;
        std::srand(randint(0, 9999999999));
        std::random_shuffle( indices.begin(), indices.end());
        return indices;
    }

    vector<vector<size_t>> random_minibatches(size_t total_elements, size_t minibatch_size) {
        vector<size_t> training_order = utils::random_arange(total_elements);
        int num_minibatches = training_order.size() / minibatch_size;
        vector<vector<size_t>> minibatches(num_minibatches);
        for (int tidx = 0; tidx < total_elements; ++tidx) {
            minibatches[tidx%num_minibatches].push_back(training_order[tidx]);
        }
        return minibatches;
    }

    namespace random {
        bool stochastic_seed = true;
        int random_seed      = std::random_device()();

        std::mt19937 generator_ = std::mt19937(random_seed);

        void reseed() {
            // replace random seed with new seed
            // on each call
            std::random_device rd;
            auto random_seed = (int)rd();
            set_seed(random_seed);
        }
        void set_seed(int new_seed) {
            random_seed = new_seed;
            generator_ = std::mt19937(new_seed);
        }

        std::mt19937& generator() {
            return generator_;
        }
    }
}
