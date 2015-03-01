#ifndef SEQUENCE_PROBABILITY_MAT_H
#define SEQUENCE_PROBABILITY_MAT_H

#include <vector>
#include "core/Graph.h"

namespace sequence_probability {

    eigen_index_vector convert_to_eigen_vector(const std::initializer_list<uint>& list) {
        eigen_index_vector vec(list.size());
        auto ptr = vec.data();
        for (auto& i : list) {
            (*(ptr++)) = i;
        }
        return vec;
    }

    eigen_index_vector convert_to_eigen_vector(const eigen_index_vector& list) {
        return list;
    }

    eigen_index_block_scalar convert_to_eigen_vector(const eigen_index_block_scalar& list) {
        return list;
    }

    eigen_index_block convert_to_eigen_vector(const eigen_index_block& list) {
        return list;
    }

    template<typename model_t, typename K>
    typename model_t::value_t sequence_probability(
        const model_t& model,
        K example) {

        auto ex = convert_to_eigen_vector(example);
        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;

        Graph<typename model_t::value_t> G(false);
        int n = ex.cols() * ex.rows();

        auto initial_state = model.initial_states();
        auto out_state_and_prob = model.activate(G, initial_state, example(0));

        typename model_t::value_t log_prob = 0.0;

        for (int i = 1; i < n; i++) {
            out_state_and_prob = model.activate(G, out_state_and_prob.first, example(i));
            log_prob += std::log(out_state_and_prob.second->w(example(i))) + log_prob;
        }
        return log_prob;
    }

    // TODO: generalize this to multiple input sequences and output the probability of each example at
    // their end points, a vector of model_t::value_t
    // template<typename model_t, typename K>
    // typename model_t::value_t sequence_probabilities(
    //     const model_t& model,
    //     const K& data) {
    //     typedef std::vector<uint> seq_type;
    //     typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;

    //     Graph<typename model_t::value_t> G(false);
    //     int n = ex.cols() * ex.rows();

    //     auto initial_state = model.initial_states();
    //     auto out_state_and_prob = model.activate(G, initial_state, example(0));

    //     typename model_t::value_t log_prob = 0.0;

    //     for (int i = 1; i < n; i++) {
    //         out_state_and_prob = model.activate(G, out_state_and_prob.first, example(i));
    //         log_prob += std::log(out_state_and_prob.second->w(example(i))) + log_prob;
    //     }
    //     return log_prob;
    // }
}

#endif
