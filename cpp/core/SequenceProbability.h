#ifndef SEQUENCE_PROBABILITY_MAT_H
#define SEQUENCE_PROBABILITY_MAT_H

#include <vector>
#include "core/Graph.h"
#include "core/utils.h"

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
    template<typename model_t, typename K, typename C>
    std::shared_ptr<Mat<typename model_t::value_t>> sequence_probabilities(
        const model_t& model,
        K& data,
        C& codelens) {
        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;

        Graph<typename model_t::value_t> G(false);

        auto initial_state      = model.initial_states();
        auto out_state_and_prob = model.activate(G, initial_state, data.col(0));

        std::shared_ptr<Mat<typename model_t::value_t>> log_prob = std::make_shared<Mat<typename model_t::value_t>>(data.rows(), data.cols());

        int n = data.cols();

        auto data_pluck = Indexing::Index::arange(0, data.rows());

        for (int i = 1; i < n-1; i++) {
            out_state_and_prob = model.activate(G, out_state_and_prob.first, data.col(i));
            auto plucked_activations = G.rows_cols_pluck(out_state_and_prob.second, data.col(i+1), data_pluck);
            log_prob->w.col(i).array() += plucked_activations->w.row(0).transpose().array().log() + log_prob->w.col(i-1).array();
        }

        auto relevant_log_prob = G.rows_cols_pluck(log_prob, data_pluck, codelens);

        return relevant_log_prob;
    }
}

#endif
