#ifndef SEQUENCE_PROBABILITY_MAT_H
#define SEQUENCE_PROBABILITY_MAT_H

#include <vector>
#include "dali/tensor/__MatMacros__.h"

namespace sequence_probability {

    inline eigen_index_vector convert_to_eigen_vector(const std::initializer_list<uint>& list) {
        eigen_index_vector vec(list.size());
        auto ptr = vec.data();
        for (auto& i : list) {
            (*(ptr++)) = i;
        }
        return vec;
    }

    inline const eigen_index_vector& convert_to_eigen_vector(const eigen_index_vector& list) {
        return list;
    }

    inline const eigen_index_block_scalar& convert_to_eigen_vector(const eigen_index_block_scalar& list) {
        return list;
    }

    inline const eigen_index_block& convert_to_eigen_vector(const eigen_index_block& list) {
        return list;
    }

    template<typename model_t, typename K>
    typename model_t::value_t sequence_probability(
        const model_t& model,
        K example) {

        auto ex = convert_to_eigen_vector(example);
        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;

        graph::NoBackprop nb;
        int n = ex.cols() * ex.rows();

        auto initial_state      = model.initial_states();
        auto out_state_and_prob = model.activate(initial_state, example(0));

        typename model_t::value_t log_prob = 0.0;

        for (int i = 1; i < n; i++) {
            out_state_and_prob = model.activate(std::get<0>(out_state_and_prob), example(i));
            log_prob += std::log(out_state_and_prob.second->w(example(i))) + log_prob;
        }
        return log_prob;
    }

    // TODO: generalize this to multiple input sequences and output the probability of each example at
    // their end points, a vector of model_t::value_t
    template<typename model_t, typename K, typename C>
    Mat<typename model_t::value_t> sequence_probabilities(
        const model_t& model,
        K& data,
        C& codelens) {
        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;
        typedef Mat<typename model_t::value_t> mat;

        graph::NoBackprop nb;

        auto initial_state      = model.initial_states();
        auto out_state_and_prob = model.activate(initial_state, data.col(0));
        auto log_prob           = mat(data.rows(), data.cols()-1);

        int n = data.cols();

        auto data_pluck = Indexing::Index::arange(0, data.rows());

        GET_MAT(log_prob).col(0) = GET_MAT(std::get<1>(out_state_and_prob)(
            data.col(1),
            data_pluck
        )).row(0).transpose().array().log();

        for (int i = 1; i < n-1; i++) {
            out_state_and_prob = model.activate(
                std::get<0>(out_state_and_prob),
                data.col(i)
            );
            auto plucked_activations = std::get<1>(out_state_and_prob)(
                data.col(i+1),
                data_pluck
            );
            GET_MAT(log_prob).col(i).array() += GET_MAT(plucked_activations).row(0).transpose().array().log() + GET_MAT(log_prob).col(i-1).array();
        }

        // inelegant manner of dealing with scalar corrections on codelens:
        auto relevant_log_prob = log_prob(
            data_pluck,
            (codelens.array() - (uint)1).matrix()
        );

        return relevant_log_prob;
    }

    #define LOG1P(X) (((X) + 1).log())
    #define SURPRISE(X) -(LOG1P(-(1 - (X).array()).sqrt()) - LOG1P((1 - (X).array()).sqrt()))

    template<typename model_t, typename K, typename C>
    Mat<typename model_t::value_t> sequence_surprises(
        const model_t& model,
        K& data,
        C& codelens) {
        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;
        typedef Mat<typename model_t::value_t> mat;

        graph::NoBackprop nb;

        auto initial_state      = model.initial_states();
        auto out_state_and_prob = model.activate(initial_state, data.col(0));
        auto log_prob           = mat(data.rows(), data.cols()-1);

        int n = data.cols();

        auto data_pluck = Indexing::Index::arange(0, data.rows());

        GET_MAT(log_prob).col(0) = SURPRISE( GET_MAT(std::get<1>(out_state_and_prob)(data.col(1), data_pluck)).row(0).transpose() );

        for (int i = 1; i < n-1; i++) {
            out_state_and_prob = model.activate(
                std::get<0>(out_state_and_prob),
                data.col(i)
            );
            auto plucked_activations = std::get<1>(out_state_and_prob)(
                data.col(i+1),
                data_pluck
            );
            GET_MAT(log_prob).col(i).array() += SURPRISE(GET_MAT(plucked_activations).row(0).transpose().array()) + GET_MAT(log_prob).col(i-1).array();
        }

        // inelegant manner of dealing with scalar corrections on codelens:
        auto relevant_log_prob = log_prob(
            data_pluck,
            (codelens.array() - (uint)1).matrix()
        );

        return relevant_log_prob;
    }
}

#endif
