#ifndef BEAM_SEARCH_MAT_H
#define BEAM_SEARCH_MAT_H

#include <vector>
#include "core/Graph.h"
#include <algorithm>

namespace beam_search {

    template<typename graph_t, typename model_t, typename T>
    std::pair<typename model_t::state_type, std::vector<std::pair<uint, T>>> beam_search_with_indices(
    	const model_t& model,
        graph_t& G,
        typename model_t::state_type& previous_state,
        uint index,
        int k,
        T prob) {

        auto out_state_and_prob = model.activate(G, previous_state, index);
        std::pair<typename model_t::state_type, std::vector<std::pair<uint,T>>> out;
        out.first = out_state_and_prob.first;
        std::vector<T> probabilities(out_state_and_prob.second->w.data(), out_state_and_prob.second->w.data() + out_state_and_prob.second->d);
        auto sorted_probs = utils::argsort(probabilities);

        // we pass along the new state, and the "winning" k predictions
        // weighed by the conditional probability `prob` passed to the function.
        for (int i = 0;i < k;i++)
            out.second.emplace_back(sorted_probs[i], probabilities[sorted_probs[i]] * prob);
        return out;
    }

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
    std::vector<std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type >> beam_search(
    	const model_t& model,
        K example,
        int max_steps,
        int symbol_offset,
        int k,
        uint end_symbol) {

        auto ex = convert_to_eigen_vector(example);


        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;

        Graph<typename model_t::value_t> G(false);
        int n = ex.cols() * ex.rows();
        auto initial_state = model.get_final_activation(G, ex.head(n - 1));

        // we start off with k different options:
        std::vector<open_list_t> open_list;
        {
            auto out_beam = beam_search_with_indices(model, G, initial_state, ex(n-1), k, 1.0);

            for (auto& candidate : out_beam.second) {
                open_list.emplace_back(
                    open_list_t(
                        {candidate.first + symbol_offset},// the new fork
                        candidate.second,                 // the new probabilities
                        out_beam.first                    // the new state
                    )
                );
            }
        }
        // for each fork in the path we expand another k
        // options forward
        int i = 0;
        while (true) {
            int stops = 0;
            decltype(open_list) options(open_list);
            open_list.clear();
            for (auto& candidate : options) {
                if (std::get<0>(candidate).back() == end_symbol) {
                    // if this path says to stop,
                    // add it back to the open list
                    // and carry on.
                    // if stops == number of k
                    // then we are done
                    stops += 1;
                    open_list.emplace_back(candidate);
                } else {
                    // if candidate is not asking to
                    // end the sequence, then:
                    auto out_beam = beam_search_with_indices(model, G,
                        std::get<2>(candidate),        // the internal state going forward
                        std::get<0>(candidate).back(), // the direction to take
                        k,                             // size of the beam
                        std::get<1>(candidate));       // the conditional probability for this
                                                       // fork in the path
                    for (auto& candidate : out_beam.second) {
                        seq_type new_seq(std::get<0>(candidate));
                        new_seq.emplace_back(candidate.first + symbol_offset);
                        open_list.emplace_back(
                            open_list_t(
                                new_seq,                // the new candidate state
                                std::get<1>(candidate), // the new probabilities
                                out_beam.first          // the new state
                            )
                        );
                    }
                }
            }
            // now that we've evaluated all the possible
            // forks, we prune the least likely ones
            // and keep the top k: that's our beam.
            std::sort(open_list.begin(), open_list.end(),
               [](open_list_t& A, open_list_t& B) {
                    return std::get<1>(A) > std::get<1>(B);
                });
            open_list.resize(k);
            i+=1;
            // if the search takes too long
            // or k paths have reached an endpoint
            // then exit the search
            if (i == max_steps || stops == k)
                break;
        }
        return open_list;
    }

}

#endif