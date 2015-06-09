#ifndef BEAM_SEARCH_MAT_H
#define BEAM_SEARCH_MAT_H

#include <algorithm>
#include <functional>
#include <vector>
#include <dali/mat/Mat.h>
#include <dali/utils/core_utils.h>

namespace beam_search {
    template<typename REAL_t,typename state_t>
    struct BeamSearchResult {
        state_t state;
        std::vector<uint> solution;
        REAL_t score;

        BeamSearchResult() {}

        BeamSearchResult(state_t _state,
                         std::vector<uint> _solution,
                         REAL_t _score) :
                state(_state),
                solution(_solution),
                score(_score) {
        }
    };

    template<typename REAL_t, typename state_t>
    struct BeamSearchProposal {
        typedef BeamSearchResult<REAL_t,state_t> result_t;

        result_t prev_result;
        REAL_t score;
        uint candidate_idx;
        bool finalized;

        BeamSearchProposal() {}

        BeamSearchProposal(result_t _prev_result,
                         REAL_t _score,
                         uint _candidate_idx,
                         bool _finalized) :
                prev_result(_prev_result),
                score(_score),
                candidate_idx(_candidate_idx),
                finalized(_finalized) {
        }

        static BeamSearchProposal finalized_solution(result_t _result) {
            return BeamSearchProposal<REAL_t,state_t>(
                    _result, _result.score, 0, true);
        }

        static BeamSearchProposal solution_candidate(result_t _prev_result,
                                                     REAL_t _updated_score,
                                                     uint _candidate_idx) {
            return BeamSearchProposal<REAL_t,state_t>(
                    _prev_result, _updated_score, _candidate_idx, false);
        }


    };


    // attempts to find maximum sum of scores candidate.
    template<typename REAL_t, typename state_t>
    std::vector<BeamSearchResult<REAL_t,state_t>>
    beam_search2(state_t initial_state,
                 uint beam_width,
                 std::function<Mat<REAL_t>(state_t)> candidate_scores,
                 std::function<state_t(state_t, uint)> make_choice,
                 uint end_symbol,
                 int max_solution_length,
                 std::vector<uint> forbidden_symbols=std::vector<uint>()) {
        utils::assert2(beam_width > 0, "Beam width must be strictly positive.");
        typedef BeamSearchResult<REAL_t, state_t> result_t;
        typedef BeamSearchProposal<REAL_t, state_t> proposal_t;

        std::vector<result_t> results = {
            result_t(initial_state, std::vector<uint>(), (REAL_t)0.0)
        };

        while (max_solution_length--) {
            std::vector<proposal_t> proposals;
            for (auto& result: results) {
                if (result.solution.size() > 0 && result.solution.back() == end_symbol) {
                    proposals.push_back(proposal_t::finalized_solution(result));
                } else {
                    auto scores = candidate_scores(result.state);
                    auto sorted_candidates = utils::argsort_rowwise(scores);
                    std::reverse(sorted_candidates.begin(), sorted_candidates.end());
                    auto candidates_remaining = beam_width;
                    for(auto& candidate_idx: sorted_candidates) {
                        if (utils::in_vector(forbidden_symbols, (uint)candidate_idx))
                            continue;
                        if (candidates_remaining-- <= 0)
                            break;
                        auto candidate_score = (*scores.w())(candidate_idx);
                        proposals.push_back(proposal_t::solution_candidate(
                                result, result.score + candidate_score, candidate_idx));
                    }
                }
            }
            // proposals due
            sort(proposals.begin(), proposals.end(), [](proposal_t a, proposal_t b) {
                return a.score > b.score;
            });
            proposals.resize(std::min((size_t)beam_width, proposals.size()));

            results = std::vector<result_t>();
            for(auto& proposal : proposals) {
                if (proposal.finalized) {
                    results.push_back(proposal.prev_result);
                } else {
                    auto new_state = make_choice(proposal.prev_result.state, proposal.candidate_idx);
                    auto new_solution = proposal.prev_result.solution;
                    new_solution.emplace_back(proposal.candidate_idx);
                    results.emplace_back(new_state, new_solution, proposal.score);
                }
            }
        }
        return results;
    }


    template<typename T>
    class ProbabilityPair {
        public:
            const uint symbol;
            const T prob;
            ProbabilityPair(uint _symbol, T _prob) : symbol(_symbol), prob(_prob) {}
    };

    template<typename model_t, typename T>
    std::pair<typename model_t::state_type, std::vector<ProbabilityPair<T>>> beam_search_with_indices(
        const model_t& model,
        typename model_t::state_type& previous_state,
        uint index,
        int k,
        T log_prob,
        uint ignore_symbol = -1) {

        auto out_state_and_prob = model.activate(previous_state, index);
        std::pair<typename model_t::state_type, std::vector<ProbabilityPair<T>>> out;
        out.first = std::get<0>(out_state_and_prob);
        std::vector<T> probabilities(
            std::get<1>(out_state_and_prob).w().data(),
            std::get<1>(out_state_and_prob).w().data() + std::get<1>(out_state_and_prob).dims(0));
        auto sorted_probs = utils::argsort(probabilities);

        // we pass along the new state, and the "winning" k predictions
        // weighed by the conditional probability `log_prob` passed to the function.
        auto sorted_probs_rbegin = sorted_probs.rbegin();

        while (out.second.size() < k) {
            if (*sorted_probs_rbegin != ignore_symbol) {
                out.second.emplace_back(*sorted_probs_rbegin, std::log(probabilities[*sorted_probs_rbegin]) + log_prob);
            }
            sorted_probs_rbegin++;
        }
        return out;
    }

    inline eigen_index_vector convert_to_eigen_vector(const std::initializer_list<uint>& list) {
        eigen_index_vector vec(list.size());
        auto ptr = vec.data();
        for (auto& i : list) {
            (*(ptr++)) = i;
        }
        return vec;
    }

    inline eigen_index_vector convert_to_eigen_vector(const std::vector<uint>& list) {
        eigen_index_vector vec(list.size());
        auto ptr = vec.data();
        for (auto& i : list) {
            (*(ptr++)) = i;
        }
        return vec;
    }

    inline eigen_index_vector convert_to_eigen_vector(const eigen_index_vector& list) {
        return list;
    }

    inline eigen_index_block_scalar convert_to_eigen_vector(const eigen_index_block_scalar& list) {
        return list;
    }

    inline eigen_index_block convert_to_eigen_vector(const eigen_index_block& list) {
        return list;
    }

    template<typename model_t, typename K>
    std::vector<std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type >> beam_search(
        const model_t& model,
        K example,
        int max_steps,
        int symbol_offset,
        int k,
        uint end_symbol,
        uint ignore_symbol = -1) {

        auto ex = convert_to_eigen_vector(example);

        typedef std::vector<uint> seq_type;
        typedef std::tuple<std::vector<uint>, typename model_t::value_t, typename model_t::state_type > open_list_t;


        graph::NoBackprop nb;

        int n = ex.cols() * ex.rows();
        auto initial_state = model.get_final_activation(ex.head(n - 1));
        // we start off with k different options:
        std::vector<open_list_t> open_list;
        {
            auto out_beam = beam_search_with_indices(model, initial_state, ex(n-1), k, 0.0, ignore_symbol);
            for (auto& candidate : out_beam.second) {
                open_list.emplace_back(
                    open_list_t(
                        std::initializer_list<uint>({candidate.symbol + symbol_offset}),// the new fork
                        candidate.prob,                 // the new probabilities
                        out_beam.first                  // the new state
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
            for (auto& fork : options) {
                if (std::get<0>(fork).back() == end_symbol) {
                    // if this path says to stop,
                    // add it back to the open list
                    // and carry on.
                    // if stops == number of k
                    // then we are done
                    stops += 1;
                    open_list.emplace_back(fork);
                } else {
                    // if fork is not asking to
                    // end the sequence, then:
                    auto forks_beam = beam_search_with_indices(model,
                        std::get<2>(fork),            // the internal state going forward
                        std::get<0>(fork).back(),     // the direction to take
                        k,                            // size of the beam
                        std::get<1>(fork),            // the conditional probability for this
                                                      // fork in the path
                        ignore_symbol                 // don't include paths with this direction
                                                      // in the open-list.
                        );
                    for (auto& forks_fork : std::get<1>(forks_beam)) {
                        seq_type new_seq(std::get<0>(fork));
                        new_seq.emplace_back(forks_fork.symbol + symbol_offset);
                        open_list.emplace_back(
                            open_list_t(
                                new_seq,          // the new candidate state
                                forks_fork.prob,  // the new probabilities
                                std::get<0>(forks_beam)    // the new state
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
            i+=1;
            open_list.resize(k);
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
