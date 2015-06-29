#ifndef BEAM_SEARCH_MAT_H
#define BEAM_SEARCH_MAT_H

#include <algorithm>
#include <functional>
#include <vector>

#include "dali/tensor/Mat.h"
#include "dali/utils/core_utils.h"

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
                    auto sorted_candidates = scores.argsort();
                    std::reverse(sorted_candidates.begin(), sorted_candidates.end());
                    auto candidates_remaining = beam_width;
                    for(auto& candidate_idx: sorted_candidates) {
                        if (utils::in_vector(forbidden_symbols, (uint)candidate_idx))
                            continue;
                        if (candidates_remaining-- <= 0)
                            break;
                        auto candidate_score = scores.w(candidate_idx);
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
}

#endif
