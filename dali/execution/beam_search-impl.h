#include <algorithm>
#include <functional>

#include "dali/utils/assert2.h"
// #include "dali/array/op.h"

namespace beam_search_helper {
    template<typename state_t>
    BeamSearchResult<state_t>::BeamSearchResult() {}

    template<typename state_t>
    BeamSearchResult<state_t>::BeamSearchResult(const state_t& state_,
                                       const std::vector<uint>& solution_,
                                       const double& score_) :
            state(state_), solution(solution_), score(score_) {
    }

    template<typename state_t>
    struct BeamSearchProposal {
        typedef BeamSearchResult<state_t> result_t;

        result_t prev_result;
        double score;
        uint candidate_idx;
        bool finalized;

        BeamSearchProposal() {}

        BeamSearchProposal(const result_t& prev_result_,
                           const double& score_,
                           const uint& candidate_idx_,
                           const bool& finalized_) :
                prev_result(prev_result_),
                score(score_),
                candidate_idx(candidate_idx_),
                finalized(finalized_) {
        }

        static BeamSearchProposal finalized_solution(const result_t& res) {
            return BeamSearchProposal<state_t>(res, res.score, 0, true);
        }

        static BeamSearchProposal solution_candidate(const result_t& prev_result_,
                                                     const double& updated_score_,
                                                     const uint& candidate_idx_) {
            return BeamSearchProposal<state_t>(
                prev_result_,
                updated_score_,
                candidate_idx_,
                false
            );
        }
    };
}

template<typename state_t, typename Container>
std::vector<beam_search_helper::BeamSearchResult<state_t>> beam_search(state_t initial_state,
            uint beam_width,
            std::function<Container(state_t)> candidate_scores,
            std::function<state_t(state_t, uint)> make_choice,
            uint end_symbol,
            int max_solution_length,
            std::vector<uint> forbidden_symbols) {
    ASSERT2(beam_width > 0,
        utils::MS() << "Beam width must be strictly positive (got beam_width = "
                    << beam_width << ")."
    );
    typedef beam_search_helper::BeamSearchResult<state_t> result_t;
    typedef beam_search_helper::BeamSearchProposal<state_t> proposal_t;

    std::vector<result_t> results = {
        result_t(initial_state, std::vector<uint>(), 0.0)
    };

    while (max_solution_length--) {
        std::vector<proposal_t> proposals;
        for (auto& result: results) {
            if (result.solution.size() > 0 && result.solution.back() == end_symbol) {
                proposals.push_back(proposal_t::finalized_solution(result));
            } else {
                auto scores = candidate_scores(result.state);
                ASSERT2(scores.ndim() == 1,
                    utils::MS() << "score function must return a vector (got scores.ndim() = "
                                << scores.ndim() << ")."
                );
                Container sorted_candidates = scores.argsort();
                sorted_candidates = sorted_candidates[Slice(0, sorted_candidates.shape()[0], -1)];
                auto candidates_remaining = beam_width;

                for(int candidate_idx = 0; candidate_idx < sorted_candidates.shape()[0]; candidate_idx++) {
                    int candidate_symbol = sorted_candidates(candidate_idx);
                    bool found = false;
                    for (const auto& val : forbidden_symbols) {
                        if (val == candidate_symbol) {
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                    if (candidates_remaining-- <= 0)
                        break;
                    double candidate_score = scores(candidate_symbol);
                    proposals.push_back(
                        proposal_t::solution_candidate(
                            result,
                            candidate_score + result.score,
                            candidate_symbol
                        )
                    );
                }
            }
        }
        // proposals due
        std::sort(proposals.begin(), proposals.end(), [](const proposal_t& a, const proposal_t& b) {
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
