#ifndef DALI_ARRAY_EXECUTION_BEAM_SEARCH_H
#define DALI_ARRAY_EXECUTION_BEAM_SEARCH_H

#include <vector>

namespace beam_search_helper {
    template<typename state_t>
    struct BeamSearchResult {
        state_t state;
        std::vector<uint> solution;
        double score;

        BeamSearchResult();
        BeamSearchResult(const state_t& state_,
                         const std::vector<uint>& solution_,
                         const double& score_);
    };
}

// breadth-first search that keeps k best solutions at every step of search.
template<typename state_t, typename Container>
std::vector<beam_search_helper::BeamSearchResult<state_t>>
beam_search(state_t initial_state,
            uint beam_width,
            std::function<Container(state_t)> candidate_scores,
            std::function<state_t(state_t, uint)> make_choice,
            uint end_symbol,
            int max_solution_length,
            std::vector<uint> forbidden_symbols=std::vector<uint>());

#include "dali/execution/beam_search-impl.h"

#endif
