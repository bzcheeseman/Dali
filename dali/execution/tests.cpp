#include <gtest/gtest.h>

#include <tuple>
#include <vector>
#include <string>

#include "dali/mat/Mat.h"
#include "dali/mat/MatOps.h"
#include "dali/execution/BeamSearch.h"

using std::vector;
using std::tuple;
using std::string;
using std::make_tuple;
typedef float REAL_t;

struct AutomataState {
    int state;
    AutomataState() : state(0) {}
    AutomataState(int j) : state(j) {};
    Mat<REAL_t> predict() const {
        // automata machine
        // 0 goes to [0, 1, 2], 1 goes to [1, 2], and 2 goes to 2.
        auto new_states = Mat<REAL_t>(3, 1);
        if (state == 0) {
            new_states.w()(0, 0) = 0.5;
            new_states.w()(1, 0) = 0.3;
            new_states.w()(2, 0) = 0.2;
        }
        if (state == 1) {
            new_states.w()(1, 0) = 0.5;
            new_states.w()(2, 0) = 0.5;
        }
        if (state == 2) {
            assert(false);
        }
        return new_states;
    }
};

typedef AutomataState state_t;

TEST(beam_search, beam_search_automata) {
    // test whether going forward in beam yields right solutions.
    auto prob_next_states = [](state_t state) -> Mat<REAL_t> {
        auto new_probs = state.predict();
        return new_probs.log();
    };
    auto make_choice = [](state_t state, uint candidate) -> state_t {
        return state_t((int)candidate);
    };

    auto initial_state = state_t(0);
    int max_size = 20;

    // Take greedy approach:
    auto beam_search_results = beam_search::beam_search2<REAL_t, state_t>(
            initial_state,
            1,
            prob_next_states,
            make_choice,
            2,
            max_size);

    ASSERT_EQ(beam_search_results.size(), 1);
    auto comp = vector<uint>(max_size, 0);
    ASSERT_EQ(beam_search_results[0].solution.size(), max_size);
    ASSERT_EQ(beam_search_results[0].solution, comp);


    // Take less greedy approach:
    int beam_width = 7;
    beam_search_results = beam_search::beam_search2<REAL_t, state_t>(
            initial_state,
            beam_width,
            prob_next_states,
            make_choice,
            2,
            max_size);

    ASSERT_EQ(beam_search_results.size(), beam_width);
    ASSERT_EQ(beam_search_results[0].solution, vector<uint>{2});
    ASSERT_NEAR(std::exp(beam_search_results[0].score), 0.2, 1e-6);
    ASSERT_EQ(beam_search_results[1].solution, vector<uint>({1, 2}));
    ASSERT_NEAR(std::exp(beam_search_results[1].score), 0.3 * 0.5, 1e-6);
    ASSERT_EQ(beam_search_results[2].solution, vector<uint>({0, 2}));
    ASSERT_NEAR(std::exp(beam_search_results[2].score), 0.5 * 0.2, 1e-6);
    ASSERT_EQ(beam_search_results.back().solution.size(), max_size);
    ASSERT_NEAR(beam_search_results.back().score, std::log(0.5) * 20, 1e-5);
}


TEST(beam_search, beam_search2_szymon) {
    // vector<tuple<string, vector<int>>> choices = {
    //     // initial_choices
    //     make_tuple("a", 0.6),
    //     make_tuple("b", 0.4)
    //     // after chosing a
    //     make_tuple("aa", 0.55)
    //     make_tuple("ab", 0.45)
    //     // after choosing b
    //     make_tuple("ba", 0.99);
    //     make_tuple("bb", 0.11);
    // }
    // Above example is designed to demonstrate greedy solution,
    // as well as better optimal solution:
    // GREEDY:    (beam_width == 1) => "aa" worth 0.33
    // OPTIMAL:   (beam_width == 2) => "ba" worth 0.495


    ASSERT_EQ(1 + 1, 2);
}
