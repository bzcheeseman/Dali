#include <gtest/gtest.h>
#include <map>
#include <stdexcept>

#include "dali/array/array.h"
#include "dali/array/op/unary.h"
#include "dali/execution/beam_search.h"

using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;

struct AutomataState {
    int state;
    static double transition_prob(int from, int to) {
        if (from == 0) {
            if (to == 0)
                return 0.6;
            if (to == 1)
                return 0.25;
            if (to == 2)
                return 0.15;
        }
        if (from == 1) {
            if (to == 1)
                return 0.5;
            if (to == 2)
                return 0.5;
        }
        return 0.0;
    }

    AutomataState() : state(0) {}
    AutomataState(int j) : state(j) {};
    Array predict() const {
        // automata machine
        // 0 goes to [0, 1, 2], 1 goes to [1, 2], and 2 goes to 2.
        auto new_states = Array::zeros({3}, DTYPE_DOUBLE);
        for (int to = 0; to < 3; to++) {
            new_states(to) = transition_prob(state, to);
        }
        if (state == 2) {
            ASSERT2(false, "Should never reach state == 2.");
        }
        return new_states;
    }
};

typedef AutomataState state_t;

TEST(beam_search, beam_search_automata) {
    // test whether going forward in beam yields right solutions.
    auto prob_next_states = [](state_t state) -> Array {
        auto new_probs = state.predict();
        return op::log(new_probs);
    };
    auto make_choice = [](state_t state, uint candidate) -> state_t {
        return state_t((int)candidate);
    };

    auto initial_state = state_t(0);
    int max_size = 20;

    // Take greedy approach:
    auto beam_search_results = beam_search<state_t, Array>(
        /*initial_state=*/initial_state,
        /*beam_width=*/1,
        /*score_function=*/prob_next_states,
        /*make_choice=*/make_choice,
        /*end_symbol=*/2,
        /*max_beam_length=*/max_size);

    ASSERT_EQ(beam_search_results.size(), 1);
    auto comp = std::vector<uint>(max_size, 0);
    ASSERT_EQ(beam_search_results[0].solution.size(), max_size);
    ASSERT_EQ(beam_search_results[0].solution, comp);

    // Take less greedy approach:
    int beam_width = 7;
    beam_search_results = beam_search_results = beam_search<state_t, Array>(
        /*initial_state=*/initial_state,
        /*beam_width=*/beam_width,
        /*score_function=*/prob_next_states,
        /*make_choice=*/make_choice,
        /*end_symbol=*/2,
        /*max_beam_length=*/max_size);

    ASSERT_EQ(beam_search_results.size(), beam_width);
    ASSERT_EQ(beam_search_results[0].solution, std::vector<uint>{2});
    ASSERT_NEAR(std::exp(beam_search_results[0].score), 0.15, 1e-6);
    ASSERT_EQ(beam_search_results[1].solution, std::vector<uint>({1, 2}));
    ASSERT_NEAR(std::exp(beam_search_results[1].score), 0.25 * 0.5, 1e-6);
    ASSERT_EQ(beam_search_results[2].solution, std::vector<uint>({0, 2}));
    ASSERT_NEAR(std::exp(beam_search_results[2].score), 0.6 * 0.15, 1e-6);
    ASSERT_EQ(beam_search_results.back().solution.size(), max_size);
    ASSERT_NEAR(beam_search_results.back().score, std::log(0.6) * 19 + std::log(0.25), 1e-5);
}

TEST(beam_search, beam_search_score_test) {
    using beam_search_helper::BeamSearchResult;
    using utils::iter_to_str;
    typedef std::string state_t;
    typedef BeamSearchResult<state_t> result_t;
    typedef std::vector<result_t> results_t;

    const int MAX_LENGTH = 2;
    std::map<state_t, double> choices = {
        // initial_choices
        {"a", 0.6},
        {"b", 0.4},
        // after chosing a
        {"aa", 0.55},  // (total worth 0.33)
        {"ab", 0.45},  // (total worth 0.18)
        // after choosing b
        {"ba", 0.99},  // (total worth 0.495)
        {"bb", 0.11},  // (total worth 0.044)
    };

    // Above example is designed to demonstrate greedy solution,
    // as well as better optimal solution:
    // GREEDY:    (beam_width == 1) => "aa" worth 0.33
    // OPTIMAL:   (beam_width == 2) => "ba" worth 0.495
    auto res_aa = result_t("aa", {0,0}, std::log(0.6 * 0.55));
    auto res_ab = result_t("ab", {0,1}, std::log(0.6 * 0.45));
    auto res_ba = result_t("ba", {1,0}, std::log(0.4 * 0.99));
    auto res_bb = result_t("bb", {1,1}, std::log(0.4 * 0.11));

    auto initial_state = "";
    auto candidate_scores = [&](state_t state) -> Array {
        Array ret({2}, DTYPE_DOUBLE);
        ret(0) = std::log(choices.at(state + "a"));
        ret(1) = std::log(choices.at(state + "b"));
        return ret;
    };
    auto make_choice = [&](state_t prev_state, uint choice) -> state_t {
        return prev_state + (choice == 0 ? "a" : "b");
    };

    const uint FAKE_END_SYMBOL = 999;

    auto my_beam_search = [&](int beam_width) -> results_t {
        return beam_search<state_t, Array>(initial_state,
                                           beam_width,
                                           candidate_scores,
                                           make_choice,
                                           FAKE_END_SYMBOL,
                                           MAX_LENGTH);
    };

    auto beam_search_results_equal =
            [](results_t as, results_t bs) -> AssertionResult {
        if (as.size() != bs.size()) {
            return AssertionFailure()
                    << "Results of different sizes: "
                    << as.size() << " != " << bs.size();
        }
        for (int ridx = 0; ridx < as.size(); ++ridx) {
            auto a = as[ridx];
            auto b = bs[ridx];
            if (a.state != b.state)
                return AssertionFailure()
                    << "Result " << ridx + 1 << " has different states: <"
                    << a.state  << "> != <" << b.state << ">";
            if (a.solution != b.solution) {
                auto a_str = iter_to_str(a.solution.begin(),
                                         a.solution.end());
                auto b_str = iter_to_str(b.solution.begin(),
                                         b.solution.end());
                return AssertionFailure()
                    << "Result " << ridx + 1 << " has different solution: <"
                    << a_str << "> != <" << b_str << ">";
            }
            if (std::abs(a.score - b.score) > 1e-9)
                return AssertionFailure()
                    << "Result " << ridx + 1 << " has different score: "
                    << a.score  << " != " << b.score;
        }
        return AssertionSuccess();
    };

    EXPECT_THROW(my_beam_search(0),std::runtime_error);
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(1),
        {res_aa}
    ));
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(2),
        {res_ba, res_aa}
    ));
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(4),
        {res_ba, res_aa, res_ab, res_bb}
    ));
    EXPECT_TRUE(beam_search_results_equal(
        my_beam_search(10),
        {res_ba, res_aa, res_ab, res_bb}
    ));
}
