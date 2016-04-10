#include <gtest/gtest.h>
#include <map>
#include <stdexcept>

#include "dali/tensor/Mat.h"
#include "dali/tensor/MatOps.h"
#include "dali/execution/BeamSearch.h"
#include "dali/execution/SequenceProbability.h"

using std::make_tuple;
using std::map;
using std::string;
using std::tuple;
using std::vector;
using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;

typedef float R;

template<typename T>
std::string iter_to_str(T begin, T end) {
    std::stringstream ss;
    bool first = true;
    for (; begin != end; begin++)
    {
        if (!first)
            ss << ", ";
        ss << *begin;
        first = false;
    }
    return ss.str();
}

struct AutomataState {
    typedef float REAL_t;

    int state;

    static REAL_t transition_prob(int from, int to) {
        if (from == 0) {
            if (to == 0)
                return 0.5;
            if (to == 1)
                return 0.3;
            if (to == 2)
                return 0.2;
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
    Mat<REAL_t> predict() const {
        // automata machine
        // 0 goes to [0, 1, 2], 1 goes to [1, 2], and 2 goes to 2.
        auto new_states = Mat<REAL_t>(3, 1);
        for (int to = 0; to < 3; to++) {
            new_states.w(to) = transition_prob(state, to);
        }
        if (state == 2) {
            ASSERT2(false, "Should never reach this state");
        }
        return new_states;
    }
};

typedef AutomataState state_t;

TEST(beam_search, beam_search_automata) {
    typedef float REAL_t;

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
    auto beam_search_results = beam_search::beam_search<REAL_t, state_t>(
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
    beam_search_results = beam_search::beam_search<REAL_t, state_t>(
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

TEST(beam_search, beam_search_score_test) {
    using beam_search::beam_search;
    using beam_search::BeamSearchResult;
    using utils::iter_to_str;
    typedef double REAL_t;
    typedef string state_t;
    typedef BeamSearchResult<REAL_t,state_t> result_t;
    typedef vector<result_t> results_t;

    const int MAX_LENGTH = 2;
    map<state_t, REAL_t> choices = {
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
    auto candidate_scores = [&](state_t state) -> Mat<REAL_t> {
        Mat<REAL_t> ret(2,1);
        ret.w(0,0) = std::log(choices.at(state + "a"));
        ret.w(1,0) = std::log(choices.at(state + "b"));

        return ret;
    };
    auto make_choice =
            [&](state_t prev_state, uint choice) -> state_t {
        return prev_state + (choice == 0 ? "a" : "b");
    };

    const uint FAKE_END_SYMBOL = 999;

    auto my_beam_search = [&](int beam_width) -> results_t {
        return beam_search<REAL_t,state_t>(initial_state,
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


TEST(sequence_probability, score) {
    Batch<R> batch;
    int seq_length = 5;
    batch.data = Mat<int>(seq_length, 1);
    R expected_prob = 0.0;
    int current_state = 0;
    auto initial_state = AutomataState(current_state);

    {
        // generate data and obtain
        // transition probability for sequence
        // 0 0 1 1 2
        int i = 1;
        for (int state : {0, 1, 1, 2}) {
            // store the transition probabilities
            // by summing the log likelihood of
            // the transition
            expected_prob += std::log(
                AutomataState::transition_prob(
                    current_state,
                    state
                )
            );
            // store state at the next timestep
            batch.data.w(i++) = state;
            // enter the new state
            current_state = state;
        }
    }
    batch.target = batch.data;
    batch.code_lengths.emplace_back(seq_length);

    // test whether going forward in beam yields right solutions.
    auto decode = [](state_t state) -> Mat<R> {
        auto new_probs = state.predict();
        return new_probs.log();
    };
    auto observe = [](Mat<int> candidate, state_t state) -> state_t {
        return state_t((int)candidate.w(0));
    };

    auto scores = sequence_probability::sequence_score<R, state_t>(
        batch,
        initial_state, // start out with state = 0
        decode,
        observe,
        1 // look 1 step ahead
    );

    ASSERT_EQ(scores.w(0), expected_prob);
}
