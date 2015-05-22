#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "dali/mat/Mat.h"
#include "dali/execution/BeamSearch.h"

using std::vector;
using std::tuple;
using std::make_tuple;


TEST(beam_search, beam_search2_jonathan) {
    // int target = 8;
    // auto functor = [&target](vector<Mat<R>> Xs)-> Mat<R> {
    //     auto soft = MatOps<R>::softmax(
    //             Xs[1].dot(Xs[0])
    //         );
    //     return MatOps<R>::cross_entropy(
    //         soft,
    //         target);
    // };
    // EXPERIMENT_REPEAT {
    //     auto input = Mat<R>(5,  3, weights<R>::uniform(-2.0, 2.0));
    //     auto layer = Mat<R>(10, 5, weights<R>::uniform(-2.0, 2.0));
    //     ASSERT_TRUE(gradient_same<R>(functor, {input, layer}, 1e-4));
    // }
    ASSERT_EQ(1 + 1, 2);
}


TEST(beam_search, beam_search2_szymon) {
    vector<tuple<string, vector<int>> choices = {
        // initial_choices
        make_tuple("a", 0.6)
        make_tuple("b", 0.4)
        // after chosing a
        make_tuple("aa", 0.55)
        make_tuple("ab", 0.45)
        // after choosing b
        make_tuple("ba", 0.99);
        make_tuple("bb", 0.11);
    }
    // Above example is designed to demonstrate greedy solution,
    // as well as better optimal solution:
    // GREEDY:    (beam_width == 1) => "aa" worth 0.33
    // OPTIMAL:   (beam_width == 2) => "ba" worth 0.495


    ASSERT_EQ(1 + 1, 2);
}
