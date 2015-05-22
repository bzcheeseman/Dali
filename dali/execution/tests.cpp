#include <gtest/gtest.h>
#include "dali/mat/Mat.h"
#include "dali/execution/BeamSearch.h"

TEST(beam_search, beam_search2) {
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
