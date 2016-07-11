#include <gtest/gtest.h>
#include <vector>

#include "dali/utils/scoring_utils.h"

TEST(utils, pearson_correlation) {
    std::vector<double> x = {43, 21, 25, 42, 57, 59};
    std::vector<double> y = {99, 65, 79, 75, 87, 81};

    auto corr = utils::pearson_correlation(x,y);

    ASSERT_NEAR(corr, 0.5298, 1e-5);
}
