#include <gtest/gtest.h>
#include <vector>
#include <functional>

#include "dali/utils/observer.h"

TEST(utils, observation) {
    Observation<int> on_increase;

    int a=0, b=0;

    EXPECT_EQ(0, on_increase.notify(4));
    EXPECT_EQ(a, 0);
    EXPECT_EQ(b, 0);

    auto a_handle = on_increase.observe([&](int inc) {
        a += inc;
    });

    EXPECT_EQ(1, on_increase.notify(3));
    EXPECT_EQ(a, 3);
    EXPECT_EQ(b, 0);

    auto b_handle = on_increase.observe([&](int inc) {
        b += inc;
    });

    EXPECT_EQ(2, on_increase.notify(2));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 2);


    on_increase.lose_interest(a_handle);

    EXPECT_EQ(1, on_increase.notify(3));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);

    on_increase.lose_interest(b_handle);

    EXPECT_EQ(0, on_increase.notify(11));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);
}


TEST(utils, observer_guard) {
    Observation<int> on_increase;

    int a=0, b=0;

    EXPECT_EQ(0, on_increase.notify(4));
    EXPECT_EQ(a, 0);
    EXPECT_EQ(b, 0);

    std::vector<ObserverGuard<int>> callback_number_duo;
    {
        auto callback_number_uno = make_observer_guard(
            [&](int inc) {
                a += inc;
            },
            &on_increase
        );

        EXPECT_EQ(1, on_increase.notify(3));
        EXPECT_EQ(a, 3);
        EXPECT_EQ(b, 0);

        callback_number_duo.emplace_back(
            make_observer_guard([&](int inc) {
                b += inc;
            },
            &on_increase
        ));

        EXPECT_EQ(2, on_increase.notify(2));
        EXPECT_EQ(a, 5);
        EXPECT_EQ(b, 2);
    }
    // callback_number_uno deallocated

    EXPECT_EQ(1, on_increase.notify(3));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);

    callback_number_duo.clear();

    EXPECT_EQ(0, on_increase.notify(11));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);
}
