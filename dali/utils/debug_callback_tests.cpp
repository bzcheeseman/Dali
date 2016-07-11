#include <gtest/gtest.h>
#include <vector>
#include <functional>

#include "dali/utils/debug_callback.h"

TEST(utils, debug_callback) {
    DebugCallback<int> on_increase;

    int a=0, b=0;

    EXPECT_EQ(0, on_increase.activate(4));
    EXPECT_EQ(a, 0);
    EXPECT_EQ(b, 0);

    auto a_handle = on_increase.register_callback([&](int inc) {
        a += inc;
    });

    EXPECT_EQ(1, on_increase.activate(3));
    EXPECT_EQ(a, 3);
    EXPECT_EQ(b, 0);

    auto b_handle = on_increase.register_callback([&](int inc) {
        b += inc;
    });

    EXPECT_EQ(2, on_increase.activate(2));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 2);


    on_increase.deregister_callback(a_handle);

    EXPECT_EQ(1, on_increase.activate(3));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);

    on_increase.deregister_callback(b_handle);

    EXPECT_EQ(0, on_increase.activate(11));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);
}


TEST(utils, scoped_debug_callback) {
    DebugCallback<int> on_increase;

    int a=0, b=0;

    EXPECT_EQ(0, on_increase.activate(4));
    EXPECT_EQ(a, 0);
    EXPECT_EQ(b, 0);

    std::vector<ScopedCallback<int>> callback_number_duo;
    {
        auto callback_number_uno = make_scoped_callback(
            [&](int inc) {
                a += inc;
            },
            &on_increase
        );

        EXPECT_EQ(1, on_increase.activate(3));
        EXPECT_EQ(a, 3);
        EXPECT_EQ(b, 0);

        callback_number_duo.emplace_back(
            make_scoped_callback([&](int inc) {
                b += inc;
            },
            &on_increase
        ));

        EXPECT_EQ(2, on_increase.activate(2));
        EXPECT_EQ(a, 5);
        EXPECT_EQ(b, 2);
    }
    // callback_number_uno deallocated

    EXPECT_EQ(1, on_increase.activate(3));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);

    callback_number_duo.clear();

    EXPECT_EQ(0, on_increase.activate(11));
    EXPECT_EQ(a, 5);
    EXPECT_EQ(b, 5);
}
