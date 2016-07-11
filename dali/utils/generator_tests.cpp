#include <gtest/gtest.h>
#include <vector>

#include "dali/utils/generator.h"


struct Range : utils::GeneratorHeart<int> {
    void run(int start, int end, int interval=1) {
        for (int i=start; i<end; i+=interval) {
            yield(i);
        }
    }
};

TEST(utils, generator_test) {
    auto vals = std::vector<int>();
    for (int i : utils::Gen<Range>(2,9,2)) vals.emplace_back(i);
    ASSERT_EQ(vals, std::vector<int>({2, 4, 6, 8}));
}

TEST(utils, lambda_generator_test) {
    auto gen = utils::Generator<int>([](utils::yield_t<int> yield) {
        for (int i=2; i<9; i+=2) yield(i);
    });
    auto vals = std::vector<int>();
    for (int i : gen)
        vals.emplace_back(i);
    ASSERT_EQ(vals, std::vector<int>({2, 4, 6, 8}));
}

TEST(utils, test_initialize_gen) {
    // This test illustrates that generator_constructor can be sometimes
    // dangerous if we do not think about initialization

    // TEST GOAL: generate {1,2,3,4,5,  1,2,3,4,5} using shared_resource.

    int shared_resource = 1;

    auto advance_noinitialization = [&shared_resource](utils::yield_t<int> yield) {
        int repeats = 5;
        while(repeats--) {
            yield(shared_resource++);
        }
    };

    auto advance_correct = [&shared_resource](utils::yield_t<int> yield) {
        shared_resource = 1;
        int repeats = 5;
        while(repeats--) {
            yield(shared_resource++);
        }
    };

    auto noinitialization = utils::Generator<int>(advance_noinitialization);
    auto correct = utils::Generator<int>(advance_correct);

    auto vals = std::vector<int>();
    for (int i : noinitialization)
        vals.emplace_back(i);
    noinitialization.reset();
    for (int i : noinitialization)
        vals.emplace_back(i);
    ASSERT_EQ(std::vector<int>({1,2,3,4,5,6,7,8,9,10}), vals);

    vals.clear();
    for (int i : correct)
        vals.emplace_back(i);
    correct.reset();
    for (int i : correct)
        vals.emplace_back(i);
    ASSERT_EQ(std::vector<int>({1,2,3,4,5,1,2,3,4,5}), vals);
}


TEST(utils, recursive_generator_test) {
    // here we are using Generator rather than make generator,
    // so that we can use it multiple times. For example each time we call
    // gen_12345() new generator is constructed.

    // TEST GOAL: generate {1,2,3,4,5} five times.
    auto gen_12345 = utils::Generator<int>([](utils::yield_t<int> yield) {
        for (int i=1; i<=5; i+=1) yield(i);
    });
    auto gen_5x_12345 = utils::Generator<int>([&gen_12345](utils::yield_t<int> yield) {
        int repeats = 5;
        while(repeats--) {
            gen_12345.reset();
            for (auto num: gen_12345)
                yield(num);
        }
    });

    auto vals = std::vector<int>();
    for (int i : gen_5x_12345)
        vals.emplace_back(i);
    ASSERT_EQ(std::vector<int>({1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5}), vals);
}

TEST(utils, combine_generators) {
    // here we take two short generators and
    // create a longer one out of the pair:
    auto comb_gen = (
        utils::Generator<int>([](utils::yield_t<int> yield) {
            for (int i=1; i<=5; i+=1) yield(i);
        })
        +
        utils::Generator<int>([](utils::yield_t<int> yield) {
            for (int i=6; i<=10; i+=1) yield(i);
        })
    );

    auto vals = std::vector<int>();
    for (int i : comb_gen)
        vals.emplace_back(i);

    ASSERT_EQ(vals, std::vector<int>({1,2,3,4,5,6,7,8,9,10}));

}
