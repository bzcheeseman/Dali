#include <chrono>
#include <vector>
#include <gtest/gtest.h>
#include "dali/data_processing/Glove.h"
#include "dali/utils/ThreadPool.h"
#include "dali/data_processing/Arithmetic.h"

using std::vector;
using std::chrono::milliseconds;

TEST(ThreadPool, wait_until_idle) {
    const int NUM_THREADS = 10;
    const int NUM_JOBS = 100;
    ThreadPool t(NUM_THREADS);

    bool barrier = true;

    for(int j=0; j<NUM_JOBS; ++j) {
        t.run([&barrier]() {
            while(barrier) {
                std::this_thread::yield();
            };
        });
    }

    // wait for threads to pick up work.
    while(t.active_workers() < NUM_THREADS);

    // Threads are currently waiting for barrier.
    // Ensure that wait until idle returns false..
    ASSERT_FALSE(t.wait_until_idle(milliseconds(1)));
    // Remove barrier and hope they come back.
    barrier = false;

    // Assert all threads will be done exentually.
    ASSERT_TRUE(t.wait_until_idle());
}

TEST(ThreadPool, thread_number) {
    const int NUM_THREADS = 4;
    const int JOBS_PER_ATTEMPT = 10;
    ThreadPool pool(NUM_THREADS);


    for(int t = 0; t < NUM_THREADS; ++t) {
        // Try to get thread t to manifest itself by checking
        // it's thread_number.
        bool thread_t_seen = false;
        while (!thread_t_seen) {
            for(int job = 0; job < JOBS_PER_ATTEMPT; ++job) {
                pool.run([&t, &thread_t_seen]() {
                    for (int i=0; i<10000; ++i) {
                        if(t == ThreadPool::get_thread_number()) {
                            thread_t_seen = true;
                        }
                    }
                });
            }
            pool.wait_until_idle();
        }
    }
}

TEST(Glove, load) {
    auto embedding = glove::load<double>( STR(DALI_DATA_DIR) "/glove/test_data.txt");
    ASSERT_EQ(std::get<1>(embedding).index2word.size(), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(0), 21);
    ASSERT_EQ(std::get<0>(embedding).dims(1), 300);
}

TEST(arithmetic, generate) {
    int min = 0;
    int max = 9;
    auto example = arithmetic::generate_example(5, min, max);
    ASSERT_EQ(std::get<0>(example).size(), 3);
    ASSERT_EQ(std::get<1>(example).size(), 2);

    auto example2     = std::make_tuple(std::vector<int>({12, 9, 3}), std::vector<std::string>({"*", "+"}));
    auto demultiplied = arithmetic::remove_multiplies(std::get<0>(example2), std::get<1>(example2));
    auto example3     = std::make_tuple(std::vector<int>({108, 3}), std::vector<std::string>({"+"}));

    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(arithmetic::compute_result(std::get<0>(example3), std::get<1>(example3)), 111);


    example2     = std::make_tuple(std::vector<int>({12, 9, 3, 5, 5}), std::vector<std::string>({"*", "-","+", "*"}));
    demultiplied = arithmetic::remove_multiplies(std::get<0>(example2), std::get<1>(example2));
    example3     = std::make_tuple(std::vector<int>({108, 3, 25}), std::vector<std::string>({"-", "+"}));

    ASSERT_EQ(demultiplied, example3);
    ASSERT_EQ(arithmetic::compute_result(std::get<0>(example3), std::get<1>(example3)), 130);
}

