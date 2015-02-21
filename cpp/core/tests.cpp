#include <vector>
#include "gtest/gtest.h"

#include "ThreadPool.h"

using std::vector;

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
    ASSERT_FALSE(t.wait_until_idle());
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


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
