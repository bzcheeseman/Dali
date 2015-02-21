#include "gtest/gtest.h"

#include "ThreadPool.h"

TEST(ThreadPool, wait_until_idle) {
    const int NUM_THREADS = 1;
    const int NUM_JOBS = 1;
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
    while(t.active_workers() <= 0);

    // Threads are waiting for barrier.
    // t.wait_until_idle(milliseconds(1));
    assert(t.active_workers() > 0);

    // Remove barrier and hope they come back.
    barrier = false;
    t.wait_until_idle(milliseconds(1));
    assert(t.active_workers() == 0);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
