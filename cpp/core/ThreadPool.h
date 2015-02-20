#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <cassert>
#include <deque>
#include <chrono>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <thread>
#include <vector>

using std::chrono::milliseconds;
using std::function;
using std::vector;
using std::thread;

typedef std::chrono::duration<double> dduration;

class ThreadPool {
    private:
        static thread_local bool in_thread_pool;
        // c++ assigns random id to each thread. This is not a thread_id
        // it's a number inside this thread pool.
        static thread_local int thread_number;

        bool should_terminate;
        std::mutex queue_mutex;
        std::condition_variable is_idle;
        int active_count;

        std::deque<function<void()> > work;
        vector<thread> pool;
        dduration between_queue_checks;
    public:
        ThreadPool(int num_threads, dduration between_queue_checks=milliseconds(1)) :
                between_queue_checks(between_queue_checks),
                should_terminate(false),
                active_count(0) {
            // Thread pool inception is not supported at this time.
            assert(!in_thread_pool);

            ThreadPool::between_queue_checks = between_queue_checks;
            for (int thread_number = 0; thread_number < num_threads; ++thread_number) {
                pool.emplace_back(&ThreadPool::thread_body, this, thread_number);
            }
        }

        void thread_body(int _thread_id) {
            in_thread_pool = true;
            thread_number = thread_number;
            bool am_i_active = false;

            while (true) {
                function<void()> f;
                {
                    std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
                    bool was_i_active = am_i_active;
                    if (should_terminate && work.empty())
                        break;
                    if (!work.empty()) {
                        am_i_active = true;
                        f = work.front();
                        work.pop_front();
                    } else {
                        am_i_active = false;
                    }

                    if (am_i_active != was_i_active) {
                        active_count += am_i_active ? 1 : -1;
                        if (active_count == 0) {
                            // number of workers decrease so maybe all are idle
                            is_idle.notify_all();
                        }
                    }
                }
                // Function defines implicit conversion to bool
                // which is true only if call target was set.
                if ((bool)f) {
                    f();
                } else {
                    std::this_thread::sleep_for(between_queue_checks);
                }
            }
        }

        int active_workers() {
            std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
            return active_count;
        }

        void wait_until_idle(dduration timeout) {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
            is_idle.wait_for(lock, timeout, [this]{
                return active_count == 0 && work.empty();
            });
        }

        void wait_until_idle() {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);

            is_idle.wait(lock, [this]{
                return active_count == 0 && work.empty();
            });
        }

        void run(function<void()> f) {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
            work.push_back(f);
        }

        ~ThreadPool() {
            // Terminates thread pool making sure that all the work
            // is completed.
            should_terminate = true;
            for(auto& t: pool) t.join();
        }
};

thread_local bool ThreadPool::in_thread_pool = false;
thread_local int ThreadPool::thread_number = -1;

void test_idle () {
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

#endif
