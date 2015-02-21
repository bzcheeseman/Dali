#include "ThreadPool.h"

#include "gtest/gtest.h"

thread_local bool ThreadPool::in_thread_pool = false;
thread_local int ThreadPool::thread_number = -1;

ThreadPool::ThreadPool(int num_threads, dduration between_queue_checks) :
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

void ThreadPool::thread_body(int _thread_id) {
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

int ThreadPool::active_workers() {
    std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
    return active_count;
}

void ThreadPool::wait_until_idle(dduration timeout) {
    std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
    is_idle.wait_for(lock, timeout, [this]{
        return active_count == 0 && work.empty();
    });
}

void ThreadPool::wait_until_idle() {
    std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);

    is_idle.wait(lock, [this]{
        return active_count == 0 && work.empty();
    });
}

void ThreadPool::run(function<void()> f) {
    std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
    work.push_back(f);
}

ThreadPool::~ThreadPool() {
    // Terminates thread pool making sure that all the work
    // is completed.
    should_terminate = true;
    for(auto& t: pool) t.join();
}

