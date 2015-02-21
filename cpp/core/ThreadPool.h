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
        ThreadPool(int num_threads, dduration between_queue_checks=milliseconds(1));

        void thread_body(int _thread_id);

        int active_workers();

        void wait_until_idle(dduration timeout);

        void wait_until_idle();

        void run(function<void()> f);

        ~ThreadPool();
};





#endif
