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
using Duration = std::chrono::duration<double>;

class ThreadPool {
    private:
        static __thread bool in_thread_pool;
        // c++ assigns random id to each thread. This is not a thread_id
        // it's a number inside this thread pool.
        static __thread int thread_number;

        bool should_terminate;
        std::mutex queue_mutex;
        std::condition_variable is_idle;
        int active_count;

        std::deque<function<void()> > work;
        vector<thread> pool;
        Duration between_queue_checks;

        void thread_body(int _thread_id);
    public:
        // Creates a thread pool composed of num_threads threads.
        // threads are started immediately and exit only once ThreadPool
        // goes out of scope. Threads periodically check for new work
        // and the frequency of those checks is at minimum between_queue_checks
        // (it can be higher due to thread scheduling).
        ThreadPool(int num_threads, Duration between_queue_checks=milliseconds(1));

        // Run a function on a thread in pool.
        void run(function<void()> f);

        // Wait until queue is empty and all the threads have finished working.
        // If timeout is specified function waits at most timeout until the
        // threads are idle. If they indeed become idle returns true.
        bool wait_until_idle(Duration timeout);
        bool wait_until_idle();

        // Return number of active busy workers.
        int active_workers();

        // Can be called from within a thread to get a thread number.
        // the number is unique for each thread in the thread pool and
        // is between 0 and num_threads-1. If called outside thread_pool returns -1.
        static int get_thread_number();

        ~ThreadPool();
};


#endif
