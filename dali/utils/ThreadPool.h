#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
    private:
        typedef std::chrono::duration<double> Duration;
        static __thread bool in_thread_pool;
        // c++ assigns random id to each thread. This is not a thread_id
        // it's a number inside this thread pool.
        static __thread int thread_number;

        static std::mutex printing_lock;

        bool should_terminate;
        std::mutex queue_mutex;
        std::condition_variable is_idle;
        int active_count;

        std::deque<std::function<void()> > work;
        std::vector<std::thread> pool;
        Duration between_queue_checks;

        void thread_body(int _thread_id);
    public:
        // Creates a thread pool composed of num_threads threads.
        // threads are started immediately and exit only once ThreadPool
        // goes out of scope. Threads periodically check for new work
        // and the frequency of those checks is at minimum between_queue_checks
        // (it can be higher due to thread scheduling).
        ThreadPool(int num_threads, Duration between_queue_checks=std::chrono::milliseconds(1));

        // Run a function on a thread in pool.
        void run(std::function<void()> f);

        // Wait until queue is empty and all the threads have finished working.
        // If timeout is specified function waits at most timeout until the
        // threads are idle. If they indeed become idle returns true.
        bool wait_until_idle(Duration timeout);
        bool wait_until_idle();

        // Retruns true if all the work is done.
        bool idle() const;
        // Return number of active busy workers.
        int active_workers();

        // Can be called from within a thread to get a thread number.
        // the number is unique for each thread in the thread pool and
        // is between 0 and num_threads-1. If called outside thread_pool returns -1.
        static int get_thread_number();

        static void print_safely(std::string message);

        ~ThreadPool();
};


#endif
