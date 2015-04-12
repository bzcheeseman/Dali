#ifndef DALI_VISUALIZER_EVENT_QUEUE_H
#define DALI_VISUALIZER_EVENT_QUEUE_H

#include <chrono>
#include <queue>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>

struct EQHandle {
    std::shared_ptr<bool> run_again;

    EQHandle(std::shared_ptr<bool> run_again);
    ~EQHandle();
    void stop();
};

class EventQueue {
    public:
        typedef std::chrono::high_resolution_clock clock_t;
        typedef clock_t::duration duration_t;
        typedef clock_t::time_point time_point_t;

        typedef std::shared_ptr<EQHandle> repeating_t;
    private:
        typedef std::tuple<time_point_t, std::function<void()>> work_item_t;
        typedef std::function<bool(work_item_t, work_item_t)> comparator_t;

        static bool cmp_work_item(work_item_t a, work_item_t b);

        bool should_terminate = false;
        std::mutex queue_mutex;
        std::condition_variable work_ready;

        std::priority_queue<work_item_t,
                            std::vector<work_item_t>,
                            comparator_t> work;
        std::thread event_thread;

        void run_thread_internal();

    public:
        // How long to sleep if queue is empty.
        duration_t between_queue_checks = std::chrono::milliseconds(300);

        EventQueue();
        ~EventQueue();

        void push(std::function<void()> f);

        void push(std::function<void()> f, time_point_t when_to_execute);

        void push(std::function<void()> f, duration_t wait_before_execution);

        void stop();

        // First run happens immediately, then every time_between_execution.
        std::shared_ptr<EQHandle> run_every(std::function<void()> f, duration_t time_between_execution);
};

#endif
