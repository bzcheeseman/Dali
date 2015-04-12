#include "EventQueue.h"

#include <iostream>

using namespace std::chrono;


EQHandle::EQHandle(std::shared_ptr<bool> run_again) : run_again(run_again) {
    *run_again = true;
}
EQHandle::~EQHandle() {
    stop();
}
void EQHandle::stop() {
    *run_again = false;
}

void EventQueue::run_thread_internal() {
     while (true) {
        if (should_terminate)
            break;
        bool good_to_go = false;
        duration_t should_sleep_for = between_queue_checks;
        time_point_t dont_run_before;

        std::function<void()> f;
        {
            std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
            if (!work.empty()) {
                std::tie(dont_run_before, f) = work.top();
                time_point_t now = clock_t::now();
                if (dont_run_before < now) {
                    work.pop();
                    good_to_go = true;
                } else {
                    // Sleep at most between queue checks.
                    // Ideally time before next event.
                    should_sleep_for = dont_run_before - now;
                }
            }
        }

        if (good_to_go) {
            f();
            std::this_thread::yield();
        } else {
                std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
                work_ready.wait_for(lock, should_sleep_for, [this]{
                    return not work.empty();
                });
        }
    }
}

bool EventQueue::cmp_work_item(work_item_t a, work_item_t b) {
    return std::get<0>(a) > std::get<0>(b);
}


EventQueue::EventQueue() :
        work(cmp_work_item),
        event_thread([this]() { this->run_thread_internal(); }) {
}

EventQueue::~EventQueue() {
    stop();
}

void EventQueue::push(std::function<void()> f) {
    push(f, duration_t::zero());
}

void EventQueue::push(std::function<void()> f, time_point_t when_to_execute) {
    std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
    // we negate the time point for the queue to execute it at the right time.
    work.push(std::make_tuple(when_to_execute, f));
    work_ready.notify_all();
}

void EventQueue::push(std::function<void()> f, duration_t wait_before_execution) {
    push(f, clock_t::now() + wait_before_execution);
}

void EventQueue::stop() {
    should_terminate = true;
    event_thread.join();
}

// First run happens immediately, then every time_between_execution.
std::shared_ptr<EQHandle> EventQueue::run_every(std::function<void()> f, duration_t time_between_execution) {
    auto run_again = std::make_shared<bool>(true);
    auto handle = std::make_shared<EQHandle>(run_again);

    auto looper = std::make_shared<std::function<void ()>>();
    *looper =
            [this, looper, f, run_again, time_between_execution]() {
        if(*run_again) {
            f();
            this->push(*looper, time_between_execution);
        }
    };
    push(*looper);
    return handle;
}
