#include "Training.h"

using utils::assert2;
using utils::MS;

/* LSTV */

LSTV::LSTV(double short_term_forgetting,
           double long_term_forgetting,
           int patience) :               short_term_forgetting(short_term_forgetting),
                                         long_term_forgetting(long_term_forgetting),
                                         patience(patience),
                                         first_run(true),
                                         num_updates_validation_increasing(0) {
}


bool LSTV::should_stop(double validation_error) {
    if (first_run) {
        short_term_validation = validation_error;
        long_term_validation = validation_error;
        first_run = false;
    } else {
        short_term_validation = short_term_forgetting * validation_error +
                                (1.0 - short_term_forgetting) * short_term_validation;
        if (validation_error <= long_term_validation) {
            long_term_validation =  long_term_forgetting * validation_error +
                                    (1.0 - long_term_forgetting) * long_term_validation;
        }
    }

    if (short_term_validation > long_term_validation) {
        ++num_updates_validation_increasing;
    } else {
        num_updates_validation_increasing = 0;
    }
    bool is_validation_nan = !(validation_error == validation_error);
    return num_updates_validation_increasing > patience || is_validation_nan;
}

void LSTV::reset() {
    first_run = true;
    short_term_validation = -1;
    long_term_validation = -1;
    num_updates_validation_increasing = 0;
}

void LSTV::report() {
    std::cout << "LSTV: short: " << short_term_validation << " long: " << long_term_validation
         << " (patience " << num_updates_validation_increasing << "/" << patience << ")" << std::endl;
}

MaxEpochs::MaxEpochs(int max_epochs) : max_epochs(max_epochs) {
}

bool MaxEpochs::should_stop(double validation_error) {
    ++epochs_so_far;
    bool is_validation_nan = !(validation_error == validation_error);
    return epochs_so_far >= max_epochs || is_validation_nan;
}

void MaxEpochs::reset() {
    epochs_so_far = 0;
}

void MaxEpochs::report() {
    std::cout << "Epochs remaining: " << epochs_so_far << "/" << max_epochs << std::endl;
}


TimeLimited::TimeLimited(clock_t::duration max_training_duration) :
        max_training_duration(max_training_duration) {
    reset();
}

bool TimeLimited::should_stop(double validation_error) {
    bool is_validation_nan = !(validation_error == validation_error);
    return (clock_t::now() - training_start >= max_training_duration) || is_validation_nan;
}

void TimeLimited::reset() {
    training_start = clock_t::now();
}

void TimeLimited::report() {
    clock_t::duration time_passed = clock_t::now() - training_start;
    clock_t::duration remaining = max_training_duration - time_passed;
    std::cout << "Remaining training time is "
              // wow c++11!
              << std::chrono::duration_cast<std::chrono::seconds>(remaining).count()
              << " seconds of total "
              << std::chrono::duration_cast<std::chrono::seconds>(max_training_duration).count()
              << " seconds." << std::endl;
}

