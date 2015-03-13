#include "Training.h"

/* LSTV */

LSTV::LSTV(double short_term_forgetting,
           double long_term_forgetting,
           int patience) :               short_term_forgetting(short_term_forgetting),
                                         long_term_forgetting(long_term_forgetting),
                                         patience(patience),
                                         first_run(true),
                                         num_updates_validation_increasing(0) {

}

void LSTV::report() {
    std::cout << "LSTV: short: " << short_term_validation << " long: " << long_term_validation
         << " (patience " << num_updates_validation_increasing << "/" << patience << ")" << std::endl;
}

bool LSTV::update(double validation_error) {
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
    return num_updates_validation_increasing > patience;
}

