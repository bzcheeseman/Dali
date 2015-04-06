#ifndef CORE_TRAINING_H
#define CORE_TRAINING_H
#include <iostream>

#include "dali/utils/core_utils.h"

class Training {
    public:
        // returns true if training should stop.
        virtual bool should_stop(double validation) = 0;
        // Reset the state.
        virtual void reset() = 0;
        // It should output a message describing progress of the training.
        // After user of Dali reads such a message she/he should have a
        // deep feeling of fulfillment. It should be a bright spark
        // in otherwise dark and depressing life of a researcher.
        virtual void report() = 0;

        // returns modified validation error. Some training methods
        // use averaging to provide better estimate of the error.
        virtual double validation_error() = 0;
};

class LSTV : public Training {
    bool first_run;
    double short_term_validation;
    double long_term_validation;

    double best_validation;

    int num_updates_validation_increasing;

    const double short_term_forgetting;
    const double long_term_forgetting;
    const int patience;

    public:

        LSTV(double short_term_forgetting=0.1,
             double long_term_forgetting=0.01,
             int patience=5);

        virtual bool should_stop(double validation) override;
        virtual void reset() override;
        virtual void report() override;
        virtual double validation_error() override;
};

class MaxEpochs : public Training {
    int epochs_so_far;

    const int max_epochs;

    double last_validation;

    public:

        MaxEpochs(int max_epochs);

        virtual bool should_stop(double validation) override;
        virtual void reset() override;
        virtual void report() override;
        virtual double validation_error() override;

};

class TimeLimited : public Training {
    typedef std::chrono::high_resolution_clock clock_t;
    clock_t::duration max_training_duration;
    clock_t::time_point training_start;

    double last_validation;

    public:

        TimeLimited(clock_t::duration max_training_duration);

        virtual bool should_stop(double validation) override;
        virtual void reset() override;
        virtual void report() override;
        virtual double validation_error() override;

};

#endif
