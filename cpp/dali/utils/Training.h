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
};

class LSTV : public Training {
    bool first_run;
    double short_term_validation;
    double long_term_validation;

    int num_updates_validation_increasing;

    const double short_term_forgetting;
    const double long_term_forgetting;
    const int patience;

    public:

        LSTV(double short_term_forgetting=0.1,
             double long_term_forgetting=0.01,
             int patience=5);

        virtual bool should_stop(double validation);
        virtual void reset();
        virtual void report();
};

class MaxEpochs : public Training {
    int epochs_so_far;

    const int max_epochs;

    public:

        MaxEpochs(int max_epochs);

        virtual bool should_stop(double validation);
        virtual void reset();
        virtual void report();
};

#endif
