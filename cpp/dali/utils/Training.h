#ifndef CORE_TRAINING_H
#define CORE_TRAINING_H
#include <iostream>


class LSTV {
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

        // prints state to stdout
        void report();

        // returns true if training should stop.
        bool update(double validation);
};

#endif
