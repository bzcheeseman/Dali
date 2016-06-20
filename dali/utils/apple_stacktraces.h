#ifndef DALI_UTILS_APPLE_STACKTRACES_H
#define DALI_UTILS_APPLE_STACKTRACES_H

#include "dali/config.h"
#if EXISTS_AND_TRUE(DALI_APPLE_STACKTRACES)
#include <signal.h>

struct ErrorHandler {
    struct sigaction sa;
    ErrorHandler();
};
#else
struct ErrorHandler {};
#endif
#endif
