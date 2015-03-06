#include <execinfo.h>
#include <iostream>
#include <string>
#include <signal.h>

namespace sane_crashes {
    void activate();

    void abortHandler(int signum);

    void printStackTrace(unsigned int max_frames=63 );
};
