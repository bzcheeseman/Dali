#include "dali/utils/assert2.h"
#include <stdexcept>
#include "dali/utils/apple_stacktraces.h"

namespace utils {
    void assert2(bool condition, std::string message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void assert2(bool condition) {
        assert2(condition, "");
    }
}

// create a global hook for error / signal handling
ErrorHandler global_error_handler = ErrorHandler();
