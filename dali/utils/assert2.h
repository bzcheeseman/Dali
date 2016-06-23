#ifndef DALI_UTILS_ASSERT2_H
#define DALI_UTILS_ASSERT2_H

#include <string>
#include <stdexcept>

// throw assertion errors with messages
namespace utils {
    void assert2(bool condition);
    void assert2(bool condition, std::string message);
}
#define ASSERT2(condition, message) if (!(condition)) utils::assert2(false, (message))

#define ASSERT2_SHAPE_ND(SHAPE,DIM,MSG) \
    if (SHAPE.size() != DIM) \
        utils::assert2(false, utils::MS() << MSG << " was expecting dimension " << DIM  << ", got shape " << SHAPE << ".");

#define ASSERT2_EQ(EXPECTED,ACTUAL,MSG) \
    if (EXPECTED != ACTUAL) \
        utils::assert2(false, utils::MS() << "Expected " << EXPECTED  << ", got " << ACTUAL << ": " << MSG << ".");

#endif
