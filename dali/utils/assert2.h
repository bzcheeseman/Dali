#ifndef DALI_UTILS_ASSERT2_H
#define DALI_UTILS_ASSERT2_H

#include <string>
#include <stdexcept>

// throw assertion errors with messages
namespace utils {
    void assert2(bool condition);
    void assert2(bool condition, std::string message);

    /** \brief Obtain backtrace for current stack
      * \param add_register : whether the address should be printed along
      *                       with function name
      * \param skip_frames : how many levels of execution should be ignored
      *                      when constructing the trace (Note: typically
      *                      this function is called within an assertion or
      *                      error-throwing method, thus the first level is
      *                      often uninformative).
      */
    std::string obtain_backtrace(const bool& add_register=true, const int& skip_frames=0);
}
#define ASSERT2(condition, message) if (!(condition)) utils::assert2(false, (message))

#endif
