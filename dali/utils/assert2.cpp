#include "dali/utils/assert2.h"
#define UNW_LOCAL_ONLY
#include <cxxabi.h>
#include <libunwind.h>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace utils {
    void assert2(bool condition, std::string message) {
        if (!condition) {
            throw std::runtime_error(obtain_backtrace(false, 1) + message);
        }
    }

    void assert2(bool condition) {
        assert2(condition, "");
    }

    std::string obtain_backtrace(const bool& add_register, const int& skip_frames) {
        std::stringstream ss;
        unw_cursor_t cursor;
        unw_context_t context;

        // Initialize cursor to current frame for local unwinding.
        unw_getcontext(&context);
        unw_init_local(&cursor, &context);

        // Unwind frames one by one, going up the frame stack.
        int level = 0;
        while (unw_step(&cursor) > 0) {
            level += 1;
            if (level <= skip_frames) {
                continue;
            }
            unw_word_t offset, pc;
            unw_get_reg(&cursor, UNW_REG_IP, &pc);
            if (pc == 0) {
                break;
            }
            if (add_register) {
                ss << "0x" << pc << ": ";
            }

            char sym[256];
            if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
                char* nameptr = sym;
                int status;
                char* demangled = abi::__cxa_demangle(sym, nullptr, nullptr, &status);
                if (status == 0) {
                    nameptr = demangled;
                }
                ss << nameptr << "+0x" << offset << "\n";
                std::free(demangled);
            } else {
                ss << "-- error: unable to obtain symbol name for this frame\n";
            }
        }
        return ss.str();
    }
}
