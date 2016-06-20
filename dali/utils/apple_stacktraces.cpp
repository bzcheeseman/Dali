#include "apple_stacktraces.h"

#if EXISTS_AND_TRUE(DALI_APPLE_STACKTRACES)

#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>   // for backtrace
#include <future>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::string get_sourceline(char *filename, void *ptr, void* base) {
    char buf[1024];
    FILE *output;
    int c, i = 0;

#ifdef __APPLE__
    snprintf(buf, sizeof(buf), "atos -o %s -l %p %p 2>&1 | tail -n1",
                     filename, base, ptr);
#else /* !__APPLE__ */
    snprintf(buf, sizeof(buf), "addr2line -e %s %p", filename, ptr);
#endif /* __APPLE__ */

    output = popen(buf, "r");
    if (output) {
        while (i < sizeof(buf)) {
            c = getc(output);
            if (c == '\n' || c == EOF) {
                buf[i++] = 0;
                break;
            }
            buf[i++] = c;
        }
        pclose(output);
        return std::string(buf);
    }
    return "";
}

void printStackTrace(unsigned int max_frames, int num_parallel_symbol_seeks) {
    // storage array for stack trace address data
    void* addrlist[max_frames + 1];
    // retrieve current stack addresses
    int addrlen = backtrace( addrlist, sizeof( addrlist ) / sizeof( void* ));

    if ( addrlen == 0 ) {
        std::cerr << std::endl;
        return;
    }

    sigset_t pending;
    sigpending(&pending);
    if (sigismember(&pending, SIGHUP) || sigismember(&pending, SIGINT) || sigismember(&pending, SIGTERM)) {
        return;
    }

    std::vector<std::future<std::string>> extracted_lines_futures;
    std::vector<std::function<std::string()>> line_extractors;

    std::atomic<int> num_done(0);
    int num_sent = 0;

    for (int i = 4; i < addrlen; i++) {
        line_extractors.emplace_back([&addrlist, i, &num_done]() -> std::string {
            Dl_info info;
            if (dladdr(addrlist[i], &info)) {
                void *ptr = addrlist[i];
                if (strstr(info.dli_fname, ".so")) {
                    ptr = (void*)((char*)addrlist[i] - (char*)info.dli_fbase);
                }
                if (info.dli_fname[0]) {
                    std::string line = get_sourceline((char*)info.dli_fname,
                        ptr, info.dli_fbase
                    );
                    num_done += 1;
                    return line;
                } else {
                    std::stringstream ss;
                    if (info.dli_sname) {
                        ss << info.dli_fname << " @ " << info.dli_fbase
                           << " " << info.dli_sname << " "
                           << info.dli_fname << " " << info.dli_fbase
                           << " [" << addrlist[i] << "]";
                    } else {
                        ss << info.dli_fname << " @ " << info.dli_fbase
                           << " [" << addrlist[i] << "]";
                    }
                    num_done += 1;
                    return ss.str();
                }
            } else {
                num_done += 1;
                return "";
            }
        });
    }

    int num_callbacks = line_extractors.size();

    // print the stack trace.
    for (std::size_t i = 0; i < num_callbacks; i++ ) {
        if (num_sent <= num_done) {
            for (std::size_t idx = 0; idx < std::min((int)line_extractors.size(), num_parallel_symbol_seeks); idx++) {
                extracted_lines_futures.emplace_back(std::async(std::launch::async, line_extractors[idx]));
            }
            num_sent += std::min((int)line_extractors.size(), num_parallel_symbol_seeks);
            line_extractors.erase(line_extractors.begin(), line_extractors.begin() + std::min((int)line_extractors.size(), num_parallel_symbol_seeks));
        }

        auto source_line = extracted_lines_futures[i].get();
        if (source_line.size() > 0) {
            std::cerr << i << ": " << source_line << std::endl;
        }
        sigset_t pending;
        sigpending(&pending);
        if (sigismember(&pending, SIGHUP) || sigismember(&pending, SIGINT) || sigismember(&pending, SIGTERM)) {
            break;
        }
    }
}

void signal_abort_handler(int signum, siginfo_t *info, void *_ctx) {
    // associate each signal with a signal name string.
    std::string name;
    switch( signum ) {
        case SIGABRT:
            name = "SIGABRT";
            break;
        case SIGSEGV:
            name = "SIGSEGV";
            break;
        case SIGBUS:
            name = "SIGBUS";
            break;
        case SIGILL:
            name = "SIGILL";
            break;
        case SIGFPE:
            name = "SIGFPE";
            break;
        default:
            name = "UNKNOWN";
            break;
    }
    std::cerr << "Caught signal " << signum << "(" << name << ")" << std::endl;

    signal(SIGABRT, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGILL, SIG_DFL);
    signal(SIGFPE, SIG_DFL);

    printStackTrace(63, 7);
    // If you caught one of the above signals, it is likely you just
    // want to quit your program right now.
    exit(signum);
}

ErrorHandler::ErrorHandler() {
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = signal_abort_handler;
    sigemptyset(&sa.sa_mask);

    sigaction(SIGABRT, &sa, 0);
    sigaction(SIGSEGV, &sa, 0);
    sigaction(SIGILL, &sa, 0);
    sigaction(SIGFPE, &sa, 0);
};

#endif
