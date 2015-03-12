#include "SaneCrashes.h"

namespace sane_crashes {
    void activate() {
        signal( SIGABRT, abortHandler );
        signal( SIGSEGV, abortHandler );
        signal( SIGILL,  abortHandler );
        signal( SIGFPE,  abortHandler );
    }


    void abortHandler( int signum ) {
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

        // Notify the user which signal was caught. We use printf, because this is the
        // most basic output function. Once you get a crash, it is possible that more
        // complex output systems like streams and the like may be corrupted. So we
        // make the most basic call possible to the lowest level, most
        // standard print function.
        std::cerr << "Caught signal " << signum << "(" << name << ")" << std::endl;


        // Dump a stack trace.
        // This is the function we will be implementing next.
        printStackTrace();

        // If you caught one of the above signals, it is likely you just
        // want to quit your program right now.
        exit(signum);
    }


    void printStackTrace(unsigned int max_frames) {
        std::cerr << "Stack trace:" << std::endl;

        // storage array for stack trace address data
        void* addrlist[max_frames + 1];

        // retrieve current stack addresses
        int addrlen = backtrace( addrlist, sizeof( addrlist ) / sizeof( void* ));

        if ( addrlen == 0 ) {
        std::cerr << std::endl;
            return;
        }

        // create readable strings to each frame.
        char** symbollist = backtrace_symbols(addrlist, addrlen);

        // print the stack trace.
        for (int i = 4; i < addrlen; i++ )
        std::cerr << symbollist[i] << std::endl;

        free(symbollist);
    }
};
