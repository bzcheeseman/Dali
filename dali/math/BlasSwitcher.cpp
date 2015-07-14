#include "BlasSwitcher.h"
#include <iostream>


namespace blas {
    BLAS_PREFERENCE preferred_blas = AUTOMATIC_BLAS;

    std::thread::id main_thread_id = std::this_thread::get_id();

    // http://stackoverflow.com/questions/20530218/check-if-current-thread-is-main-thread
    bool running_in_main_thread () {
        return main_thread_id == std::this_thread::get_id();
    }

    bool should_use_cblas() {
        if (preferred_blas == EIGEN_BLAS) {
            return false;
        }
        if (preferred_blas == CBLAS_BLAS) {
            return true;
        }
        if (running_in_main_thread()) {
            return true;
        }
        return false;
    }
}
