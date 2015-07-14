#ifndef DALI_MATH_BLAS_SWITCHER_H
#define DALI_MATH_BLAS_SWITCHER_H
#include <thread>

namespace blas {
    enum BLAS_PREFERENCE {
        AUTOMATIC_BLAS,
        CBLAS_BLAS,
        EIGEN_BLAS
    };

    extern std::thread::id main_thread_id;

    extern BLAS_PREFERENCE preferred_blas;

    bool should_use_cblas();
}

#endif
