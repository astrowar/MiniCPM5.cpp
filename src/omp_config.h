#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace omp_config {

inline void initialize(int threads) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
    #pragma omp parallel
    {
        // warm-up worker creation
    }
#else
    (void)threads;
#endif
}

} // namespace omp_config

