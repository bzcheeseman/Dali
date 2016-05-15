#include "dali_gemm_engine_exp.h"

#include "dali/config.h"

namespace mshadow {
    namespace expr {
#ifdef DALI_USE_CUDA
        mshadow::Stream<mshadow::gpu>* default_stream = NULL;

        template<>
        mshadow::Stream<mshadow::gpu>* get_default_gemm_stream() {
            if (default_stream == NULL) {
                default_stream = new mshadow::Stream<mshadow::gpu>();
                default_stream->CreateBlasHandle();
            }
            return default_stream;
        }
#endif

        template<>
        mshadow::Stream<mshadow::cpu>* get_default_gemm_stream() {
            return NULL;
        }

    }
}
