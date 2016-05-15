#include "dali_gemm_engine_exp.h"

namespace mshadow {
    namespace expr {
        mshadow::Stream<mshadow::gpu>* default_stream = NULL;

        template<>
        mshadow::Stream<mshadow::gpu>* get_default_gemm_stream() {
            if (default_stream == NULL) {
                default_stream = new mshadow::Stream<mshadow::gpu>();
                default_stream->CreateBlasHandle();
            }
            return default_stream;
        }

        template<>
        mshadow::Stream<mshadow::cpu>* get_default_gemm_stream() {
            return NULL;
        }

    }
}
