#include <iostream>
#include <vector>
#include "mshadow/tensor.h"
#include "dali/mat/Mat.h"
#include "dali/mat/math/LazyTensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using std::vector;

typedef Tensor<cpu, 2, float> cpu_t;
typedef Tensor<gpu, 2, float> gpu_t;

struct sigmoid {
    MSHADOW_XINLINE static float Map(float a) {
        return 1.0f / (1.0f + expf(-a));
    }
};

typedef float R;

typedef LazyTensor<cpu_t, gpu_t, R, type::kRValue> wrapped_t;

int main() {
    dali_init();


    Mat<R> A(2, 3, weights<R>::gaussian(2.0));
    Mat<R> B(2, 3);

    ELOG(A.w()->w.gpu_fresh);
    ELOG(A.w()->w.cpu_fresh);

    A.print();

    A.w(0) = 1.0;
    A.w(1) = 2.0;

    A.print();

    ELOG(A.w()->w.gpu_fresh);
    ELOG(A.w()->w.cpu_fresh);

    wrapped_t a(A.w()->w);
    wrapped_t b(B.w()->w);

    auto c = a;// + b;

    auto d   = c + b;
    auto g   = a.T();
    auto gg  = g.T().T();
    auto f   = c * b;
    auto sig = F<sigmoid>(a);

    ELOG(A.w()->w.gpu_fresh);
    ELOG(A.w()->w.cpu_fresh);

    auto out = Mat<R>::empty_like(A);
    out.w()->w = c;

    ELOG(A.w()->w.gpu_fresh);
    ELOG(A.w()->w.cpu_fresh);

    out.print();

    mshadow::ShutdownTensorEngine<mshadow::gpu>();

}
