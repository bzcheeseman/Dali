#include <iostream>
#include <vector>
#include "mshadow/tensor.h"
#include "dali/tensor/Mat.h"
#include "dali/math/LazyTensor.h"

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

    ELOG(A.w().gpu_fresh);
    ELOG(A.w().cpu_fresh);
    std::cout << "A=" << std::endl;
    A.print();

    A.w(0) = 1.0;
    A.w(1) = 2.0;
    std::cout << "A=" << std::endl;
    A.print();

    ELOG(A.w().gpu_fresh);
    ELOG(A.w().cpu_fresh);

    auto a = A.w().wrapper();
    auto b = B.w().wrapper();


    auto XXXX = ((a+b).T());
    auto YYYY = Mat<R>(3,2);
    YYYY.w() = XXXX;

    std::cout << "YYYY equals" << std::endl;
    YYYY.print();


    auto c = a;// + b;

    auto d   = c + b;
    auto g   = a.T();
    auto gg  = g.T().T();
    auto f   = c * b;
    auto sig = F<sigmoid>(a);

    ELOG(A.w().gpu_fresh);
    ELOG(A.w().cpu_fresh);

    auto out = Mat<R>::empty_like(A);
    out.w() = c * (float)3.0;

    ELOG(A.w().gpu_fresh);
    ELOG(A.w().cpu_fresh);
    std::cout << "out=" << std::endl;
    out.print();


    auto out2 = Mat<R>::empty_like(A);

    out2.w() = c.softmax();
    std::cout << "out2=" << std::endl;
    out2.print();
    Mat<R> C(3, 1);

    C.w() = c[0].broadcast<0>(C.w().shape());
    std::cout << "C=" << std::endl;
    C.print();

    Mat<R> D(4, 3);

    std::cout << c[0].repmat(4).left.shape_[0] << ", " << c[0].repmat(4).left.shape_[1] << std::endl;

    D.w() = c[0].repmat(4);
    std::cout << "D=" << std::endl;
    D.print();

    Mat<R> lhs(1, 3);

    lhs.w(0) = 1;
    lhs.w(1) = 2;
    lhs.w(2) = 3;

    Mat<R> rhs(1, 3);

    rhs.w(0) = 4;
    rhs.w(1) = 5;
    rhs.w(2) = 6;

    Mat<R> out3(1,1);

    out3.w() = dot(lhs.w().wrapper(), rhs.w().wrapper().T());

    std::cout << "out3=" << std::endl;
    out3.print();

    mshadow::ShutdownTensorEngine<mshadow::gpu>();

}
