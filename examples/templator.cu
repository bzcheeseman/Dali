#include <iostream>
#include "mshadow/tensor.h"

using namespace mshadow;
using namespace mshadow::expr;

typedef Tensor<cpu, 2, float> cpu_t;
typedef Tensor<gpu, 2, float> gpu_t;

template<typename LeftType, typename RightType, typename DType, int ktype>
class Wrapper {
    public:
        LeftType left;
        RightType right;

        Wrapper(
            const LeftType& _left,
            const RightType& _right) : left(_left), right(_right) {}

        inline Wrapper<TransposeExp<LeftType, DType>, TransposeExp<RightType, DType>, DType, ktype> T(void) const {
            auto cpu_T = left.T();
            auto gpu_T = right.T();
            return Wrapper<decltype(cpu_T),decltype(gpu_T), DType, ktype>(cpu_T, gpu_T);
        }
};

#define BINARY_OP(opname, opsymbol) \
template<typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb> \
Wrapper< BinaryMapExp<opname, TA, TC, DType, (ta|tb|type::kMapper)>, BinaryMapExp<opname, TB, TD, DType, (ta|tb|type::kMapper)>, DType, (ta|tb|type::kMapper)> operator opsymbol( \
        const Wrapper<TA, TB, DType, ta> &left, \
        const Wrapper<TC, TD, DType, tb> &right) { \
    const auto& l_cpu = left.left; \
    const auto& r_cpu = right.left; \
    auto res_cpu = l_cpu opsymbol r_cpu; \
    const auto& l_gpu = left.right; \
    const auto& r_gpu = right.right; \
    auto res_gpu = l_gpu opsymbol r_gpu; \
    return Wrapper<decltype(res_cpu), decltype(res_gpu), DType, (ta|tb|type::kMapper)>(res_cpu, res_gpu); \
}

BINARY_OP(op::plus,  +);
BINARY_OP(op::mul,   *);
BINARY_OP(op::minus, -);

/*! \brief make expression */
template<typename OP, typename TA, typename TB, typename DType, int ta>
inline Wrapper<UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>, UnaryMapExp<OP, TB, DType, (ta|type::kMapper)>, DType, (ta|type::kMapper)>
MakeExp(const Wrapper<TA, TB, DType, ta> &src) {
    auto unary_l = UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>(src.left);
    auto unary_r = UnaryMapExp<OP, TB, DType, (ta|type::kMapper)>(src.right);
    return Wrapper<decltype(unary_l), decltype(unary_r), DType, (ta|type::kMapper)>(unary_l, unary_r);;
}

template<typename OP, typename TA, typename TB, typename DType, int ta>
inline Wrapper<UnaryMapExp<OP, TA, DType, (ta|type::kMapper)>, UnaryMapExp<OP, TB, DType, (ta|type::kMapper)>, DType, (ta|type::kMapper)>
F(const Wrapper<TA, TB, DType, ta> &src) {
    return MakeExp<OP>(src);
}

typedef Wrapper<cpu_t, gpu_t, float, type::kRValue> cool_cool_t;

struct sigmoid {
  MSHADOW_XINLINE static float Map(float a) {
    return 1.0f / (1.0f + expf(-a));
  }
};

int main() {

    cool_cool_t a(
        cpu_t(Shape2(2,3)),
        gpu_t(Shape2(2,3))
    );

    cool_cool_t b(
        cpu_t(Shape2(2,3)),
        gpu_t(Shape2(2,3))
    );

    auto c = a + b;
    auto d = c + b;

    auto g = a.T();

    auto f = c * b;

    auto sig = F<sigmoid>(a);

}
