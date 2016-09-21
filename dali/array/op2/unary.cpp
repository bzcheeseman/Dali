#include "unary.h"
#include "dali/array/op2/fused_operation.h"

namespace op2 {
	FusedOperation identity(const FusedOperation& x) {
		return elementwise(x, "functor::identity");
    }

    FusedOperation sigmoid(const FusedOperation& x) {
        return elementwise(x, "functor::sigmoid");
    }

    FusedOperation tanh(const FusedOperation& x) {
        return elementwise(x, "functor::tanh");
    }

    FusedOperation exp(const FusedOperation& x) {
        return elementwise(x, "functor::exp");
    }

    FusedOperation softplus(const FusedOperation& x) {
        return elementwise(x, "functor::softplus");
    }

    FusedOperation eltinv(const FusedOperation& x) {
        return elementwise(x, "functor::inv");
    }

    FusedOperation relu(const FusedOperation& x) {
        return elementwise(x, "functor::relu");
    }

    FusedOperation log(const FusedOperation& x) {
        return elementwise(x, "functor::log");
    }

    FusedOperation log_or_zero(const FusedOperation& x) {
        return elementwise(x, "functor::log_or_zero");
    }

    FusedOperation abs(const FusedOperation& x)  {
        return elementwise(x, "functor::abs");
    }

    FusedOperation sign(const FusedOperation& x) {
        return elementwise(x, "functor::sign");
    }

    FusedOperation square(const FusedOperation& x) {
        return elementwise(x, "functor::square");
    }

    FusedOperation cube(const FusedOperation& x) {
        return elementwise(x, "functor::cube");
    }

    FusedOperation sqrt(const FusedOperation& x) {
        return elementwise(x, "functor::sqrt_f");
    }

    FusedOperation rsqrt(const FusedOperation& x) {
        return elementwise(x, "functor::rsqrt");
    }

    FusedOperation scalar_add(const FusedOperation& x, const double& scalar) {
        return elementwise(x, scalar, "functor::add");
    }
    FusedOperation scalar_sub(const FusedOperation& x, const double& scalar) {
        return elementwise(x, scalar, "functor::sub");
    }
    FusedOperation scalar_mul(const FusedOperation& x, const double& scalar) {
        return elementwise(x, scalar, "functor::eltmul");
    }
    FusedOperation scalar_div(const FusedOperation& x, const double& scalar) {
        return elementwise(x, scalar, "functor::eltdiv");
    }
    FusedOperation scalar_pow(const FusedOperation& x, const double& scalar) {
        return elementwise(x, scalar, "functor::power");
    }

}  // namespace op2
