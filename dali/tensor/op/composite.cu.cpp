#include "composite.h"

#include "dali/array/array.h"
#include "dali/array/op/dot.h"
#include "dali/tensor/tape.h"
#include "dali/tensor/tensor_macros.h"

namespace tensor_ops {
	Tensor quadratic_form(const Tensor& left, const Tensor& middle, const Tensor& right) {
		if (graph::backprop_enabled()) {
			Array left_side_mul = op::dot(left.w.transpose(), middle.w);
			Tensor out(op::dot(left_side_mul, right.w));
			graph::emplace_back([left_side_mul, left, middle, right, out]() mutable {
				MAYBE_GRAD(right) <<= op::dot(left_side_mul.transpose(), out.dw);
				Array LeftT_dot_middle_grad = op::dot(out.dw, right.w.transpose());
				MAYBE_GRAD(left) <<= op::dot(middle.w, LeftT_dot_middle_grad.transpose());
				MAYBE_GRAD(middle) <<= op::dot(left.w, LeftT_dot_middle_grad);
			});
			return out;
		} else {
			return Tensor(
				op::dot(
					op::dot(
						left.w.transpose(),
						middle.w
					),
					right.w
				)
			);
		}
	}
} // namespace tensor_ops
