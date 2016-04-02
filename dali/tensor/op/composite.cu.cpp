#include "dali/tensor/op/composite.h"

#include "dali/tensor/__MatMacros__.h"
#include "dali/array/TensorOps.h"
#include "dali/array/LazyTensor.h"
#include "dali/tensor/Weights.h"

using std::vector;
using utils::MS;

namespace matops {
    template<typename R>
    Mat<R> Composite<R>::quadratic_form(
            Mat<R> left,
            Mat<R> middle,
            Mat<R> right) {
        ASSERT2(middle.dims(1) == right.dims(0), "Quadratic form right matrix has wrong dimensions.");
        ASSERT2(left.dims(0) == middle.dims(0) , "Quadratic form left matrix has wrong dimensions.");
        Mat<R> out (left.dims(1), right.dims(1), weights<R>::empty());
        if (graph::backprop_enabled()) {
            TensorInternal<R,2> left_side_mul(mshadow::Shape2(left.dims(1), middle.dims(1)));
            // left_side_mul = left.T * middle;
            left_side_mul = dot(MAT(left).wrapper().T(), MAT(middle).wrapper());
            // out = left_side_mul * right;
            MAT(out) = dot(left_side_mul.wrapper(), MAT(right).wrapper());
            graph::emplace_back([left_side_mul, left, middle, right, out]() mutable {
                // right.dw() += left_side_mul.T() * out.dw();
                SAFE_GRAD(right) += dot(left_side_mul.wrapper().T(), GRAD(out).wrapper());
                TensorInternal<R,2> LeftT_dot_middle_grad(mshadow::Shape2(left.dims(1), right.dims(0)));
                // leftT_dot_middle_grad = out.dw() * right.T()
                LeftT_dot_middle_grad = dot(GRAD(out).wrapper(), MAT(right).wrapper().T());
                // left.dw() += leftT_dot_middle_grad * middle.T()
                SAFE_GRAD(left) += dot(MAT(middle).wrapper(), LeftT_dot_middle_grad.wrapper().T());
                // middle.dw() += left * LeftT_dot_middle_grad
                SAFE_GRAD(middle) += dot(MAT(left).wrapper(), LeftT_dot_middle_grad.wrapper());
            });
        } else {
            TensorInternal<R,2> left_side_mul(mshadow::Shape2(left.dims(1), middle.dims(1)));
            left_side_mul = dot(MAT(left).wrapper().T(), MAT(middle).wrapper());
            MAT(out) = dot(left_side_mul.wrapper(), MAT(right).wrapper());
        }
        return out;
    }

    template<typename R>
    Mat<R> Composite<R>::mul_with_bias(
            Mat<R> matrix1,
            Mat<R> matrix2,
            Mat<R> bias) {
        return mul_add_mul_with_bias({matrix1}, {matrix2}, bias);
    }


    template<typename R>
    Mat<R> Composite<R>::mul_add_mul_with_bias(std::initializer_list<Mat<R>> weight_mats,
                                               std::initializer_list<Mat<R>> inputs,
                                               Mat<R> bias) {
        vector<Mat<R>> weights_v(weight_mats);
        vector<Mat<R>> inputs_v(inputs);
        return mul_add_mul_with_bias(weights_v, inputs_v, bias);
    }



    template<typename R>
    Mat<R> Composite<R>::mul_add_mul_with_bias_colwise(const vector<Mat<R>>& weight_mats,
                                               const vector<Mat<R>>& inputs,
                                               Mat<R> bias) {

        ASSERT2(weight_mats.size() == inputs.size(),
                "Different number of weights and inputs passed to mul_add_mul_with_bias");
        // broacast to largest number of examples
        dim_t max_num_examples = 0;
        for (auto input : inputs) {
            max_num_examples = std::max(max_num_examples, input.dims(1));
        }

        Mat<R> out(weight_mats[0].dims(0), max_num_examples, weights<R>::empty());
        MAT(out) = MAT(bias).ravel().wrapper().template broadcast<0>(MAT(out).shape);

        for (int i = 0; i < weight_mats.size(); ++i) {
            // inputs must either match the broadcasted size, or be broadcastable by having their
            // outer dimension be 1 (a column vector essentially)
            ASSERT2((inputs[i].dims(1) == max_num_examples) || (inputs[i].dims(1) == 1),
                    MS() << "incorrect outer dimension for input " << i);
            ASSERT2(inputs[i].dims(0) == weight_mats[i].dims(1),
                    MS() << "Disagreement on inner dimension on input pair " << i);

            if (inputs[i].dims(1) == max_num_examples) {
                MAT(out) += dot(MAT(weight_mats[i]).wrapper(), MAT(inputs[i]).wrapper());
            } else {
                TensorInternal<R, 2> temp(mshadow::Shape2(weight_mats[i].dims(0), 1));
                temp = dot(MAT(weight_mats[i]).wrapper(), MAT(inputs[i]).wrapper());
                MAT(out) += temp.ravel().wrapper().template broadcast<0>(MAT(out).shape);
            }

            DEBUG_ASSERT_MAT_NOT_NAN(out)
        }

        if (graph::backprop_enabled())
            graph::emplace_back([weight_mats, inputs, bias, out, max_num_examples]() mutable {

                for (int i = 0; i < weight_mats.size(); ++i) {
                    if (inputs[i].dims(1) == max_num_examples) {
                        SAFE_GRAD(weight_mats[i]) += dot(GRAD(out).wrapper(),
                                                         MAT(inputs[i]).wrapper().T());
                        SAFE_GRAD(inputs[i]) += dot(MAT(weight_mats[i]).wrapper().T(),
                                                    GRAD(out).wrapper());
                    } else {
                        // broadcasting input means taking outer product here:
                        {
                            TensorInternal<R, 2> temp(mshadow::Shape2(1, out.dims(0)));
                            temp[0] = sum_cols(GRAD(out).wrapper());
                            SAFE_GRAD(weight_mats[i]) += dot(
                                temp.wrapper().T(), MAT(inputs[i]).wrapper().T()
                            );
                        }
                        // broadcasting output means sum after the reverse product here:
                        {
                            TensorInternal<R, 2> temp(mshadow::Shape2(weight_mats[i].dims(1), max_num_examples));
                            temp = dot(MAT(weight_mats[i]).wrapper().T(), GRAD(out).wrapper());
                            SAFE_GRAD(inputs[i]).ravel() += sum_cols(temp.wrapper());
                        }
                    }
                }
                SAFE_GRAD(bias).ravel() += sum_cols(GRAD(out).wrapper());
            });

        return out;
    }


    template<typename R>
    Mat<R> Composite<R>::mul_add_mul_with_bias(const vector<Mat<R>>& weight_mats,
                                               const vector<Mat<R>>& inputs,
                                               Mat<R> bias) {
        ASSERT2(weight_mats.size() == inputs.size(),
                "Different number of weights and inputs passed to mul_add_mul_with_bias");
        // broacast to largest number of examples
        dim_t max_num_examples = 0;
        for (auto input : inputs) {
            max_num_examples = std::max(max_num_examples, input.dims(0));
        }

        Mat<R> out(max_num_examples, weight_mats[0].dims(1), weights<R>::empty());
        MAT(out) = MAT(bias).ravel().wrapper().template broadcast<1>(MAT(out).shape);

        for (int i = 0; i < weight_mats.size(); ++i) {
            // inputs must either match the broadcasted size, or be broadcastable by having their
            // outer dimension be 1 (a column vector essentially)
            ASSERT2((inputs[i].dims(0) == max_num_examples) || (inputs[i].dims(0) == 1),
                    MS() << "incorrect outer dimension for input " << i);
            ASSERT2(inputs[i].dims(1) == weight_mats[i].dims(0),
                    MS() << "Disagreement on inner dimension on input pair " << i);


            if (inputs[i].dims(0) == max_num_examples) {
                MAT(out) += dot(MAT(inputs[i]).wrapper(), MAT(weight_mats[i]).wrapper());
            } else {
                TensorInternal<R, 2> temp(mshadow::Shape2(1, weight_mats[i].dims(1)));

                temp = dot(MAT(inputs[i]).wrapper(), MAT(weight_mats[i]).wrapper());

                MAT(out) += temp.ravel().wrapper().template broadcast<1>(MAT(out).shape);
            }

            DEBUG_ASSERT_MAT_NOT_NAN(out)
        }

        if (graph::backprop_enabled())
            graph::emplace_back([weight_mats, inputs, bias, out, max_num_examples]() mutable {

                for (int i = 0; i < weight_mats.size(); ++i) {

                    if (inputs[i].dims(0) == max_num_examples) {
                        SAFE_GRAD(inputs[i]) += dot(GRAD(out).wrapper(),
                                                    MAT(weight_mats[i]).wrapper().T());

                        SAFE_GRAD(weight_mats[i]) += dot(MAT(inputs[i]).wrapper().T(),
                                                         GRAD(out).wrapper());
                    } else {
                        TensorInternal<R, 2> temp(mshadow::Shape2(1, out.dims(1)));
                        temp[0] = sum_rows(GRAD(out).wrapper());


                        SAFE_GRAD(inputs[i]) += dot(
                            temp.wrapper(), MAT(weight_mats[i]).wrapper().T()
                        );

                        SAFE_GRAD(weight_mats[i]) += dot(MAT(inputs[i]).wrapper().T(), temp.wrapper());
                    }
                }
                SAFE_GRAD(bias).ravel() += sum_rows(GRAD(out).wrapper());
            });

        return out;
    }


    template class Composite<float>;
    template class Composite<double>;
    template class Composite<int>;
}
