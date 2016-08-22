#define DALI_DECLARE_FUNCTOR_NAME(NAME) \
    template<typename T>\
    const std::string NAME<T>::name = #NAME\

#define DALI_DECLARE_FUNCTOR_PRETTY_NAME(NAME, PRETTY) \
    template<typename T>\
    const std::string NAME<T>::name = PRETTY\

namespace functor {
    DALI_DECLARE_FUNCTOR_NAME(near_equal);
    DALI_DECLARE_FUNCTOR_NAME(square);
    DALI_DECLARE_FUNCTOR_NAME(cube);
    DALI_DECLARE_FUNCTOR_NAME(eye);
    DALI_DECLARE_FUNCTOR_NAME(fill);
    DALI_DECLARE_FUNCTOR_NAME(arange);
    DALI_DECLARE_FUNCTOR_NAME(add);
    DALI_DECLARE_FUNCTOR_NAME(equals);
    DALI_DECLARE_FUNCTOR_NAME(sub);
    DALI_DECLARE_FUNCTOR_NAME(eltmul);
    DALI_DECLARE_FUNCTOR_NAME(eltdiv);
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(sqrt_f, "square_root");
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(rsqrt, "reciprocal_square_root");
    DALI_DECLARE_FUNCTOR_NAME(inv);
    DALI_DECLARE_FUNCTOR_NAME(sigmoid);
    DALI_DECLARE_FUNCTOR_NAME(identity);
    DALI_DECLARE_FUNCTOR_NAME(log);
    DALI_DECLARE_FUNCTOR_NAME(negative_log);
    DALI_DECLARE_FUNCTOR_NAME(safe_entropy_log);
    DALI_DECLARE_FUNCTOR_NAME(exp);
    DALI_DECLARE_FUNCTOR_NAME(isnotanumber);
    DALI_DECLARE_FUNCTOR_NAME(isinfinity);
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(div_grad, "division_gradient");
    DALI_DECLARE_FUNCTOR_NAME(tanh);
    DALI_DECLARE_FUNCTOR_NAME(inverse_tanh);
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(dtanh, "tanh_gradient");
    DALI_DECLARE_FUNCTOR_NAME(power);
    DALI_DECLARE_FUNCTOR_NAME(abs);
    DALI_DECLARE_FUNCTOR_NAME(log_or_zero);
    DALI_DECLARE_FUNCTOR_NAME(sign);
    DALI_DECLARE_FUNCTOR_NAME(threshold);
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(max_scalar, "max");
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(min_scalar, "min");
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(greaterthanequal, "greater_than_or_equal");
    DALI_DECLARE_FUNCTOR_NAME(binary_cross_entropy);
    DALI_DECLARE_FUNCTOR_NAME(binary_cross_entropy_grad);
    DALI_DECLARE_FUNCTOR_NAME(softplus);
    DALI_DECLARE_FUNCTOR_PRETTY_NAME(softplus_backward, "softplus_gradient");
    DALI_DECLARE_FUNCTOR_NAME(prelu);
    DALI_DECLARE_FUNCTOR_NAME(prelu_backward_weights);
    DALI_DECLARE_FUNCTOR_NAME(prelu_backward_inputs);
    DALI_DECLARE_FUNCTOR_NAME(clipped_relu);
    DALI_DECLARE_FUNCTOR_NAME(steep_sigmoid);
    DALI_DECLARE_FUNCTOR_NAME(clip);
    DALI_DECLARE_FUNCTOR_NAME(relu);
}  // namespace functor
