#ifndef DALI_MATH_LAZY_DOT_H
#define DALI_MATH_LAZY_DOT_H

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int ktype>
    class LazyTensor;
#else
    template<typename LeftType, typename DType, int ktype>
    class LazyTensor;
#endif

#ifdef DALI_USE_CUDA
    template<template <typename, typename, typename, int> class wrapper_t1, template <typename, typename, typename, int> class wrapper_t2, typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb>
    inline auto
    dot(const wrapper_t1<TA, TB, DType, ta> &left,
        const wrapper_t2<TC, TD, DType, tb> &right) -> LazyTensor<
            decltype(dot(left.left, right.left)),
            decltype(dot(left.right, right.right)),
            DType,
            mshadow::expr::type::kComplex> {
        auto cpu_dot = dot(left.left, right.left);
        auto gpu_dot = dot(left.right, right.right);
        auto joined_sts = decltype(left.sync_tensors)(left.sync_tensors);
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end());
        return LazyTensor<
            decltype(cpu_dot),
            decltype(gpu_dot),
            DType,
            mshadow::expr::type::kComplex>(
                cpu_dot,
                gpu_dot,
                joined_sts
            );
    }
#else
    template<template <typename, typename, int> class wrapper_t1, template <typename, typename, int> class wrapper_t2, typename TA, typename TC, typename DType, int ta, int tb>
    inline auto
    dot(const wrapper_t1<TA, DType, ta> &left,
        const wrapper_t2<TC, DType, tb> &right) -> LazyTensor<
            decltype(dot(left.left, right.left)),
            DType,
            mshadow::expr::type::kComplex> {
        auto cpu_dot = dot(left.left, right.left);
        return LazyTensor<
            decltype(cpu_dot),
            DType,
            mshadow::expr::type::kComplex>(
                cpu_dot,
                left.sync_tensors
            );
    }
#endif

#endif
