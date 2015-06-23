#ifndef DALI_MATH_LAZY_DOT_H
#define DALI_MATH_LAZY_DOT_H

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int dimension, int ktype>
    class LazyTensor;
#else
    template<typename LeftType, typename DType, int dimension, int ktype>
    class LazyTensor;
#endif

#ifdef DALI_USE_CUDA
    template<typename TA, typename TB, typename TC, typename TD, typename DType, int dimension, int ta, int tb>
    inline auto
    dot(const LazyTensor<TA, TB, DType, dimension, ta> &left,
        const LazyTensor<TC, TD, DType, dimension, tb> &right) -> LazyTensor<
            decltype(dot(left.left, right.left)),
            decltype(dot(left.right, right.right)),
            DType,
            dimension,
            mshadow::expr::type::kComplex> {
        auto cpu_dot = dot(left.left, right.left);
        auto gpu_dot = dot(left.right, right.right);
        auto joined_sts = decltype(left.sync_tensors)(left.sync_tensors);
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end());
        auto joined_dts = decltype(left.dependent_tensors)(left.dependent_tensors);
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end());
        return LazyTensor<
            decltype(cpu_dot),
            decltype(gpu_dot),
            DType,
            dimension,
            mshadow::expr::type::kComplex>(
                cpu_dot,
                gpu_dot,
                joined_sts,
                joined_dts
            );
    }
#else
    template<typename TA, typename TC, typename DType, int dimension, int ta, int tb>
    inline auto
    dot(const LazyTensor<TA, DType, dimension, ta> &left,
        const LazyTensor<TC, DType, dimension, tb> &right) -> LazyTensor<
            decltype(dot(left.left, right.left)),
            DType,
            dimension,
            mshadow::expr::type::kComplex> {
        auto cpu_dot = dot(left.left, right.left);
        auto joined_sts = decltype(left.sync_tensors)(left.sync_tensors);
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end());
        auto joined_dts = decltype(left.dependent_tensors)(left.dependent_tensors);
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end());
        return LazyTensor<
            decltype(cpu_dot),
            DType,
            dimension,
            mshadow::expr::type::kComplex>(
                cpu_dot,
                joined_sts,
                joined_dts
            );
    }
#endif

#endif
