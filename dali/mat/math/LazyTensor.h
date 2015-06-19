#ifndef DALI_MAT_MATH_LAZY_TENSOR_H
#define DALI_MAT_MATH_LAZY_TENSOR_H

#include <functional>
#include <vector>
#include "mshadow/tensor.h"

template<typename DType>
class SynchronizedTensor;

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int ktype>
    class LazyTensor;

    template<typename LeftType, typename RightType, typename DType, int ktype>
#else
    template<typename LeftType, typename DType, int ktype>
    class LazyTensor;

    template<typename LeftType, typename DType, int ktype>
#endif
class LazyTensorTransposed {
    public:
        typedef std::vector<std::reference_wrapper<SynchronizedTensor<DType>>> sync_tensors_t;
        mshadow::expr::TransposeExp<LeftType, DType>              left;
        sync_tensors_t sync_tensors;

        #ifdef DALI_USE_CUDA
        mshadow::expr::TransposeExp<RightType, DType>            right;
        LazyTensorTransposed(
            const mshadow::expr::TransposeExp<LeftType, DType>& _left,
            const mshadow::expr::TransposeExp<RightType, DType>& _right,
            const sync_tensors_t& st)
            : left(_left), right(_right), sync_tensors({st}) {}

        inline LazyTensor<LeftType, RightType, DType, ktype> T(void) const;

        #else
        LazyTensorTransposed(
            const mshadow::expr::TransposeExp<LeftType, DType>& _left,
            const sync_tensors_t& st)
            : left(_left), sync_tensors({st}) {}

        inline LazyTensor<LeftType, DType, ktype> T(void) const;

        #endif
};

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int ktype>
#else
    template<typename LeftType, typename DType, int ktype>
#endif
class LazyTensor {
    public:
        typedef std::vector<std::reference_wrapper<SynchronizedTensor<DType>>> sync_tensors_t;
        LeftType               left;
        sync_tensors_t sync_tensors;

        #ifdef DALI_USE_CUDA
            RightType right;
            LazyTensor(
                const LeftType& _left,
                const RightType& _right,
                const sync_tensors_t& _sync_tensors)
                : left(_left), right(_right), sync_tensors(_sync_tensors) {}

            LazyTensor(std::reference_wrapper<SynchronizedTensor<DType>> st)
                : left(st.get().mem_cpu), right(st.get().mem_gpu), sync_tensors({st}) {}

            inline LazyTensorTransposed<LeftType, RightType, DType, ktype> T(void) const {
                auto cpu_T = left.T();
                auto gpu_T = right.T();
                return LazyTensorTransposed<LeftType, RightType, DType, ktype>(cpu_T, gpu_T, sync_tensors);
            }
        #else
            LazyTensor(
                const LeftType& _left,
                const sync_tensors_t& _sync_tensors)
                : left(_left), sync_tensors(_sync_tensors) {}

            LazyTensor(std::reference_wrapper<SynchronizedTensor<DType>> st)
                : left(st.get().mem_cpu), sync_tensors({st}) {}

            inline LazyTensorTransposed<LeftType, DType, ktype> T(void) const {
                auto cpu_T = left.T();
                return LazyTensorTransposed<LeftType, DType, ktype>(cpu_T, sync_tensors);
            }
        #endif
};

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int ktype>
    inline LazyTensor<LeftType, RightType, DType, ktype> LazyTensorTransposed<LeftType, RightType, DType, ktype>::T(void) const {
        auto cpu_T = left.T();
        auto gpu_T = right.T();
        return LazyTensor<decltype(cpu_T), decltype(gpu_T), DType, ktype>(cpu_T, gpu_T, sync_tensors);
    }
#else
    template<typename LeftType, typename DType, int ktype>
    inline LazyTensor<LeftType, DType, ktype> LazyTensorTransposed<LeftType, DType, ktype>::T(void) const {
        auto cpu_T = left.T();
        return LazyTensor<decltype(cpu_T), DType, ktype>(cpu_T, sync_tensors);
    }
#endif

#ifdef DALI_USE_CUDA
    #define BINARY_OP(opname, opsymbol) \
    template<template <typename, typename, typename, int> class wrapper_t1, template <typename, typename, typename, int> class wrapper_t2, typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb> \
    LazyTensor< mshadow::expr::BinaryMapExp<opname, TA, TC, DType, (ta|tb|mshadow::expr::type::kMapper)>, mshadow::expr::BinaryMapExp<opname, TB, TD, DType, (ta|tb|mshadow::expr::type::kMapper)>, DType, (ta|tb|mshadow::expr::type::kMapper)> operator opsymbol( \
            const wrapper_t1<TA, TB, DType, ta> &left, \
            const wrapper_t2<TC, TD, DType, tb> &right) { \
        const auto& l_cpu = left.left; \
        const auto& r_cpu = right.left; \
        auto res_cpu = l_cpu opsymbol r_cpu; \
        const auto& l_gpu = left.right; \
        const auto& r_gpu = right.right; \
        auto res_gpu = l_gpu opsymbol r_gpu; \
        auto joined_sts = std::vector<std::reference_wrapper<SynchronizedTensor<DType>>>(left.sync_tensors); \
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end()); \
        return LazyTensor<decltype(res_cpu), decltype(res_gpu), DType, (ta|tb|mshadow::expr::type::kMapper)>(res_cpu, res_gpu, joined_sts); \
    }
#else
    #define BINARY_OP(opname, opsymbol) \
    template<template <typename, typename, int> class wrapper_t1, template <typename, typename, int> class wrapper_t2, typename TA, typename TC, typename DType, int ta, int tb> \
    LazyTensor< mshadow::expr::BinaryMapExp<opname, TA, TC, DType, (ta|tb|mshadow::expr::type::kMapper)>, DType, (ta|tb|mshadow::expr::type::kMapper)> operator opsymbol( \
            const wrapper_t1<TA, DType, ta> &left, \
            const wrapper_t2<TC, DType, tb> &right) { \
        const auto& l_cpu = left.left; \
        const auto& r_cpu = right.left; \
        auto res_cpu = l_cpu opsymbol r_cpu; \
        auto joined_sts = std::vector<std::reference_wrapper<SynchronizedTensor<DType>>>(left.sync_tensors); \
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end()); \
        return LazyTensor<decltype(res_cpu), DType, (ta|tb|mshadow::expr::type::kMapper)>(res_cpu, joined_sts); \
    }
#endif


BINARY_OP(mshadow::op::plus,  +);
BINARY_OP(mshadow::op::mul,   *);
BINARY_OP(mshadow::op::minus, -);

#ifdef DALI_USE_CUDA
    template<typename OP, template <typename, typename, typename, int> class wrapper_t, typename TA, typename TB, typename DType, int ta>
    inline wrapper_t<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>, mshadow::expr::UnaryMapExp<OP, TB, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)>
    MakeExp(const wrapper_t<TA, TB, DType, ta> &src) {
        auto unary_l = mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>(src.left);
        auto unary_r = mshadow::expr::UnaryMapExp<OP, TB, DType, (ta|mshadow::expr::type::kMapper)>(src.right);
        return wrapper_t<decltype(unary_l), decltype(unary_r), DType, (ta|mshadow::expr::type::kMapper)>(unary_l, unary_r, src.sync_tensors);;
    }

    template<typename OP, template <typename, typename, typename, int> class wrapper_t, typename TA, typename TB, typename DType, int ta>
    inline wrapper_t<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>, mshadow::expr::UnaryMapExp<OP, TB, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)>
    F(const wrapper_t<TA, TB, DType, ta> &src) {
        return MakeExp<OP>(src);
    }
#else
    template<typename OP, template <typename, typename, int> class wrapper_t, typename TA, typename DType, int ta>
    inline wrapper_t<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)>
    MakeExp(const wrapper_t<TA, DType, ta> &src) {
        auto unary_l = mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>(src.left);
        return wrapper_t<decltype(unary_l), DType, (ta|mshadow::expr::type::kMapper)>(unary_l, src.sync_tensors);;
    }

    template<typename OP, template <typename, typename, int> class wrapper_t, typename TA, typename DType, int ta>
    inline wrapper_t<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)>
    F(const wrapper_t<TA, DType, ta> &src) {
        return MakeExp<OP>(src);
    }
#endif

#endif
