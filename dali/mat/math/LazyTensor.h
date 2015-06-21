#ifndef DALI_MAT_MATH_LAZY_TENSOR_H
#define DALI_MAT_MATH_LAZY_TENSOR_H

#include <functional>
#include <vector>

#include "mshadow/tensor.h"

#include "dali/mat/math/LazySoftmax.h"
#include "dali/mat/math/LazyUtils.h"
#include "dali/mat/math/LazyPluck.h"

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

            inline LazyTensor<dali_expr::SoftmaxExpression<LeftType, DType>, dali_expr::SoftmaxExpression<RightType, DType>, DType, (ktype|mshadow::expr::type::kComplex)> softmax(void) const {
                auto cpu_soft = dali_expr::SoftmaxExpression<LeftType, DType>(left);
                auto gpu_soft = dali_expr::SoftmaxExpression<RightType, DType>(right);
                return LazyTensor<
                    decltype(cpu_soft),
                    decltype(gpu_soft),
                    DType,
                    (ktype|mshadow::expr::type::kComplex)
                    >(
                        cpu_soft,
                        gpu_soft,
                        sync_tensors
                    );
            }

            /* Future Lazy plucking
            inline LazyTensor<
                    dali_expr::PluckExpression<
                        LeftType,
                        DType,
                        mshadow::expr::ExpInfo<LeftType>::kDim - 1>,
                    dali_expr::PluckExpression<
                        RightType,
                        DType,
                        mshadow::expr::ExpInfo<RightType>::kDim - 1>,
                    DType,
                    mshadow::expr::type::kChainer
                    > operator[](mshadow::index_t idx) const {
                auto cpu_pluck = dali_expr::PluckExpression<LeftType, DType, mshadow::expr::ExpInfo<LeftType>::kDim - 1>(
                    left,
                    idx);
                auto gpu_pluck = dali_expr::PluckExpression<RightType, DType, mshadow::expr::ExpInfo<RightType>::kDim - 1>(
                    right,
                    idx);
            */

            inline LazyTensor<
                mshadow::Tensor<
                    typename extract_tensor_arguments<LeftType>::device_t,
                    extract_tensor_arguments<LeftType>::subdim,
                    DType >,
                mshadow::Tensor<
                    typename extract_tensor_arguments<RightType>::device_t,
                    extract_tensor_arguments<RightType>::subdim,
                    DType >, DType, ktype> operator[](mshadow::index_t idx) const {

                auto cpu_pluck = left[idx];
                auto gpu_pluck = right[idx];

                return LazyTensor<decltype(cpu_pluck), decltype(gpu_pluck), DType, ktype>(
                    cpu_pluck,
                    gpu_pluck, sync_tensors
                );
            }

            // Expression that replicate a 1 dimension tensor in
            // dimension dimcast
            template<int dimcast, int dimdst>
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, dimdst, dimdst - dimcast>,
                mshadow::expr::Broadcast1DExp<RightType, DType, dimdst, dimdst - dimcast>,
                DType,
                ktype >broadcast(mshadow::Shape<dimdst> shape) const {

                auto cpu_broad = mshadow::expr::broadcast<dimcast>(left, shape);
                auto gpu_broad = mshadow::expr::broadcast<dimcast>(right, shape);

                return LazyTensor<decltype(cpu_broad), decltype(gpu_broad), DType, ktype>(
                    cpu_broad,
                    gpu_broad, sync_tensors
                );
            }

            // Expression that replicate a 1 dimension tensor for
            // nrow times
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, 2, 1>,
                mshadow::expr::Broadcast1DExp<RightType, DType, 2, 1>,
                DType,
                ktype >
            repmat(mshadow::index_t nrow) {
                return broadcast<1>(
                    mshadow::Shape2(
                        nrow,
                        mshadow::expr::ShapeCheck<1, LeftType>::Check(
                            left.self()
                        )[0]
                    )
                );
            }
        #else
            // Same expression with the gpu twin
            // ignored.
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

            inline LazyTensor<dali_expr::SoftmaxExpression<LeftType, DType>,
                              DType,
                              (ktype|mshadow::expr::type::kComplex)
                              > softmax(void) const {
                auto cpu_soft = dali_expr::SoftmaxExpression<LeftType, DType>(left);
                return LazyTensor<
                    decltype(cpu_soft),
                    DType,
                    (ktype|mshadow::expr::type::kComplex)
                    >(
                        cpu_soft,
                        sync_tensors
                    );
            }

            inline LazyTensor<
                mshadow::Tensor<
                    typename extract_tensor_arguments<LeftType>::device_t,
                    extract_tensor_arguments<LeftType>::subdim,
                    DType >, DType, ktype> operator[](mshadow::index_t idx) const {

                auto cpu_pluck = left[idx];
                return LazyTensor<decltype(cpu_pluck), DType, ktype>(
                    cpu_pluck, sync_tensors
                );
            }

            // Expression that replicate a 1 dimension tensor in
            // dimension dimcast
            template<int dimcast, int dimdst>
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, dimdst, dimdst - dimcast>,
                DType,
                ktype >broadcast(mshadow::Shape<dimdst> shape) const {
                auto cpu_broad = mshadow::expr::broadcast<dimcast>(left, shape);
                return LazyTensor<decltype(cpu_broad), DType, ktype>(
                    cpu_broad, sync_tensors
                );
            }

            // Expression that replicate a 1 dimension tensor for
            // nrow times
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, 2, 1>,
                DType,
                ktype >
            repmat(mshadow::index_t nrow) {
                return broadcast<1>(
                    mshadow::Shape2(
                        nrow,
                        mshadow::expr::ShapeCheck<1, LeftType>::Check(
                            left.self()
                        )[0]
                    )
                );
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
    template<typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb>
    inline LazyTensor<
        mshadow::expr::DotExp<TA, TC, false, false, DType>,
        mshadow::expr::DotExp<TB, TD, false, false, DType>,
        DType,
        mshadow::expr::type::kComplex>
    dot(
        const LazyTensor<TA, TB, DType, ta> &left,
        const LazyTensor<TC, TD, DType, tb> &right) {
        auto cpu_dot = dot(left.left, right.left);
        auto gpu_dot = dot(left.right, right.right);
        auto joined_sts = std::vector<std::reference_wrapper<SynchronizedTensor<DType>>>(left.sync_tensors);
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
    template<typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb>
    inline LazyTensor<
        mshadow::expr::DotExp<TA, TC, true, false, DType>,
        mshadow::expr::DotExp<TB, TD, true, false, DType>,
        DType,
        mshadow::expr::type::kComplex>
    dot(
        const LazyTensorTransposed<TA, TB, DType, ta> &left,
        const LazyTensor<TC, TD, DType, tb> &right) {
        auto cpu_dot = dot(left.left, right.left);
        auto gpu_dot = dot(left.right, right.right);
        auto joined_sts = std::vector<std::reference_wrapper<SynchronizedTensor<DType>>>(left.sync_tensors);
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
    template<typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb>
    inline LazyTensor<
        mshadow::expr::DotExp<TA, TC, false, true, DType>,
        mshadow::expr::DotExp<TB, TD, false, true, DType>,
        DType,
        mshadow::expr::type::kComplex>
    dot(
        const LazyTensor<TA, TB, DType, ta> &left,
        const LazyTensorTransposed<TC, TD, DType, tb> &right) {
        auto cpu_dot = dot(left.left, right.left);
        auto gpu_dot = dot(left.right, right.right);
        auto joined_sts = std::vector<std::reference_wrapper<SynchronizedTensor<DType>>>(left.sync_tensors);
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
    template<typename TA, typename TB, typename TC, typename TD, typename DType, int ta, int tb>
    inline LazyTensor<
        mshadow::expr::DotExp<TA, TC, true, true, DType>,
        mshadow::expr::DotExp<TB, TD, true, true, DType>,
        DType,
        mshadow::expr::type::kComplex>
    dot(
        const LazyTensorTransposed<TA, TB, DType, ta> &left,
        const LazyTensorTransposed<TC, TD, DType, tb> &right) {
        auto cpu_dot = dot(left.left, right.left);
        auto gpu_dot = dot(left.right, right.right);
        auto joined_sts = std::vector<std::reference_wrapper<SynchronizedTensor<DType>>>(left.sync_tensors);
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
#endif
// /*! \brief dot operator def */
// template<typename TA, typename TB, typename DType>
// inline DotExp<TA, TB, true, false, DType>
// dot(const TransposeExp<TA, DType> &lhs, const RValueExp<TB, DType> &rhs) {
//   return DotExp<TA, TB, true, false, DType>(lhs.exp, rhs.self(), 1.0f);
// }
// /*! \brief dot operator def */
// template<typename TA, typename TB, typename DType>
// inline DotExp<TA, TB, false, true, DType>
// dot(const RValueExp<TA, DType> &lhs, const TransposeExp<TB, DType> &rhs) {
//   return DotExp<TA, TB, false, true, DType>(lhs.self(), rhs.exp, 1.0f);
// }
// /*! \brief dot operator def */
// template<typename TA, typename TB, typename DType>
// inline DotExp<TA, TB, true, true, DType>
// dot(const TransposeExp<TA, DType> &lhs, const TransposeExp<TB, DType> &rhs) {
//   return DotExp<TA, TB, true, true, DType>(lhs.exp, rhs.exp, 1.0f);
// }

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

    #define BINARY_SCALAR_OP(opname, opsymbol) \
    template<template <typename, typename, typename, int> class wrapper_t1, typename TA, typename TB, typename DType, int ta> \
    LazyTensor< mshadow::expr::BinaryMapExp<opname, TA, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, mshadow::expr::BinaryMapExp<opname, TB, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)> operator opsymbol( \
            const wrapper_t1<TA, TB, DType, ta> &left, \
            const mshadow::expr::ScalarExp<DType> &right) { \
        const auto& l_cpu = left.left; \
        auto res_cpu = l_cpu opsymbol right; \
        const auto& l_gpu = left.right; \
        auto res_gpu = l_gpu opsymbol right; \
        return LazyTensor<decltype(res_cpu), decltype(res_gpu), DType, (ta|mshadow::expr::type::kMapper)>(res_cpu, res_gpu, left.sync_tensors); \
    } \
    \
    template<template <typename, typename, typename, int> class wrapper_t1, typename TA, typename TB, typename DType, int ta> \
    LazyTensor< mshadow::expr::BinaryMapExp<opname, TA, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, mshadow::expr::BinaryMapExp<opname, TB, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)> operator opsymbol( \
            const mshadow::expr::ScalarExp<DType> &left, \
            const wrapper_t1<TA, TB, DType, ta> &right) { \
        const auto& l_cpu = right.left; \
        auto res_cpu = left opsymbol l_cpu; \
        const auto& l_gpu = right.right; \
        auto res_gpu = left opsymbol l_gpu; \
        return LazyTensor<decltype(res_cpu), decltype(res_gpu), DType, (ta|mshadow::expr::type::kMapper)>(res_cpu, res_gpu, left.sync_tensors); \
    } \
    template<template <typename, typename, typename, int> class wrapper_t1, typename TA, typename TB, typename DType, int ta> \
    LazyTensor< mshadow::expr::BinaryMapExp<opname, TA, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, mshadow::expr::BinaryMapExp<opname, TB, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)> operator opsymbol( \
            const wrapper_t1<TA, TB, DType, ta> &left, \
            DType right) { \
        const auto& l_cpu = left.left; \
        auto res_cpu = l_cpu opsymbol mshadow::expr::ScalarExp<DType>(right); \
        const auto& l_gpu = left.right; \
        auto res_gpu = l_gpu opsymbol mshadow::expr::ScalarExp<DType>(right); \
        return LazyTensor<decltype(res_cpu), decltype(res_gpu), DType, (ta|mshadow::expr::type::kMapper)>(res_cpu, res_gpu, left.sync_tensors); \
    } \
    \
    template<template <typename, typename, typename, int> class wrapper_t1, typename TA, typename TB, typename DType, int ta> \
    LazyTensor< mshadow::expr::BinaryMapExp<opname, TA, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, mshadow::expr::BinaryMapExp<opname, TB, mshadow::expr::ScalarExp<DType>, DType, (ta|mshadow::expr::type::kMapper)>, DType, (ta|mshadow::expr::type::kMapper)> operator opsymbol( \
            DType left, \
            const wrapper_t1<TA, TB, DType, ta> &right) { \
        const auto& l_cpu = right.left; \
        auto res_cpu = mshadow::expr::ScalarExp<DType>(left) opsymbol l_cpu; \
        const auto& l_gpu = right.right; \
        auto res_gpu = mshadow::expr::ScalarExp<DType>(left) opsymbol l_gpu; \
        return LazyTensor<decltype(res_cpu), decltype(res_gpu), DType, (ta|mshadow::expr::type::kMapper)>(res_cpu, res_gpu, left.sync_tensors); \
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

BINARY_SCALAR_OP(mshadow::op::plus,  +);
BINARY_SCALAR_OP(mshadow::op::mul,  *);
BINARY_SCALAR_OP(mshadow::op::minus,  -);
BINARY_SCALAR_OP(mshadow::op::div,  /);


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
