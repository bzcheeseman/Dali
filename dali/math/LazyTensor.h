#ifndef DALI_MAT_MATH_LAZY_TENSOR_H
#define DALI_MAT_MATH_LAZY_TENSOR_H

#include <functional>
#include <vector>

#include "mshadow/extension/reduceto1d.h"
#include "mshadow/tensor.h"


#include "dali/math/LazySoftmax.h"
#include "dali/math/LazyUtils.h"
#include "dali/math/LazyPluck.h"
#include "dali/math/LazyDot.h"

template<typename DType, int dimension>
class TensorInternal;

class MemoryMover;

typedef std::vector<const MemoryMover*> sync_tensors_t;

class DormantTensor {
    public:
        // call this method with a new tensor that should be placed here.
        virtual void update_tensor() = 0;
};
typedef std::vector<DormantTensor*> dependent_tensors_t;

#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int dimension, int ktype>
#else
    template<typename LeftType, typename DType, int dimension, int ktype>
#endif
class LazyTensor : DormantTensor {
    public:
        typedef mshadow::Tensor<mshadow::cpu, dimension, DType> cpu_tensor_t;

        // store list of dependant tensors whose memory
        // may need to be refreshed before evaluation
        dependent_tensors_t dependent_tensors;

        LeftType               left;
        // for storing where the tensor came from
        // if it was constructed from TensorInternal
        cpu_tensor_t*   left_source;
        sync_tensors_t sync_tensors;




        #ifdef DALI_USE_CUDA
            typedef mshadow::Tensor<mshadow::gpu, dimension, DType> gpu_tensor_t;
            gpu_tensor_t*  right_source;
            RightType right;
            LazyTensor(
                const LeftType& _left,
                const RightType& _right,
                const sync_tensors_t& _sync_tensors,
                const dependent_tensors_t& _dependent_tensors)
                : left(_left), right(_right), sync_tensors(_sync_tensors),
                dependent_tensors(_dependent_tensors) {}
        #else
            // Same expression with the gpu twin
            // ignored.
            LazyTensor(
                const LeftType& _left,
                const sync_tensors_t& _sync_tensors,
                const dependent_tensors_t& _dependent_tensors)
                : left(_left), sync_tensors(_sync_tensors),
                  dependent_tensors(_dependent_tensors) {}
        #endif

        #ifdef DALI_USE_CUDA
            LazyTensor(const TensorInternal<DType,dimension>& st)
                : left(st.mem_cpu),
                  right(st.mem_gpu),
                  sync_tensors({&st}),
                  left_source(&st.mem_cpu),
                  right_source(&st.mem_gpu),
                  // adding this LazyTensor to dependencies that
                  // need their memory refreshed on eval
                  dependent_tensors({this}) {}
        #else
            LazyTensor(const TensorInternal<DType,dimension>& st)
                : left(st.mem_cpu), sync_tensors({&st}),
                  dependent_tensors({this}),
                  left_source(&st.mem_cpu) {}
        #endif
            // replace the tensors used in this LazyTensor
            // by a copy with fresh memory before evaluation.
            virtual void update_tensor() override {
                assert(left_source != NULL);
                auto old_left_ptr    = (cpu_tensor_t*)(&left);
                *old_left_ptr        = *left_source;
            #ifdef DALI_USE_CUDA
                assert(right_source != NULL);
                auto old_right_ptr    = (gpu_tensor_t*)(&right);
                *old_right_ptr        = *right_source;
            #endif
            }

        #ifdef DALI_USE_CUDA
            inline auto T(void) const -> LazyTensor<decltype(left.T()), decltype(right.T()), DType, dimension, ktype> {
                return LazyTensor<decltype(left.T()), decltype(right.T()), DType, dimension, ktype>(
                        left.T(), right.T(), sync_tensors, dependent_tensors);
            }
        #else
            inline auto T(void) const -> LazyTensor<decltype(left.T()), DType, dimension, ktype> {
                return LazyTensor<decltype(left.T()), DType, dimension, ktype>(
                        left.T(), sync_tensors, dependent_tensors);
            }
        #endif

        #ifdef DALI_USE_CUDA
            inline LazyTensor<dali_expr::SoftmaxExpression<LeftType, DType>,
                              dali_expr::SoftmaxExpression<RightType, DType>,
                              DType,
                              dimension,
                              (ktype|mshadow::expr::type::kComplex)> softmax(void) const {
                auto cpu_soft = dali_expr::SoftmaxExpression<LeftType, DType>(left);
                auto gpu_soft = dali_expr::SoftmaxExpression<RightType, DType>(right);
                return LazyTensor<
                    decltype(cpu_soft), decltype(gpu_soft),
                    DType, dimension,
                    (ktype|mshadow::expr::type::kComplex)>(
                        cpu_soft, gpu_soft, sync_tensors, dependent_tensors);
            }
        #else
            inline LazyTensor<dali_expr::SoftmaxExpression<LeftType, DType>,
                              DType, dimension,
                              (ktype|mshadow::expr::type::kComplex)
                              > softmax(void) const {
                auto cpu_soft = dali_expr::SoftmaxExpression<LeftType, DType>(left);
                return LazyTensor<
                    decltype(cpu_soft),
                    DType, dimension,
                    (ktype|mshadow::expr::type::kComplex)
                    >(cpu_soft, sync_tensors, dependent_tensors);
            }
        #endif

        #ifdef DALI_USE_CUDA
            inline LazyTensor<
                mshadow::Tensor<
                    typename extract_tensor_arguments<LeftType>::device_t,
                    extract_tensor_arguments<LeftType>::subdim,
                    DType >,
                mshadow::Tensor<
                    typename extract_tensor_arguments<RightType>::device_t,
                    extract_tensor_arguments<RightType>::subdim,
                    DType >,
                    DType,
                    extract_tensor_arguments<LeftType>::subdim,
                    ktype> operator[](mshadow::index_t idx) const {
                return LazyTensor<decltype(left[idx]), decltype(right[idx]), DType, dimension - 1, ktype>(
                    left[idx], right[idx],
                    sync_tensors,
                    dependent_tensors
                );
            }
        #else
            inline LazyTensor<
                mshadow::Tensor<
                    typename extract_tensor_arguments<LeftType>::device_t,
                    extract_tensor_arguments<LeftType>::subdim,
                    DType >, DType, dimension - 1, ktype> operator[](mshadow::index_t idx) const {
                auto cpu_pluck = left[idx];
                return LazyTensor<decltype(cpu_pluck), DType, dimension -1, ktype>(
                    cpu_pluck, sync_tensors, dependent_tensors
                );
            }
        #endif

        #ifdef DALI_USE_CUDA
            // Expression that replicate a 1 dimension tensor in
            // dimension dimcast
            template<int dimcast, int dimdst>
            inline auto broadcast(mshadow::Shape<dimdst> shape) ->
                    LazyTensor<decltype(mshadow::expr::broadcast<dimcast>(left, shape)),
                               decltype(mshadow::expr::broadcast<dimcast>(right, shape)),
                               DType, dimdst, ktype> const {
                auto cpu_broad = mshadow::expr::broadcast<dimcast>(left, shape);
                auto gpu_broad = mshadow::expr::broadcast<dimcast>(right, shape);
                return LazyTensor<decltype(cpu_broad), decltype(gpu_broad),
                                  DType, dimdst, ktype>(
                    cpu_broad, gpu_broad, sync_tensors, dependent_tensors
                );
            }
        #else
            template<int dimcast, int dimdst>
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, dimdst, dimdst - dimcast>,
                DType, dimdst, ktype >broadcast(mshadow::Shape<dimdst> shape) const {
                auto cpu_broad = mshadow::expr::broadcast<dimcast>(left, shape);
                return LazyTensor<decltype(cpu_broad), DType, dimdst, ktype>(
                    cpu_broad, sync_tensors, dependent_tensors
                );
            }
        #endif

        #ifdef DALI_USE_CUDA
            inline auto repmat(mshadow::index_t nrow) ->
                    LazyTensor<decltype(mshadow::expr::repmat(left, nrow)),
                               decltype(mshadow::expr::repmat(right, nrow)),
                               DType, 2, ktype> const{
                return broadcast<1>(
                    mshadow::Shape2(
                        nrow,
                        mshadow::expr::ShapeCheck<1, LeftType>::Check(
                            left.self()
                        )[0]));
            }
        #else
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, 2, 1>,
                DType, 2, ktype >
            repmat(mshadow::index_t nrow) {
                return broadcast<1>(
                    mshadow::Shape2(
                        nrow,
                        mshadow::expr::ShapeCheck<1, LeftType>::Check(
                            left.self()
                        )[0]));
            }
        #endif
};


#ifdef DALI_USE_CUDA
    #define BINARY_OP(opname, opsymbol) \
    template<typename TA, typename TB, typename TC, typename TD, typename DType, int dimension, int ta, int tb> \
    auto operator opsymbol( \
            const LazyTensor<TA, TB, DType, dimension, ta> &left, \
            const LazyTensor<TC, TD, DType, dimension, tb> &right) -> \
                LazyTensor< decltype(left.left opsymbol right.left), \
                            decltype(left.right opsymbol right.right), \
                            DType, \
                            dimension, \
                            (ta|tb|mshadow::expr::type::kMapper)>   { \
        const auto& l_cpu = left.left; \
        const auto& r_cpu = right.left; \
        auto res_cpu = l_cpu opsymbol r_cpu; \
        const auto& l_gpu = left.right; \
        const auto& r_gpu = right.right; \
        auto res_gpu = l_gpu opsymbol r_gpu; \
        auto joined_sts = sync_tensors_t(left.sync_tensors); \
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end()); \
        auto joined_dts = dependent_tensors_t(left.dependent_tensors); \
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end()); \
        return LazyTensor<decltype(res_cpu), \
                          decltype(res_gpu), \
                          DType, \
                          dimension, \
                          (ta|tb|mshadow::expr::type::kMapper)>(\
                res_cpu, res_gpu, joined_sts, joined_dts); \
    }

    #define BINARY_SCALAR_OP(opname, opsymbol) \
        template<typename TA, typename TB, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const LazyTensor<TA, TB, DType, dimension, ta> &tensor, \
                const mshadow::expr::ScalarExp<DType> &scalar) -> LazyTensor<decltype(tensor.left opsymbol scalar), \
                                                                             decltype(tensor.right opsymbol scalar), \
                                                                             DType, \
                                                                             dimension, \
                                                                             (ta|mshadow::expr::type::kMapper)> { \
            const auto& l_cpu = tensor.left; \
            auto res_cpu = l_cpu opsymbol scalar; \
            const auto& l_gpu = tensor.right; \
            auto res_gpu = l_gpu opsymbol scalar; \
            return LazyTensor<decltype(res_cpu), \
                              decltype(res_gpu), \
                              DType, \
                              dimension, \
                              (ta|mshadow::expr::type::kMapper)>(\
                res_cpu, res_gpu, tensor.sync_tensors, tensor.dependent_tensors); \
        } \
        \
        template<typename TA, typename TB, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const mshadow::expr::ScalarExp<DType> &scalar, \
                const LazyTensor<TA, TB, DType, dimension, ta>   &tensor) -> LazyTensor<decltype(scalar opsymbol tensor.left), \
                                                                             decltype(scalar opsymbol tensor.right), \
                                                                             DType, \
                                                                             dimension, \
                                                                             (ta|mshadow::expr::type::kMapper)> { \
            const auto& l_cpu = tensor.left; \
            auto res_cpu = scalar opsymbol l_cpu; \
            const auto& l_gpu = tensor.right; \
            auto res_gpu = scalar opsymbol l_gpu; \
            return LazyTensor<decltype(res_cpu), \
                              decltype(res_gpu), \
                              DType, \
                              dimension, \
                              (ta|mshadow::expr::type::kMapper)>(\
                res_cpu, res_gpu, tensor.sync_tensors, tensor.dependent_tensors); \
        } \
        template<typename TA, typename TB, typename DType, int dimension, int ta> \
        inline auto operator opsymbol( \
                const LazyTensor<TA, TB, DType, dimension, ta> &tensor, DType scalar) -> \
                    decltype(tensor opsymbol mshadow::expr::ScalarExp<DType>(scalar)) { \
            return tensor opsymbol mshadow::expr::ScalarExp<DType>(scalar); \
        } \
        \
        template<typename TA, typename TB, typename DType, int dimension, int ta> \
        inline auto operator opsymbol(DType scalar,  const LazyTensor<TA, TB, DType, dimension, ta> &tensor) -> \
                decltype(mshadow::expr::ScalarExp<DType>(scalar) opsymbol tensor) { \
            return mshadow::expr::ScalarExp<DType>(scalar) opsymbol tensor; \
        }
#else
    #define BINARY_OP(opname, opsymbol) \
    template<typename TA, typename TC, typename DType, int dimension, int ta, int tb> \
    auto operator opsymbol( \
            const LazyTensor<TA, DType, dimension, ta> &left, \
            const LazyTensor<TC, DType, dimension, tb> &right) -> \
                LazyTensor< decltype(left.left opsymbol right.left), \
                            DType, \
                            dimension, \
                            (ta|tb|mshadow::expr::type::kMapper)>   { \
        const auto& l_cpu = left.left; \
        const auto& r_cpu = right.left; \
        auto res_cpu = l_cpu opsymbol r_cpu; \
        auto joined_sts = sync_tensors_t(left.sync_tensors); \
        joined_sts.insert(joined_sts.end(), right.sync_tensors.begin(), right.sync_tensors.end()); \
        auto joined_dts = dependent_tensors_t(left.dependent_tensors); \
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end()); \
        return LazyTensor<decltype(res_cpu), \
                          DType, \
                          dimension, \
                          (ta|tb|mshadow::expr::type::kMapper)>(\
                res_cpu, joined_sts, joined_dts); \
    }

    #define BINARY_SCALAR_OP(opname, opsymbol) \
        template<typename TA, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const LazyTensor<TA, DType, dimension, ta> &tensor, \
                const mshadow::expr::ScalarExp<DType> &scalar) -> LazyTensor<decltype(tensor.left opsymbol scalar), \
                                                                             DType, \
                                                                             dimension, \
                                                                             (ta|mshadow::expr::type::kMapper)> { \
            const auto& l_cpu = tensor.left; \
            auto res_cpu = l_cpu opsymbol scalar; \
            return LazyTensor<decltype(res_cpu), \
                              DType, \
                              dimension, \
                              (ta|mshadow::expr::type::kMapper)>(\
                res_cpu, tensor.sync_tensors, tensor.dependent_tensors); \
        } \
        \
        template<typename TA, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const mshadow::expr::ScalarExp<DType> &scalar, \
                const LazyTensor<TA, DType, dimension, ta>   &tensor) -> LazyTensor<decltype(scalar opsymbol tensor.left), \
                                                                             DType, \
                                                                             dimension, \
                                                                             (ta|mshadow::expr::type::kMapper)> { \
            const auto& l_cpu = tensor.left; \
            auto res_cpu = scalar opsymbol l_cpu; \
            return LazyTensor<decltype(res_cpu), \
                              DType, \
                              dimension, \
                              (ta|mshadow::expr::type::kMapper)>(\
                res_cpu, tensor.sync_tensors, tensor.dependent_tensors); \
        } \
        template<typename TA, typename DType, int dimension, int ta> \
        inline auto operator opsymbol( \
                const LazyTensor<TA, DType, dimension, ta> &tensor, DType scalar) -> \
                    decltype(tensor opsymbol mshadow::expr::ScalarExp<DType>(scalar)) { \
            return tensor opsymbol mshadow::expr::ScalarExp<DType>(scalar); \
        } \
        \
        template<typename TA, typename DType, int dimension, int ta> \
        inline auto operator opsymbol(DType scalar,  const LazyTensor<TA, DType, dimension, ta> &tensor) -> \
                decltype(mshadow::expr::ScalarExp<DType>(scalar) opsymbol tensor) { \
            return mshadow::expr::ScalarExp<DType>(scalar) opsymbol tensor; \
        }
#endif

BINARY_OP(mshadow::op::plus,  +);
BINARY_OP(mshadow::op::mul,   *);
BINARY_OP(mshadow::op::minus, -);
BINARY_OP(mshadow::op::div,   /);

BINARY_SCALAR_OP(mshadow::op::plus,  +);
BINARY_SCALAR_OP(mshadow::op::mul,  *);
BINARY_SCALAR_OP(mshadow::op::minus,  -);
BINARY_SCALAR_OP(mshadow::op::div,  /);

#ifdef DALI_USE_CUDA
    template<typename OP, typename TA, typename TB, typename DType, int dimension, int ta>
    inline LazyTensor<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>,
                      mshadow::expr::UnaryMapExp<OP, TB, DType, (ta|mshadow::expr::type::kMapper)>,
                      DType,
                      dimension,
                      (ta|mshadow::expr::type::kMapper)>
    MakeExp(const LazyTensor<TA, TB, DType, dimension, ta> &exp) {
        auto unary_l = mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>(exp.left);
        auto unary_r = mshadow::expr::UnaryMapExp<OP, TB, DType, (ta|mshadow::expr::type::kMapper)>(exp.right);
        return LazyTensor<decltype(unary_l),
                          decltype(unary_r),
                          DType,
                          dimension,
                          (ta|mshadow::expr::type::kMapper)>(unary_l, unary_r, exp.sync_tensors, exp.dependent_tensors);
    }

    template<typename OP, typename TA, typename TB, typename DType, int dimension, int ta>
    inline LazyTensor<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>,
                        mshadow::expr::UnaryMapExp<OP, TB, DType, (ta|mshadow::expr::type::kMapper)>,
                        DType,
                        dimension,
                        (ta|mshadow::expr::type::kMapper)>
    F(const LazyTensor<TA, TB, DType, dimension, ta> &exp) {
        return MakeExp<OP>(exp);
    }
#else
    template<typename OP, typename TA, typename DType, int dimension, int ta>
    inline LazyTensor<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>,
                      DType,
                      dimension,
                      (ta|mshadow::expr::type::kMapper)>
    MakeExp(const LazyTensor<TA, DType, dimension, ta> &exp) {
        auto unary_l = mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>(exp.left);
        return LazyTensor<decltype(unary_l),
                          DType,
                          dimension,
                          (ta|mshadow::expr::type::kMapper)>(unary_l, exp.sync_tensors, exp.dependent_tensors);
    }

    template<typename OP, typename TA, typename DType, int dimension, int ta>
    inline LazyTensor<mshadow::expr::UnaryMapExp<OP, TA, DType, (ta|mshadow::expr::type::kMapper)>,
                        DType,
                        dimension,
                        (ta|mshadow::expr::type::kMapper)>
    F(const LazyTensor<TA, DType, dimension, ta> &exp) {
        return MakeExp<OP>(exp);
    }
#endif

#ifdef DALI_USE_CUDA
    template<int dimkeep, typename TA, typename TB, typename DType, int dimension, int ta>
    auto  sumall_except_dim(const LazyTensor<TA, TB, DType, dimension, ta> &exp) ->
            LazyTensor<decltype(mshadow::expr::sumall_except_dim<dimkeep>(exp.left)),
                       decltype(mshadow::expr::sumall_except_dim<dimkeep>(exp.right)),
                       DType,
                       1,
                       mshadow::expr::type::kComplex> {
        auto cpu_sumall = mshadow::expr::sumall_except_dim<dimkeep>(exp.left);
        auto gpu_sumall = mshadow::expr::sumall_except_dim<dimkeep>(exp.right);
        return LazyTensor<
            decltype(cpu_sumall),
            decltype(gpu_sumall),
            DType,
            1,
            mshadow::expr::type::kComplex
            >(cpu_sumall, gpu_sumall, exp.sync_tensors, exp.dependent_tensors);
    }

    template<typename TA, typename TB, typename DType, int ta>
    auto sum_rows(const LazyTensor<TA, TB, DType, 2, ta> &exp) -> decltype(sumall_except_dim<1>(exp)) {
        return sumall_except_dim<1>(exp);
    }

    template<typename TA, typename TB, typename DType, int ta>
    auto sum_cols(const LazyTensor<TA, TB, DType, 2, ta> &exp) -> decltype(sumall_except_dim<0>(exp)) {
        return sumall_except_dim<0>(exp);
    }
#else
    template<int dimkeep, typename TA, typename DType, int dimension, int ta>
    auto  sumall_except_dim(const LazyTensor<TA, DType, dimension, ta> &exp) ->
            LazyTensor<decltype(mshadow::expr::sumall_except_dim<dimkeep>(exp.left)),
                       DType,
                       1,
                       mshadow::expr::type::kComplex> {
        auto cpu_sumall = mshadow::expr::sumall_except_dim<dimkeep>(exp.left);
        return LazyTensor<
            decltype(cpu_sumall),
            DType,
            1,
            mshadow::expr::type::kComplex
            >(cpu_sumall, exp.sync_tensors, exp.dependent_tensors);
    }

    template<typename TA, typename DType, int ta>
    auto sum_rows(const LazyTensor<TA, DType, 2, ta> &exp) -> decltype(sumall_except_dim<1>(exp)) {
        return sumall_except_dim<1>(exp);
    }

    template<typename TA, typename DType, int ta>
    auto sum_cols(const LazyTensor<TA, DType, 2, ta> &exp) -> decltype(sumall_except_dim<0>(exp)) {
        return sumall_except_dim<0>(exp);
    }
#endif

#endif
