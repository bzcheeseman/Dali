#ifndef DALI_MAT_MATH_LAZY_TENSOR_H
#define DALI_MAT_MATH_LAZY_TENSOR_H

#include <functional>
#include <vector>

#include "mshadow/extension/reduceto1d.h"
#include "mshadow/tensor.h"


#include "dali/math/LazySoftmax.h"
#include "dali/math/LazyUtils.h"
#include "dali/math/LazyDot.h"

template<typename DType, int dimension>
class TensorInternal;
template<typename DType>
class SynchronizedMemory;

template<typename DType>
class DormantTensor {
    public:
        std::shared_ptr<SynchronizedMemory<DType>> source;
        // call this method with a new tensor that should be placed here.
        virtual void update_tensor(Device where_to_update) = 0;
        DormantTensor() : source(nullptr) {}
        DormantTensor(std::shared_ptr<SynchronizedMemory<DType>> _source) : source(_source) {}
};

template<typename DType>
using dependent_tensors_t = std::vector<DormantTensor<DType>*>;

template<typename DType>
std::vector<const SynchronizedMemory<DType>*> extract_memory(const dependent_tensors_t<DType>& dts) {
    std::vector<const SynchronizedMemory<DType>*> res;
    for (auto dt: dts) {
        res.push_back(dt.source.get());
    }
    return res;
}


#ifdef DALI_USE_CUDA
    template<typename LeftType, typename RightType, typename DType, int dimension, int ktype>
#else
    template<typename LeftType, typename DType, int dimension, int ktype>
#endif
class LazyTensor : public DormantTensor<DType> {
    public:
        typedef mshadow::Tensor<mshadow::cpu, dimension, DType> cpu_tensor_t;

        // store list of dependant tensors whose memory
        // may need to be refreshed before evaluation
        dependent_tensors_t<DType> dependent_tensors;

        LeftType                  left;
        mshadow::Shape<dimension> source_shape;

        #ifdef DALI_USE_CUDA
            typedef mshadow::Tensor<mshadow::gpu, dimension, DType> gpu_tensor_t;
            RightType right;
        #endif

        #ifdef DALI_USE_CUDA
            LazyTensor(
                const LeftType& _left,
                const RightType& _right,
                const dependent_tensors_t<DType>& _dependent_tensors)
                : DormantTensor<DType>(), left(_left), right(_right), dependent_tensors(_dependent_tensors) {}
        #else
            // Same expression with the gpu twin
            // ignored.
            LazyTensor(
                const LeftType& _left,
                const dependent_tensors_t<DType>& _dependent_tensors)
                : DormantTensor<DType>(),
                  left(_left),
                  dependent_tensors(_dependent_tensors) {}
        #endif

        #ifdef DALI_USE_CUDA
            LazyTensor(const TensorInternal<DType,dimension>& st)
                : DormantTensor<DType>(st.memory),
                  source_shape(st.shape),
                  left(LeftType(st.shape)),
                  right(RightType(st.shape)),
                  dependent_tensors({this}) {}
        #else
            LazyTensor(const TensorInternal<DType,dimension>& st)
                : DormantTensor<DType>(st.memory),
                  source_shape(st.shape),
                  left(LeftType(st.shape)),
                  dependent_tensors({this}) {}
        #endif

            // replace the tensors used in this LazyTensor
            // by a copy with fresh memory before evaluation.
            virtual void update_tensor(Device where_to_update) override {
                assert((bool)(this->source));
                if (where_to_update == DEVICE_CPU) {
                    cpu_tensor_t * bjarne_stop = (cpu_tensor_t*)&left;
                    *bjarne_stop = cpu_tensor_t(this->source->cpu_data(), source_shape);
                }
                #ifdef DALI_USE_CUDA
                else if (where_to_update == DEVICE_GPU) {
                    gpu_tensor_t * bjarne_stop = (gpu_tensor_t*)&right;
                    *bjarne_stop = gpu_tensor_t(this->source->gpu_data(), source_shape);
                }
                #endif
            }

        #ifdef DALI_USE_CUDA
            inline auto T(void) const -> LazyTensor<decltype(left.T()), decltype(right.T()), DType, dimension, ktype> {
                return LazyTensor<decltype(left.T()), decltype(right.T()), DType, dimension, ktype>(
                        left.T(), right.T(), dependent_tensors);
            }
        #else
            inline auto T(void) const -> LazyTensor<decltype(left.T()), DType, dimension, ktype> {
                return LazyTensor<decltype(left.T()), DType, dimension, ktype>(
                        left.T(), dependent_tensors);
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
                        cpu_soft, gpu_soft, dependent_tensors);
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
                    >(cpu_soft, dependent_tensors);
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
                    cpu_broad, gpu_broad, dependent_tensors
                );
            }
        #else
            template<int dimcast, int dimdst>
            inline LazyTensor<
                mshadow::expr::Broadcast1DExp<LeftType, DType, dimdst, dimdst - dimcast>,
                DType, dimdst, ktype >broadcast(mshadow::Shape<dimdst> shape) const {
                auto cpu_broad = mshadow::expr::broadcast<dimcast>(left, shape);
                return LazyTensor<decltype(cpu_broad), DType, dimdst, ktype>(
                    cpu_broad, dependent_tensors
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
        auto joined_dts = dependent_tensors_t<DType>(left.dependent_tensors); \
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end()); \
        return LazyTensor<decltype(res_cpu), \
                          decltype(res_gpu), \
                          DType, \
                          dimension, \
                          (ta|tb|mshadow::expr::type::kMapper)>(\
                res_cpu, res_gpu, joined_dts); \
    }

    #define BINARY_SCALAR_OP(opname, opsymbol) \
        template<typename TA, typename TB, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const LazyTensor<TA, TB, DType, dimension, ta> &tensor, \
                const mshadow::expr::ScalarExp<DType> scalar) -> LazyTensor<decltype(tensor.left opsymbol scalar), \
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
                res_cpu, res_gpu, tensor.dependent_tensors); \
        } \
        \
        template<typename TA, typename TB, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const mshadow::expr::ScalarExp<DType> scalar, \
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
                res_cpu, res_gpu, tensor.dependent_tensors); \
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
        auto joined_dts = dependent_tensors_t<DType>(left.dependent_tensors); \
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end()); \
        return LazyTensor<decltype(res_cpu), \
                          DType, \
                          dimension, \
                          (ta|tb|mshadow::expr::type::kMapper)>(\
                res_cpu, joined_dts); \
    }

    #define BINARY_SCALAR_OP(opname, opsymbol) \
        template<typename TA, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const LazyTensor<TA, DType, dimension, ta> &tensor, \
                const mshadow::expr::ScalarExp<DType> scalar) -> LazyTensor<decltype(tensor.left opsymbol scalar), \
                                                                             DType, \
                                                                             dimension, \
                                                                             (ta|mshadow::expr::type::kMapper)> { \
            const auto& l_cpu = tensor.left; \
            auto res_cpu = l_cpu opsymbol scalar; \
            return LazyTensor<decltype(res_cpu), \
                              DType, \
                              dimension, \
                              (ta|mshadow::expr::type::kMapper)>(\
                res_cpu, tensor.dependent_tensors); \
        } \
        \
        template<typename TA, typename DType, int dimension, int ta> \
        auto operator opsymbol( \
                const mshadow::expr::ScalarExp<DType> scalar, \
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
                res_cpu, tensor.dependent_tensors); \
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


/////////////////////////////// UNARY EXPRESSION FUNCTIONS ///////////////////////////////////////////////////////////////


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
                          (ta|mshadow::expr::type::kMapper)>(unary_l, unary_r, exp.dependent_tensors);
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
                          (ta|mshadow::expr::type::kMapper)>(unary_l, exp.dependent_tensors);
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

/////////////////////////////// BINARY EXPRESSION FUNCTIONS ///////////////////////////////////////////////////////////////

#ifdef DALI_USE_CUDA
    template<typename OP, typename TA, typename TB, typename TC, typename TD, typename DType, int dimension, int ta, int tb>
    inline auto
    MakeExp(const LazyTensor<TA, TB, DType, dimension, ta> &left,
            const LazyTensor<TC, TD, DType, dimension, tb> &right) ->
                        LazyTensor<decltype(mshadow::expr::F<OP>(left.left,  right.left)),
                                   decltype(mshadow::expr::F<OP>(left.right, right.right)),
                                   DType, dimension,
                                   (ta|tb|mshadow::expr::type::kMapper)>{

        auto cpu_res = mshadow::expr::F<OP>(left.left,  right.left);
        auto gpu_res = mshadow::expr::F<OP>(left.right, right.right);
        auto joined_dts = dependent_tensors_t<DType>(left.dependent_tensors);
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end());
        return LazyTensor<decltype(cpu_res),
                          decltype(gpu_res),
                          DType, dimension,
                          (ta|tb|mshadow::expr::type::kMapper)>(
                            cpu_res, gpu_res, joined_dts
                        );
    }

    template<typename OP, typename TA, typename TB, typename DType, int dimension, int ta>
    inline auto
    MakeExp(const LazyTensor<TA, TB, DType, dimension, ta> &left,
            const mshadow::expr::ScalarExp<DType> right) ->
                        LazyTensor<decltype(mshadow::expr::F<OP>(left.left,  right)),
                                   decltype(mshadow::expr::F<OP>(left.right, right)),
                                   DType, dimension,
                                   (ta|mshadow::expr::type::kMapper)>{

        auto cpu_res = mshadow::expr::F<OP>(left.left,  right);
        auto gpu_res = mshadow::expr::F<OP>(left.right, right);
        return LazyTensor<decltype(cpu_res),
                          decltype(gpu_res),
                          DType, dimension,
                          (ta|mshadow::expr::type::kMapper)>(
                            cpu_res, gpu_res, left.dependent_tensors
                        );
    }

    template<typename OP, typename TA, typename TB, typename DType, int dimension, int ta>
    inline auto
    MakeExp(const mshadow::expr::ScalarExp<DType> left,
            const LazyTensor<TA, TB, DType, dimension, ta> &right) ->
                        LazyTensor<decltype(mshadow::expr::F<OP>(left, right.left)),
                                   decltype(mshadow::expr::F<OP>(left, right.left)),
                                   DType, dimension,
                                   (ta|mshadow::expr::type::kMapper)>{

        auto cpu_res = mshadow::expr::F<OP>(left, right.left);
        auto gpu_res = mshadow::expr::F<OP>(left, right.right);
        return LazyTensor<decltype(cpu_res),
                          decltype(gpu_res),
                          DType, dimension,
                          (ta|mshadow::expr::type::kMapper)>(
                            cpu_res, gpu_res, right.dependent_tensors
                        );
    }

    template<typename OP, typename TA, typename TB, typename TC, typename TD, typename DType, int dimension, int ta, int tb>
    inline auto F(const LazyTensor<TA, TB, DType, dimension, ta> &left,
                  const LazyTensor<TC, TD, DType, dimension, tb> &right) -> decltype(MakeExp<OP>(left, right)) {
        return MakeExp<OP>(left, right);
    }

    template<typename OP, typename TA, typename TB, typename DType, int dimension, int ta>
    inline auto F(const DType left,
                  const LazyTensor<TA, TB, DType, dimension, ta> &right)
            -> decltype(MakeExp<OP>(mshadow::expr::ScalarExp<DType>(left), right)) {
        return MakeExp<OP>(mshadow::expr::ScalarExp<DType>(left), right);
    }

    template<typename OP, typename TA, typename TB, typename DType, int dimension, int ta>
    inline auto F(const LazyTensor<TA, TB, DType, dimension, ta> &left,
                  const DType right) -> decltype(MakeExp<OP>(left, mshadow::expr::ScalarExp<DType>(right))) {
        return MakeExp<OP>(left, mshadow::expr::ScalarExp<DType>(right));
    }
#else

    template<typename OP, typename TA, typename TC, typename DType, int dimension, int ta, int tb>
    inline auto
    MakeExp(const LazyTensor<TA, DType, dimension, ta> &left,
            const LazyTensor<TC, DType, dimension, tb> &right) ->
                        LazyTensor<decltype(mshadow::expr::F<OP>(left.left,  right.left)),
                                   DType, dimension,
                                   (ta|tb|mshadow::expr::type::kMapper)>{

        auto cpu_res = mshadow::expr::F<OP>(left.left,  right.left);
        auto joined_dts = dependent_tensors_t<DType>(left.dependent_tensors);
        joined_dts.insert(joined_dts.end(), right.dependent_tensors.begin(), right.dependent_tensors.end());
        return LazyTensor<decltype(cpu_res),
                          DType, dimension,
                          (ta|tb|mshadow::expr::type::kMapper)>(
                            cpu_res, joined_dts
                        );
    }

    template<typename OP, typename TA, typename DType, int dimension, int ta>
    inline auto
    MakeExp(const LazyTensor<TA, DType, dimension, ta> &left,
            mshadow::expr::ScalarExp<DType> right) ->
                        LazyTensor<decltype(mshadow::expr::F<OP>(left.left, right)),
                                   DType, dimension,
                                   (ta|mshadow::expr::type::kMapper)>{
        auto cpu_res = mshadow::expr::F<OP>(left.left, right);
        return LazyTensor<decltype(cpu_res),
                          DType, dimension,
                          (ta|mshadow::expr::type::kMapper)>(
                              cpu_res, left.dependent_tensors
                          );
    }

    template<typename OP, typename TA, typename DType, int dimension, int ta>
    inline auto
    MakeExp(const mshadow::expr::ScalarExp<DType> left,
            const LazyTensor<TA, DType, dimension, ta> &right) ->
                        LazyTensor<decltype(mshadow::expr::F<OP>(left,  right.left)),
                                   DType, dimension,
                                   (ta|mshadow::expr::type::kMapper)>{
        auto cpu_res = mshadow::expr::F<OP>(left,  right.left);
        return LazyTensor<decltype(cpu_res),
                          DType, dimension,
                          (ta|mshadow::expr::type::kMapper)>(
                              cpu_res, right.dependent_tensors
                          );
    }


    template<typename OP, typename TA, typename TC, typename DType, int dimension, int ta, int tb>
    inline auto F(const LazyTensor<TA, DType, dimension, ta> &left,
                  const LazyTensor<TC, DType, dimension, tb> &right) -> decltype(MakeExp<OP>(left, right)) {
        return MakeExp<OP>(left, right);
    }

    template<typename OP, typename TA, typename DType, int dimension, int ta>
    inline auto F(const DType left,
                  const LazyTensor<TA, DType, dimension, ta> &right)
            -> decltype(MakeExp<OP>(mshadow::expr::ScalarExp<DType>(left), right)) {
        return MakeExp<OP>(mshadow::expr::ScalarExp<DType>(left), right);
    }

    template<typename OP, typename TA, typename DType, int dimension, int ta>
    inline auto F(const LazyTensor<TA, DType, dimension, ta> &left,
                  const DType right)
            -> decltype(MakeExp<OP>(left, mshadow::expr::ScalarExp<DType>(right))) {
        return MakeExp<OP>(left, mshadow::expr::ScalarExp<DType>(right));
    }
#endif



#ifdef DALI_USE_CUDA
template<int a1, int a2, typename TA, typename TB, typename DType, int dimension, int ta>
inline auto swapaxis(const LazyTensor<TA, TB, DType, dimension, ta> &exp)
    -> LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)), decltype(mshadow::expr::swapaxis<a1,a2>(exp.right)), DType, dimension, ta> {
        return LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)),
                          decltype(mshadow::expr::swapaxis<a1,a2>(exp.right)),
                          DType, dimension,
                          (ta|mshadow::expr::type::kMapper)>(
                              mshadow::expr::swapaxis<a1,a2>(exp.left),
                              mshadow::expr::swapaxis<a1,a2>(exp.right),
                              exp.dependent_tensors
                );

    }
#else
template<int a1, int a2, typename TA, typename DType, int dimension, int ta>
inline auto swapaxis(const LazyTensor<TA, DType, dimension, ta> &exp)
    -> LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)), DType, dimension, ta> {
        return LazyTensor<decltype(mshadow::expr::swapaxis<a1,a2>(exp.left)),
                          DType, dimension,
                          (ta|mshadow::expr::type::kMapper)>(
            mshadow::expr::swapaxis<a1,a2>(exp.left),
            exp.dependent_tensors
        );
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
            >(cpu_sumall, gpu_sumall, exp.dependent_tensors);
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
            >(cpu_sumall, exp.dependent_tensors);
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
