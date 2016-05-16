#ifndef DALI_ARRAY_OP_H
#define DALI_ARRAY_OP_H

#include "dali/config.h"

#include "dali/array/op/other.h"
#include "dali/array/op/dot.h"

#if EXISTS_AND_TRUE(DALI_USE_LAZY)
    #include "dali/array/lazy/binary.h"
    #include "dali/array/lazy/reducers.h"
    #include "dali/array/lazy/unary.h"
    namespace lazy {
        static bool ops_loaded = true;
    }

    // Scalar templates need to be explicitly instantiated for every primitive
    // type we need to support. Otherwise C++ does not implicitly cast them.
    // this macro should be used N times with TYPE=[float,double,int,...]
    #define LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,TYPE) \
        template<typename T> \
        auto operator OP (const Exp<T>& left, const TYPE& right) -> decltype( FNAME (left.self(),right)) { \
            return FNAME (left.self(),right); \
        } \
        template<typename T> \
        auto operator OP (const TYPE& left, const Exp<T>& right) -> decltype( FNAME (left,right.self())) { \
            return FNAME (left,right.self()); \
        }

    #define LAZY_BINARY_OPERATOR(OP,FNAME) \
        template<typename T, typename T2> \
        auto operator OP(const Exp<T>& left, const Exp<T2>& right) -> decltype( FNAME (left.self(),right.self())) { \
            return FNAME (left.self(),right.self()); \
        } \
        LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,float) \
        LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,double) \
        LAZY_BINARY_SCALAR_OPERATORS(OP,FNAME,int)

        LAZY_BINARY_OPERATOR(+, lazy::add)
        LAZY_BINARY_OPERATOR(-, lazy::sub)
        LAZY_BINARY_OPERATOR(*, lazy::eltmul)
        LAZY_BINARY_OPERATOR(/, lazy::eltdiv)
#else
    #include "dali/array/op/binary.h"
    #include "dali/array/op/initializer.h"
    #include "dali/array/op/unary.h"
    namespace lazy {
        static bool ops_loaded = false;
    }

    #define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
        AssignableArray operator SYMBOL (const Array& left, const Array& right);\

    #define DALI_DECLARE_ARRAY_INTERACTION_INPLACE(SYMBOL)\
        Array& operator SYMBOL (Array& left, const Array& right);\

    #define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
        AssignableArray operator SYMBOL (const Array& left, const double& right);\
        AssignableArray operator SYMBOL (const Array& left, const float& right);\
        AssignableArray operator SYMBOL (const Array& left, const int& right);\

    #define DALI_DECLARE_SCALAR_INTERACTION_INPLACE(SYMBOL)\
        Array& operator SYMBOL (Array& left, const double& right);\
        Array& operator SYMBOL (Array& left, const float& right);\
        Array& operator SYMBOL (Array& left, const int& right);\

    DALI_DECLARE_ARRAY_INTERACTION(+);
    DALI_DECLARE_ARRAY_INTERACTION(-);
    DALI_DECLARE_ARRAY_INTERACTION(*);
    DALI_DECLARE_ARRAY_INTERACTION(/);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(+=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(-=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(*=);
    DALI_DECLARE_ARRAY_INTERACTION_INPLACE(/=);

    Array& operator<<=(Array& left, const Array& right);

    DALI_DECLARE_SCALAR_INTERACTION(-);
    DALI_DECLARE_SCALAR_INTERACTION(+);
    DALI_DECLARE_SCALAR_INTERACTION(*);
    DALI_DECLARE_SCALAR_INTERACTION(/);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(-=);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(+=);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(*=);
    DALI_DECLARE_SCALAR_INTERACTION_INPLACE(/=);

#endif

#endif
