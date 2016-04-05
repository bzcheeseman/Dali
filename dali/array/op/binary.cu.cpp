#include "dali/array/op/binary.h"

#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/TensorFunctions.h"


template<template<class> class Functor>
struct BinaryElementwise : public Function<BinaryElementwise<Functor>, Array, Array, Array> {
    template<int devT, typename T>
    Array run(MArray<devT,T> left, MArray<devT,T> right) {
        Array out(left.array.shape(), left.array.dtype());

        auto mout = MArray<devT,T>{out, left.device};

        mout.d1(memory::AM_OVERWRITE) = mshadow::expr::F<Functor<T>>(left.d1(), right.d1());
        return out;
    }
};

typedef BinaryElementwise<TensorOps::op::plus> Plus;

Array add(const Array& a, const Array& b) {return Plus::eval(a, b);}

typedef BinaryElementwise<TensorOps::op::mul> EltMul;
Array eltmul(const Array& a, const Array& b) {return EltMul::eval(a, b);}
