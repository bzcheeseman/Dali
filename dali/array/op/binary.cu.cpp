#include "dali/array/op/binary.h"
#include "dali/array/function/function.h"
#include "dali/array/TensorFunctions.h"

typedef BinaryElementwise<TensorOps::op::add> Plus;
AssignableArray add(const Array& a, const Array& b) {return Plus::run(a, b);}

typedef BinaryElementwise<TensorOps::op::sub> Sub;
AssignableArray sub(const Array& a, const Array& b) {return Sub::run(a, b);}

typedef BinaryElementwise<TensorOps::op::eltmul> EltMul;
AssignableArray eltmul(const Array& a, const Array& b) {return EltMul::run(a, b);}

typedef BinaryElementwise<TensorOps::op::eltdiv> EltDiv;
AssignableArray eltdiv(const Array& a, const Array& b) {return EltDiv::run(a, b);}
