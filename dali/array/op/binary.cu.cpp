#include "dali/array/op/binary.h"

#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/TensorFunctions.h"

typedef BinaryElementwise<TensorOps::op::plus> Plus;
Array add(const Array& a, const Array& b) {return Plus::eval(a, b);}
Array operator+(const Array& a, const Array& b) {return add(a,b);}

typedef BinaryElementwise<TensorOps::op::sub> Sub;
Array sub(const Array& a, const Array& b) {return Sub::eval(a, b);}
Array operator-(const Array& a, const Array& b) {return sub(a,b);}

typedef BinaryElementwise<TensorOps::op::mul> EltMul;
Array eltmul(const Array& a, const Array& b) {return EltMul::eval(a, b);}
Array operator*(const Array& a, const Array& b) {return eltmul(a,b);}

typedef BinaryElementwise<TensorOps::op::div> EltDiv;
Array eltdiv(const Array& a, const Array& b) {return EltDiv::eval(a, b);}
Array operator/(const Array& a, const Array& b) {return eltdiv(a,b);}
