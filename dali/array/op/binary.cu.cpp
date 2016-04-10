#include "dali/array/op/binary.h"

#include "dali/array/array.h"
#include "dali/array/array_function.h"
#include "dali/array/TensorFunctions.h"

typedef BinaryElementwise<TensorOps::op::plus> Plus;
AssignableArray add(const Array& a, const Array& b) {return Plus::run(a, b);}
AssignableArray operator+(const Array& a, const Array& b) {return add(a,b);}

typedef BinaryElementwise<TensorOps::op::sub> Sub;
AssignableArray sub(const Array& a, const Array& b) {return Sub::run(a, b);}
AssignableArray operator-(const Array& a, const Array& b) {return sub(a,b);}

typedef BinaryElementwise<TensorOps::op::mul> EltMul;
AssignableArray eltmul(const Array& a, const Array& b) {return EltMul::run(a, b);}
AssignableArray operator*(const Array& a, const Array& b) {return eltmul(a,b);}

typedef BinaryElementwise<TensorOps::op::div> EltDiv;
AssignableArray eltdiv(const Array& a, const Array& b) {return EltDiv::run(a, b);}
AssignableArray operator/(const Array& a, const Array& b) {return eltdiv(a,b);}
