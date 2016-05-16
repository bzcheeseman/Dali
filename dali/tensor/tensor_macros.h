#ifndef DALI_TENSOR_TENSOR_MACROS_H
#define DALI_TENSOR_TENSOR_MACROS_H

#define MAYBE_GRAD(X) if (!(X).constant) X.dw

#endif
