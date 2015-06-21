#ifndef DALI_MAT_MATH___MAT_MACROS___H
#define DALI_MAT_MATH___MAT_MACROS___H

#define MAT(matrix) ((matrix).w())
#define GRAD(X) ((X).dw())

#define SAFE_GRAD(X) if (!(X).constant) GRAD(X)

#endif
