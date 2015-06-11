#define GET_MAT(X) (X).w()->w
#define GET_GRAD(X) (X).dw()->dw
#define GRAD(X) if (!(X).constant) GET_GRAD(X)
