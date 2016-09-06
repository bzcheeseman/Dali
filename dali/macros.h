#ifdef __CUDACC__
#define XINLINE __device__ __host__
#else
#define XINLINE inline
#endif
