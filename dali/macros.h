#ifdef __CUDACC__
#define XINLINE inline __device__ __host__
#else
#define XINLINE inline
#endif
