#include <iostream>
#include "mshadow/tensor.h"
#include "assert.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/detail/normal_iterator.h>

using namespace mshadow;

template<typename xpu>
void Print1DTensor(Tensor<xpu, 1, float> const &ts);

template<typename xpu>
void Print2DTensor(Tensor<xpu, 2, float> const &ts);

template<>
void Print1DTensor(Tensor<cpu, 1, float> const &ts) {
  for (index_t i = 0; i < ts.size(0); ++i) {
    printf("%.2f ", ts[i]);
  }
  printf("\n");
}

template<>
void Print1DTensor(Tensor<gpu, 1, float> const &tg) {
    Tensor<cpu, 1, float> tc = NewTensor<cpu, float>(tg.shape_, 0.0f);
    Copy(tc, tg);
    Print1DTensor(tc);
    FreeSpace(&tc);
}

template<>
void Print2DTensor(Tensor<cpu, 2, float> const &ts) {
    for (index_t i = 0; i < ts.size(0); ++i) {
        Print1DTensor(ts[i]);
    }
}

template<>
void Print2DTensor(Tensor<gpu, 2, float> const &tg) {
    Tensor<cpu, 2, float> tc = NewTensor<cpu, float>(tg.shape_, 0.0f);
    Copy(tc, tg);
    Print2DTensor(tc);
    FreeSpace(&tc);
}


thrust::device_ptr<float> to_thrust(Tensor<gpu, 2, float>& tg) {
    auto dev_ptr = thrust::device_pointer_cast(tg.dptr_);
    return dev_ptr;
}

#define PRINT2D_SMALL(X) if ((X).shape_[0] <= 20 && (X).shape_[1] <= 20) Print2DTensor((X))
#define PRINT1D_SMALL(X) if ((X).shape_[0] <= 30) Print1DTensor((X))


void conv_showcase() {
    std::cout << "CONV SHOWCASE" << std::endl;

    int ksize = 2;
    int kstride = 1;
    int nkernels = 2;

    int H = 3;
    int W = 4;

    Tensor<cpu, 3, float> src = NewTensor<cpu, float>(Shape3(1, H, W), 0.0f);
    Tensor<cpu, 2, float> kernels = NewTensor<cpu, float>(Shape2(nkernels, ksize * ksize), 0.0f);


    int o_height = (src.size(1)- ksize) / kstride + 1;
    int o_width  = (src.size(2)- ksize) / kstride + 1;

    for (int i=0; i<H; ++i) {
        for (int j=0; j<W; ++j) {
            src[0][i][j] = i*W +j;
        }
    }

    for (int i=0; i <nkernels; ++i) {
        for (int j=0; j <ksize*ksize; ++j) {
            kernels[i][j] = i + 1;
        }
    }

    std::cout << "== Image ==" << std::endl;
    Print2DTensor(src[0]);
    std::cout << "== Kernel ==" << std::endl;
    Print2DTensor(kernels);


    Tensor<cpu, 2> tmp_col = NewTensor<cpu, float>(Shape2(ksize * ksize, o_height * o_width), 0.0f);
    tmp_col = unpack_patch2col(src, ksize, ksize, kstride);
    std::cout << "== Image Patches ==" << std::endl;
    Print2DTensor(tmp_col);

    std::cout << "== Applied Kernels (packed) ==" << std::endl;
    Tensor<cpu, 2> tmp_dst = NewTensor<cpu, float>(Shape2(nkernels, o_height * o_width), 0.0f);
    tmp_dst = dot(kernels, tmp_col);
    Print2DTensor(tmp_dst);

    Tensor<cpu, 3, float> dst = NewTensor<cpu, float>(Shape3(nkernels, o_height, o_width), 0.0f);

    dst = reshape(tmp_dst, dst.shape_);
    std::cout << "== Applied Kernel 1 ==" << std::endl;
    Print2DTensor(dst[0]);
    std::cout << "== Applied Kernel 2 ==" << std::endl;
    Print2DTensor(dst[1]);



    FreeSpace(&src);
    FreeSpace(&tmp_col);
    FreeSpace(&tmp_dst);
}


int main () {
    conv_showcase();

    int first_dim = 3;
    int second_dim = 1000;
    int third_dim = 2000;

    InitTensorEngine<gpu>();
    Tensor<cpu, 3, float> tc = NewTensor<cpu, float>(Shape3(first_dim, second_dim, third_dim), 0.0f);
    // init
    for (index_t i = 0; i < tc.size(0); ++i) {
      for (index_t j = 0; j < tc.size(1); ++j) {
        for (index_t k = 0; k < tc.size(2); ++k) {
          tc[i][j][k] = i * 0.1f + j * 0.2f + k * 0.1f;
        }
      }
    }
    // print
    printf("\n#print batch 0 of cpu tensor:\n");
    PRINT2D_SMALL(tc[0]);
    printf("\n");
    PRINT2D_SMALL(tc[1]);
    printf("\n");
    PRINT2D_SMALL(tc[2]);
    // check
    // sum of row
    Tensor<cpu, 1, float> tmp_tc = NewTensor<cpu, float>(Shape1(tc[0].size(1)), 0.0f);
    printf("\n#sum_rows of batch 0:\n");
    tmp_tc = sum_rows(tc[0]);
    PRINT1D_SMALL(tmp_tc);
    FreeSpace(&tmp_tc);
    // softmax
    printf("\n#Softmax\n");
    Tensor<cpu, 2, float> sm_tc = NewTensor<cpu, float>(tc[0].shape_, 0.0f);

    Softmax(sm_tc, tc[0]);

    // summing matrices
    printf("\n#Sum\n");
    Tensor<cpu, 2, float> A = NewTensor<cpu, float>(Shape2(5, 5), 0.2f);
    Tensor<cpu, 2, float> B = NewTensor<cpu, float>(Shape2(5, 5), 0.3f);



    auto bob = NewTensor<cpu, float>(Shape2(5, 5), 0.0f);
    auto joe = NewTensor<gpu, float>(Shape2(5, 5), 0.0f);

    std::cout << "expected => " <<  0.2 + 0.3 + 0.3 * 2.0 + 0.2 * 0.3 + 1.0 << std::endl;
    bob = A + B + B * 2.0 + A * B + 1.0;
    Copy(joe, bob);

    auto joe_ptr = to_thrust(joe);

    int ymax = joe.shape_.shape_[0] * joe.shape_.shape_[1];

    auto are_equal = thrust::equal(
        joe_ptr,
        joe_ptr + ymax,
        joe_ptr
    );

    std::cout << "joe equals bob = " << (are_equal ? "true" : "false") << std::endl;

    //bob += B;

    //Tensor<cpu, 2, float> bob = A + B;

    Print2DTensor(bob);
    Print2DTensor(joe);

    FreeSpace(&joe);
    FreeSpace(&bob);

    ShutdownTensorEngine<gpu>();
    return 0;
}
