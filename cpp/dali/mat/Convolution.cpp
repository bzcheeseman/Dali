#include "dali/utils.h"
#include "dali/mat/MatOps.h"
#include "dali/mat/Tape.h"
#include "SQLiteCpp/Database.h"

using utils::assert2;
using utils::MS;
using std::make_shared;
using std::shared_ptr;
using std::vector;

#define GRAD(X) if (!(X).constant) (X).dw()

// Note if kernel is 3D (as in multi kernel)
// Then result must also be a tensor (first dimension is kernel dimension)
template<typename R>
Mat<R> MatOps<R>::conv2d(Mat<R> image, Mat<R> kernel) {
    assert2(image.dims(0) >= kernel.dims(0),
        MS() << "Kernel's first dimension (" << kernel.dims(0)
             << ") must be smaller than or equal to argument's first dimension ("
             << image.dims(0));
    assert2(image.dims(1) >= kernel.dims(1),
        MS() << "Kernel's second dimension (" << kernel.dims(1)
             << ") must be smaller than or equal to argument's first dimenion ("
             << image.dims(1));

    auto out = Mat<R>(
        image.dims(0) - kernel.dims(0) + 1, // as many times as the kernel fits
        image.dims(1) - kernel.dims(1) + 1, // as many times as the kernel fits
        false // fill zeros
    );
    auto& out_mat = out.w();
    auto& image_mat = image.w();
    auto& kernel_mat = kernel.w();
    int col=0,
        row=0,
        KSizeX = kernel.dims(0),
        KSizeY = kernel.dims(1),
        SizeX  = image.dims(0),
        SizeY  = image.dims(1);
    R kernel_sum = kernel_mat.sum();

    for ( row = 0; row < out.dims(0); row++ ) {
        for ( col = 0; col < out.dims(1); col++ ) {
            out_mat(row,col) = (image_mat.block(row, col, KSizeX, KSizeY).array() * kernel_mat.array()).sum() / kernel_sum;
        }
    }
    if (graph::backprop_enabled) {
        graph::emplace_back([image, kernel, out, kernel_sum](){
            auto& image_mat = image.w();
            auto& kernel_mat = kernel.w();
            int col=0,
                row=0,
                KSizeX = kernel.dims(0),
                KSizeY = kernel.dims(1),
                SizeX  = image.dims(0),
                SizeY  = image.dims(1);
            bool grad_image = !image.constant;
            bool grad_kernel = !kernel.constant;
            auto& out_grad = out.dw();
            auto& out_weight = out.w();
            R kern_sum_squared = kernel_sum * kernel_sum;

            if (grad_image || grad_kernel) {
                if (grad_kernel) {
                    kernel.dw().array() -= (out_weight.array() * out_grad.array() / (kernel_sum)).sum();
                }
                for ( row = 0; row < out.dims(0); row++ ) {
                    for ( col = 0; col < out.dims(1); col++ ) {
                        if (grad_image) {
                            image.dw().block(row, col, KSizeX, KSizeY).noalias() += kernel_mat * (out_grad(row, col) / kernel_sum);
                        }
                        if (grad_kernel) {
                            kernel.dw().noalias() += (image_mat.block(row, col, KSizeX, KSizeY).array() * (out_grad(row, col) / (kernel_sum))).matrix();
                        }
                    }
                }
            }
        });
    }
    return out;
}

template<typename R>
Mat<R> MatOps<R>::conv1d(Mat<R> image, Mat<R> kernel) {
    auto kerns = vector<Mat<R>>({kernel});
    return MatOps<R>::conv1d(image, kerns);
}

// Here multiple kernels are allowable
template<typename R>
Mat<R> MatOps<R>::conv1d(Mat<R> image, const vector<Mat<R>>& kernels) {
    assert2(kernels.size() > 0, "Must pass at least 1 kernel to conv1d.");
    int kern_col_size = kernels[0].dims(1);
    for (auto& kernel : kernels) {
        assert2(image.dims(0) == kernel.dims(0),
            MS() << "Kernel's first dimension (" << kernel.dims(0)
                 << ") must be equal than or equal to argument's first dimension ("
                 << image.dims(0));
        assert2(image.dims(1) >= kernel.dims(1),
            MS() << "Kernel's second dimension (" << kernel.dims(1)
                 << ") must be smaller than or equal to argument's first dimenion ("
                 << image.dims(1));
        assert2(kern_col_size == kernel.dims(1),
            MS() << "All Kernel's second dimension (" << kernel.dims(1)
                 << ") must be equal");
    }
    auto out = Mat<R>(
        kernels.size(), // 1d convolution only holds one row
        image.dims(1) - kern_col_size + 1, // as many times as the kernel fits
        false // fill zeros
    );
    auto& out_mat = out.w();
    auto& image_mat = image.w();
    int col=0,
        KSizeX = image.dims(0),
        SizeX  = image.dims(0),
        SizeY  = image.dims(1);
    vector<R> kernel_sums;
    kernel_sums.reserve(kernels.size());
    std::transform(kernels.begin(), kernels.end(), std::back_inserter(kernel_sums), [](const Mat<R>& kern) {
        return kern.w().sum();
    });

    for ( col = 0; col < out.dims(1); col++ ) {
        for (int i = 0; i < kernels.size();i++) {
            out_mat(i,col) = (image_mat.block(0, col, KSizeX, kern_col_size).array() * kernels[i].w().array()).sum() / kernel_sums[i];
        }
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([image, kernels, out, kernel_sums, kern_col_size](){
            auto& image_mat = image.w();
            int col=0,
                KSizeX = image.dims(0),
                SizeX  = image.dims(0),
                SizeY  = image.dims(1);
            bool grad_image = !image.constant;
            auto& out_grad = out.dw();
            auto& out_weight = out.w();
            std::shared_ptr<Eigen::Matrix<R, Eigen::Dynamic, 1>> surplus;
            bool computed_surplus = false;
            for (int i=0; i < kernels.size();i++) {
                if (!kernels[i].constant) {
                    if (!computed_surplus) {
                        surplus = make_shared<Eigen::Matrix<R, Eigen::Dynamic, 1>>((out_weight.array() * out_grad.array()).rowwise().sum());
                        computed_surplus = true;
                    }
                    kernels[i].dw().array() -= (*surplus)(i,0) / kernel_sums[i];
                }
            }
            for ( col = 0; col < out.dims(1); col++ ) {
                if (grad_image) {
                    for (int i=0; i < kernels.size();i++) {
                        image.dw().block(0, col, KSizeX, kern_col_size).noalias() += kernels[i].w() * (out_grad(i, col) / kernel_sums[i]);
                    }
                }
                for (int i=0; i < kernels.size();i++) {
                    if (!kernels[i].constant) {
                        kernels[i].dw().noalias() += (image_mat.block(0, col, KSizeX, kern_col_size).array() * (out_grad(i, col) / (kernel_sums[i]))).matrix();
                    }
                }
            }
        });
    }
    return out;
}

template class MatOps<float>;
template class MatOps<double>;
