#ifndef DALI_TEST_UTILS_H
#define DALI_TEST_UTILS_H


#include "dali/config.h"

#ifdef GPERFTOOLS_FOUND
    #include <gperftools/heap-checker.h>
#endif

#include <gtest/gtest.h>
#include <memory>

#include "dali/array/tests/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/tape.h"


using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;


// #ifdef GPERFTOOLS_FOUND
//     class MemorySafeTest : public ::testing::Test {
//         private:
//             std::shared_ptr<HeapLeakChecker> heap_checker;
//         protected:
//             virtual void SetUp() {
//                 graph::clear();
//                 if (HeapLeakChecker::IsActive())
//                     heap_checker = std::make_shared<HeapLeakChecker>("memory_leak_checker");
//             }

//             virtual void TearDown() {
//                 if (HeapLeakChecker::IsActive())
//                     ASSERT_TRUE(heap_checker->NoLeaks()) << "Memory Leak";

//                 graph::clear();
//             }

//     };
// #else
    class MemorySafeTest : public ::testing::Test {
        protected:
            virtual void SetUp() {
                graph::clear();
            }
            virtual void TearDown() {
                graph::clear();
            }
    };
// #endif

typedef Tensor(* vector_tensor_op)(const std::vector<Tensor>&);

#ifdef DALI_USE_CUDA
const double DEFAULT_GRAD_EPS=1e-3;
#define SCALAR_COMP_LE ::testing::DoubleLE
#define SCALAR_COMP_GE ::testing::DoubleGE
#else
const double DEFAULT_GRAD_EPS=1e-7;
#define SCALAR_COMP_LE ::testing::DoubleLE
#define SCALAR_COMP_GE ::testing::DoubleGE
#endif

#ifdef DALI_USE_CUDA
    // numeric gradient is slow on GPU.
    #define NUM_RETRIES 2
#else
    #define NUM_RETRIES 10
#endif

#define EXPERIMENT_REPEAT for(int __repetition=0; __repetition < NUM_RETRIES; ++__repetition)

#define ASSERT_MATRIX_EQ(A, B) ASSERT_TRUE(ArrayOps::equals((A).w,(B).w))
#define ASSERT_MATRIX_NEQ(A, B) ASSERT_FALSE(ArrayOps::equals((A).w,(B).w))
#define ASSERT_MATRIX_CLOSE(A, B, eps) ASSERT_TRUE(ArrayOps::allclose((A).w,(B).w, (eps)))
#define ASSERT_MATRIX_NOT_CLOSE(A, B, eps) ASSERT_FALSE(ArrayOps::allclose((A).w,(B).w, (eps)))
#define ASSERT_MATRIX_GRAD_CLOSE(A, B, eps) ASSERT_TRUE(ArrayOps::allclose((A).dw,(B).dw,(eps)))
#define ASSERT_MATRIX_GRAD_NOT_CLOSE(A, B, eps) ASSERT_FALSE(ArrayOps::allclose((A).dw,(B).dw,(eps)))

#define EXPECT_MATRIX_EQ(A, B) EXPECT_TRUE(ArrayOps::equals((A).w,(B).w))
#define EXPECT_MATRIX_NEQ(A, B) EXPECT_FALSE(ArrayOps::equals((A).w,(B).w))
#define EXPECT_MATRIX_CLOSE(A, B, eps) EXPECT_TRUE(ArrayOps::allclose((A).w,(B).w, (eps)))
#define EXPECT_MATRIX_NOT_CLOSE(A, B, eps) EXPECT_FALSE(ArrayOps::allclose((A).w,(B).w, (eps)))
#define EXPECT_MATRIX_GRAD_CLOSE(A, B, eps) EXPECT_TRUE(ArrayOps::allclose((A).dw,(B).dw,(eps)))
#define EXPECT_MATRIX_GRAD_NOT_CLOSE(A, B, eps) EXPECT_FALSE(ArrayOps::allclose((A).dw,(B).dw,(eps)))

#ifdef DALI_USE_CUDA
#define ASSERT_MAT_ON_GPU(A) ASSERT_TRUE((A).w.memory()->is_fresh(memory::Device::gpu(0)))
#define EXPECT_MAT_ON_GPU(A) EXPECT_TRUE((A).w.memory()->is_fresh(memory::Device::gpu(0)))
#else
#define ASSERT_MAT_ON_GPU(A)
#define EXPECT_MAT_ON_GPU(A)
#endif




// restrict to this compilation unit - avoid linking errors.
namespace {

    AssertionResult buffer_is_nonzero(const Array& buffer) {
        for (int i=0; i<buffer.number_of_elements(); ++i) {
            if (std::abs((double)buffer(i)) > 1e-5) {
                return AssertionSuccess();
            }
        }
        return AssertionFailure() << "buffer is all zeros";
    }

    void print_buffer(const Array& arr, int highlight) {
        std::cout << "[";
        int num = arr.number_of_elements();
        for (uint i = 0; i < num; ++i) {
            if (i == highlight) {
                std::cout << utils::red << std::setw( 7 ) // keep 7 digits
                          << std::setprecision( 3 ) // use 3 decimals
                          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                          << (double)arr(i) << utils::reset_color;
            } else {
                std::cout << std::setw( 7 ) // keep 7 digits
                          << std::setprecision( 3 ) // use 3 decimals
                          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                          << (double)arr(i);
            }
            if (i != num - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    AssertionResult buffer_almost_equals(const Array& buffer1, const Array& buffer2, const double& eps) {
        if (buffer1.number_of_elements() != buffer2.number_of_elements()) {
            return AssertionFailure() << "Not the same number of elements in buffer1 as buffer2 (got "
                                      << buffer1.number_of_elements() << " != " << buffer2.number_of_elements()
                                      << ").";
        }
        for (int i = 0; i < buffer1.number_of_elements(); ++i) {
            if (std::abs((double)buffer1(i) - (double)buffer2(i)) > eps) {
                return AssertionFailure() << "Difference in datum " << i << " first tensor has "
                                          << (double)buffer1(i) << ", while second has " << (double)buffer2(i)
                                          << "(tolerance = " << eps << ")";
            }
        }
        return AssertionSuccess();
    }

    AssertionResult buffer_ratio_almost_equals(const Array& buffer1, const Array& buffer2, const double& eps) {
        if (buffer1.number_of_elements() != buffer2.number_of_elements()) {
            return AssertionFailure() << "Not the same number of elements in buffer1 as buffer2 (got "
                                      << buffer1.number_of_elements() << " != " << buffer2.number_of_elements()
                                      << ").";
        }
        for (int i = 0; i < buffer1.number_of_elements(); ++i) {
            if (std::abs((double)buffer2(i)) > eps &&
                std::abs((double)buffer1(i)) > eps &&
                std::abs(((double)buffer1(i) / (double)buffer2(i)) -1.0) > eps) {
                return AssertionFailure() << "Difference in ratio at position " << i << " first tensor has "
                                          << (double)buffer1(i) << ", while second has " << (double)buffer2(i)
                                          << "(tolerance = " << 100.0 * eps << ")";
            }
        }
        return AssertionSuccess();
    }

    /**
    Gradient Same
    -------------

    Numerical gradient checking method. Performs
    a finite difference estimation of the gradient
    over each argument to a functor.

    **/

    void expect_computation_on_gpu(
            std::function<Tensor(std::vector<Tensor>&)> functor,
            std::vector<Tensor> arguments) {
        graph::NoBackprop nb;
        auto res = functor(arguments);
        EXPECT_MAT_ON_GPU(res);
        for (auto& arg : arguments) {
            EXPECT_MAT_ON_GPU(arg);
        }
    }

    void expect_args_remain_on_gpu(
            std::function<Tensor(std::vector<Tensor>&)> functor,
            std::vector<Tensor> arguments) {
        graph::NoBackprop nb;
        auto res = functor(arguments);
        for (auto& arg : arguments) {
            EXPECT_MAT_ON_GPU(arg);
        }
    }

    bool cpu_vs_gpu(std::function<Tensor(std::vector<Tensor>&)> functor,
                    std::vector<Tensor> arguments,
                    const double& tolerance = 1e-5) {
        #ifdef DALI_USE_CUDA
            for (auto arg: arguments) {
                arg.w.memory()->to_gpu(0);
                arg.dw.memory()->to_gpu(0);
            }

            auto res_gpu = functor(arguments);
            if (!res_gpu.w.memory()->is_fresh(memory::Device::gpu(0))) {
                std::cout << "Computation did not happen on GPU." << std::endl;
                return false;
            }
            std::vector<Tensor> gpu_gradients;
            //TODO(jonathan,szymon): generalize to-ND
            for (auto arg : arguments) {
                gpu_gradients.push_back(Tensor::zeros_like(arg));
                for (int i = 0; i < arg.number_of_elements(); i++) {
                    gpu_gradients.back().dw(i).assign(arg.dw(i)).eval();
                }
            }

            {
                memory::WithDevicePreference device_prefence(memory::Device::cpu());

                for (auto arg: arguments) {
                    arg.w.memory()->to_cpu();
                    arg.dw.memory()->to_cpu();
                }
                Tensor res_cpu = functor(arguments);

                if (!buffer_almost_equals(res_cpu.w, res_gpu.w, tolerance)) {
                    std::cout << "Forward step produces different results on gpu and cpu.";
                    return false;
                }
                for (int i = 0; i < arguments.size(); ++i) {
                    if(!buffer_almost_equals(gpu_gradients[i].dw, arguments[i].dw, tolerance)) {
                        std::cout << "backward step produces different results on gpu and cpu.";
                        return false;
                    }
                }
            }
            return true;
        #else
            return true;
        #endif

    }

    void print_disagreement(const Array& fd_grad, const Array& arg_dw) {
        ASSERT2(fd_grad.number_of_elements() == arg_dw.number_of_elements(),
            "arguments to print_disagreement must have the same number_of_elements()");
        // find biggest disagreement:
        double max_disagreement = 0;
        int loc_disagreement = 0;
        for (int i = 0; i < fd_grad.number_of_elements(); i++) {
            double disagreement = std::abs((double)fd_grad(i) - (double)arg_dw(i));
            if (disagreement > max_disagreement) {
                max_disagreement = disagreement;
                loc_disagreement = i;
            }
        }
        // print it:
        auto start = std::max(loc_disagreement - 6, 0);
        auto length = std::min(fd_grad.number_of_elements() - start, 12);
        std::cout << "-----------\nfd_grad[" << start << ":" << start + length << "] = " << std::endl;
        print_buffer((Array)fd_grad.ravel()[Slice(start, start + length)], loc_disagreement - start);
        std::cout << "-----------\n arg.dw[" << start << ":" << start + length << "] = " << std::endl;
        print_buffer((Array)arg_dw.ravel()[Slice(start, start + length)], loc_disagreement - start);
        std::cout << "-----------" << std::endl;
        std::cout << "Largest disparity = "
                  << std::setw( 7 ) // keep 7 digits
                  << std::setprecision( 5 ) // use 3 decimals
                  << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                  << max_disagreement << std::endl;
    }


    template<typename T>
    Array finite_difference_dtype_helper(
            std::function<Tensor()> functor,
            Tensor& arg,
            const double& grad_epsilon) {
        Array res_grad(arg.shape(), arg.dtype());

        graph::NoBackprop nb;

        for (int i = 0; i < arg.number_of_elements(); i++) {
            const T prev_val    = (T)arg.w(i);
            arg.w(i).assign(prev_val + grad_epsilon).eval();
            T obj_positive      = (T)Array(functor().w.sum());
            arg.w(i).assign(prev_val - grad_epsilon).eval();
            T obj_negative      = (T)Array(functor().w.sum());
            // return to previous value
            arg.w(i).assign(prev_val).eval();
            res_grad(i).assign((obj_positive - obj_negative) / (2.0 * grad_epsilon)).eval();
        }
        return res_grad;
    }

    Array finite_difference(std::function<Tensor()> functor,
                            Tensor& arg, const double& grad_epsilon) {
        if (arg.dtype() == DTYPE_FLOAT) {
            return finite_difference_dtype_helper<float>(functor, arg, grad_epsilon);
        } else if (arg.dtype() == DTYPE_DOUBLE) {
            return finite_difference_dtype_helper<double>(functor, arg, grad_epsilon);
        } else {
            ASSERT2(false, "cannot take finite_difference derivative of an integer variable.");
            return Array();
        }
    }

    bool gradient_comparison_helper(std::function<AssertionResult(const Array&, const Array&, const double&)> checker,
                                    std::function<Tensor(std::vector<Tensor>&)> functor,
                                    std::vector<Tensor> arguments,
                                    const double& tolerance,
                                    const double& grad_epsilon,
                                    const bool& fail_on_zero_gradient) {
        auto error = functor(arguments);
        error.grad();
        graph::backward();
        error.dw.eval();
        bool worked_out = true;
        // from now on gradient is purely numerical:

        int param_idx = 1;
        auto bound_functor = [&functor,&arguments](){return functor(arguments);};

        for (auto& arg : arguments) {
            auto fd_grad = finite_difference(bound_functor, arg, grad_epsilon);
            AssertionResult did_work_out = checker(fd_grad, arg.dw, tolerance);
            if (fail_on_zero_gradient) {
                auto is_nonzero = (bool)buffer_is_nonzero(fd_grad);
                if (is_nonzero == false) {
                    std::cout << "Gradient for parameter " << param_idx << " (" << arg << ") should not be all zeros." << std::endl;
                    return false;
                }
            }
            // AssertionResult is a GoogleTest magic and it's castable to bool.
            if (!did_work_out) {
                std::cout << "Gradient for parameter " << param_idx << " is incorrect:" << std::endl;
                print_disagreement(fd_grad, arg.dw);
            }
            worked_out = worked_out && (bool)did_work_out;
            if (!worked_out) break;
            param_idx++;
        }
        return worked_out;
    }

    bool gradient_same(std::function<Tensor(std::vector<Tensor>&)> functor,
                       std::vector<Tensor> arguments,
                       const double& tolerance = 1e-5,
                       const double& grad_epsilon = DEFAULT_GRAD_EPS,
                       const bool& fail_on_zero_gradient = true) {
        return gradient_comparison_helper(buffer_almost_equals,
                                          functor,
                                          arguments,
                                          tolerance,
                                          grad_epsilon,
                                          fail_on_zero_gradient);
    }

    bool gradient_ratio_same(std::function<Tensor(std::vector<Tensor>&)> functor,
                             std::vector<Tensor> arguments,
                             const double& tolerance = 1e-5,
                             const double& grad_epsilon = DEFAULT_GRAD_EPS,
                             const bool& fail_on_zero_gradient = true) {
        return gradient_comparison_helper(buffer_ratio_almost_equals,
                                          functor,
                                          arguments,
                                          tolerance,
                                          grad_epsilon,
                                          fail_on_zero_gradient);
    }
}

#endif
