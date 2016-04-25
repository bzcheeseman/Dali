#ifndef DALI_TEST_UTILS_H
#define DALI_TEST_UTILS_H


#include "dali/config.h"

#ifdef GPERFTOOLS_FOUND
    #include <gperftools/heap-checker.h>
#endif

#include <gtest/gtest.h>
#include <memory>


#include "dali/test_utils.h"
#include "dali/tensor/tensor.h"
#include "dali/tensor/tape.h"
// #include "dali/tensor/MatOps.h"

using ::testing::AssertionResult;
using ::testing::AssertionSuccess;
using ::testing::AssertionFailure;


#ifdef GPERFTOOLS_FOUND
    class MemorySafeTest : public ::testing::Test {
        private:
            std::shared_ptr<HeapLeakChecker> heap_checker;
        protected:
            virtual void SetUp() {
                if (HeapLeakChecker::IsActive())
                    heap_checker = std::make_shared<HeapLeakChecker>("memory_leak_checker");
            }

            virtual void TearDown() {
                if (HeapLeakChecker::IsActive())
                    ASSERT_TRUE(heap_checker->NoLeaks()) << "Memory Leak";
            }

    };
#else
    class MemorySafeTest : public ::testing::Test {
    };
#endif

#ifdef DALI_USE_CUDA
// most gpus often don't support double
typedef double R;
const double DEFAULT_GRAD_EPS=1e-3;
#define SCALAR_COMP_LE ::testing::DoubleLE
#define SCALAR_COMP_GE ::testing::DoubleGE
#else
typedef double R;
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

template<typename T>
AssertionResult buffer_equals (T* buffer1, T* buffer2, uint size1, uint size2) {
    if (size1 != size2)
        return AssertionFailure() << "Sizes differ first matrix is " << size1 << ", while second is " << size2;
    for (int i=0; i<size1; ++i) {
        if (buffer1[i] != buffer2[i])
            return AssertionFailure() << "Difference in datum " << i << " first tensor has "
                                      << buffer1[i] << ", while second has " << buffer2[i];
    }
    return AssertionSuccess();
}

template<typename T>
AssertionResult buffer_is_nonzero(T* buffer, uint size) {
    for (int i=0; i<size; ++i) {
        if (std::abs(buffer[i]) > 1e-5) {
            return AssertionSuccess();
        }
    }
    return AssertionFailure() << "buffer is all zeros";
}

template<typename T>
void print_buffer(T* buffer, int num) {
    std::cout << "[";
    for (uint i = 0; i < num; ++i) {
        std::cout << std::setw( 7 ) // keep 7 digits
                  << std::setprecision( 3 ) // use 3 decimals
                  << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                  << buffer[i];
        if (i != num - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template<typename T>
void print_buffer(T* buffer, int num, int highlight) {
    std::cout << "[";
    for (uint i = 0; i < num; ++i) {
        if (i == highlight) {
            std::cout << utils::red << std::setw( 7 ) // keep 7 digits
                      << std::setprecision( 3 ) // use 3 decimals
                      << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                      << buffer[i] << utils::reset_color;
        } else {
            std::cout << std::setw( 7 ) // keep 7 digits
                      << std::setprecision( 3 ) // use 3 decimals
                      << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                      << buffer[i];
        }
        if (i != num - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template<typename T, typename J>
AssertionResult buffer_almost_equals(T* buffer1, T* buffer2, uint size1, uint size2, J eps) {
    if (size1 != size2)
        return AssertionFailure() << "Sizes differ first matrix is " << size1 << ", while second is " << size2;
    for (int i=0; i<size1; ++i) {
        if (std::abs(buffer1[i] - buffer2[i]) > eps) {
            return AssertionFailure() << "Difference in datum " << i << " first tensor has "
                                      << buffer1[i] << ", while second has " << buffer2[i]
                                      << "(tolerance = " << eps << ")";
        }
    }
    return AssertionSuccess();
}

template<typename T, typename J>
AssertionResult buffer_ratio_almost_equals(T* buffer1, T* buffer2, uint size1, uint size2, J eps) {
    if (size1 != size2)
        return AssertionFailure() << "Sizes differ first matrix is " << size1 << ", while second is " << size2;
    for (int i=0; i<size1; ++i) {
        if (std::abs(buffer2[i]) > eps && std::abs(buffer1[i]) > eps && std::abs((buffer1[i]/ buffer2[i]) - 1.0) > eps) {
            return AssertionFailure() << "Difference in ratio datum " << i << " first tensor has "
                                      << buffer1[i] << ", while second has " << buffer2[i]
                                      << "(tolerance = " << 100.0 * eps << ")";
        }
    }
    return AssertionSuccess();
}

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
        for (auto& arg: arguments) {
            EXPECT_MAT_ON_GPU(arg);
        }
    }

    void expect_args_remain_on_gpu(
            std::function<Tensor(std::vector<Tensor>&)> functor,
            std::vector<Tensor> arguments) {
        graph::NoBackprop nb;
        auto res = functor(arguments);
        for (auto& arg: arguments) {
            EXPECT_MAT_ON_GPU(arg);
        }
    }

    bool cpu_vs_gpu(
            std::function<Tensor(std::vector<Tensor>&)> functor,
            std::vector<Tensor> arguments,
            R tolerance    = 1e-5) {
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
                    gpu_gradients.back().dw(i) = arg.dw(i);
                }
            }

            {
                memory::WithDevicePreference device_prefence(memory::Device::cpu());

                for (auto arg: arguments) {
                    arg.w.memory()->to_cpu();
                    arg.dw.memory()->to_cpu();
                }
                Tensor res_cpu = functor(arguments);

                if (!buffer_almost_equals(
                        (R*)res_cpu.w.memory()->readonly_data(memory::Device::cpu()),
                        (R*)res_gpu.w.memory()->readonly_data(memory::Device::cpu()),
                        res_cpu.number_of_elements(),
                        res_gpu.number_of_elements(),
                        tolerance)) {
                    std::cout << "Forward step produces different results on gpu and cpu.";
                    return false;
                }
                for (int i = 0; i < arguments.size(); ++i) {
                    if(!buffer_almost_equals(
                            (R*)gpu_gradients[i].dw.memory()->data(memory::Device::cpu()),
                            (R*)arguments[i].dw.memory()->data(memory::Device::cpu()), //cpu gradient
                            gpu_gradients[i].dw.number_of_elements(),
                            arguments[i].number_of_elements(),
                            tolerance)) {
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

    bool gradient_same(
            std::function<Tensor(std::vector<Tensor>&)> functor,
            std::vector<Tensor> arguments,
            R tolerance    = 1e-5,
            R grad_epsilon = DEFAULT_GRAD_EPS,
            bool fail_on_zero_gradient = true) {

        auto error = functor(arguments);

        error.grad();

        graph::backward();

        bool worked_out = true;

        // from now on gradient is purely numerical:
        graph::NoBackprop nb;
        int param_idx = 1;
        for (auto& arg : arguments) {
            R  Arg_prime[arg.number_of_elements()];
            for (int i = 0; i < arg.number_of_elements(); i++) {
                const R prev_val = arg.w(i);
                arg.w(i)            = prev_val +  grad_epsilon;
                R obj_positive      = (R)Array(functor(arguments).w.sum());
                arg.w(i)            = prev_val - grad_epsilon;
                R obj_negative      = (R)Array(functor(arguments).w.sum());
                arg.w(i)            = prev_val;
                Arg_prime[i]        = (obj_positive - obj_negative) / (2.0 * grad_epsilon);
            }
            AssertionResult did_work_out = buffer_almost_equals(
                    (R*)Arg_prime,
                    (R*)arg.dw.memory()->readonly_data(memory::Device::cpu()),
                    arg.number_of_elements(),
                    arg.number_of_elements(),
                    tolerance);
            if (fail_on_zero_gradient) {
                auto is_nonzero = buffer_is_nonzero((R*)Arg_prime, arg.number_of_elements());
                if (((bool)is_nonzero) == false) {
                    std::cout << "Gradient for parameter " << param_idx << " (" << arg << ") should not be all zeros." << std::endl;
                    if (arg.name != nullptr) {
                        std::cout << "arg.name = " << *arg.name << std::endl;
                    }
                    return false;
                }
            }
            // AssertionResult is a GoogleTest magic and it's castable to bool.
            if (!did_work_out) {
                R max_disagreement = 0;
                int loc_disagreement = 0;
                for (int i = 0; i < arg.number_of_elements(); i++) {
                    auto disagreement = std::abs(Arg_prime[i] - (R)arg.dw(i));
                    if (disagreement > max_disagreement) {
                        max_disagreement = disagreement;
                        loc_disagreement = i;
                    }
                }

                auto start = std::max(loc_disagreement - 6, 0);
                auto length = std::min(arg.number_of_elements() - start, 12);

                std::cout << "-----------\nArg_prime[" << start << ":" << start + length << "] = " << std::endl;
                print_buffer((R*)Arg_prime + start,       length, loc_disagreement - start);
                std::cout << "-----------\n arg.dw()[" << start << ":" << start + length << "] = " << std::endl;
                print_buffer(
                    (R*)arg.dw.memory()->readonly_data(memory::Device::cpu()) + start,
                    length, loc_disagreement - start
                );
                if (arg.name != nullptr) {
                    std::cout << "arg.name = " << *arg.name << std::endl;
                }
                std::cout << "-----------" << std::endl;

                std::cout << "Largest disparity = "
                          << std::setw( 7 ) // keep 7 digits
                          << std::setprecision( 5 ) // use 3 decimals
                          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                          << max_disagreement << std::endl;


            }
            worked_out = worked_out && (bool)did_work_out;
            if (!worked_out) {
                break;
            }
            param_idx++;
        }
        return worked_out;
    }

    // TODO(jonathan) merge with function above
    bool gradient_ratio_same(
            std::function<Tensor(std::vector<Tensor>&)> functor,
            std::vector<Tensor> arguments,
            R tolerance    = 1e-5,
            R grad_epsilon = DEFAULT_GRAD_EPS,
            bool fail_on_zero_gradient = true) {

        auto error = functor(arguments);

        error.grad();

        graph::backward();

        bool worked_out = true;

        // from now on gradient is purely numerical:
        graph::NoBackprop nb;
        int param_idx = 1;
        for (auto& arg : arguments) {
            R  Arg_prime[arg.number_of_elements()];
            for (int i = 0; i < arg.number_of_elements(); i++) {
                const R prev_val    = arg.w(i);
                arg.w(i)            = prev_val +  grad_epsilon;
                auto obj_positive   = (R)Array(functor(arguments).w.sum());
                arg.w(i)            = prev_val - grad_epsilon;
                auto obj_negative   = (R)Array(functor(arguments).w.sum());
                arg.w(i)            = prev_val;
                Arg_prime[i]        = (obj_positive - obj_negative) / (2.0 * grad_epsilon);
            }
            AssertionResult did_work_out = buffer_ratio_almost_equals(
                    (R*)Arg_prime,
                    (R*)arg.dw.memory()->readonly_data(memory::Device::cpu()),
                    arg.number_of_elements(),
                    arg.number_of_elements(),
                    tolerance);
            if (fail_on_zero_gradient) {
                auto is_nonzero = buffer_is_nonzero((R*)Arg_prime, arg.number_of_elements());
                if (((bool)is_nonzero) == false) {
                    std::cout << "Gradient for parameter " << param_idx << " (" << arg << ") should not be all zeros." << std::endl;
                    if (arg.name != nullptr) {
                        std::cout << "arg.name = " << *arg.name << std::endl;
                    }
                    return false;
                }
            }
            // AssertionResult is a GoogleTest magic and it's castable to bool.
            if (!did_work_out) {
                R max_disagreement = 0;
                int loc_disagreement = 0;
                for (int i = 0; i < arg.number_of_elements(); i++) {
                    auto disagreement = std::abs((Arg_prime[i] / (R)arg.dw(i)) - (R)1.0);
                    if (disagreement > max_disagreement) {
                        max_disagreement = disagreement;
                        loc_disagreement = i;
                    }
                }

                auto start = std::max(loc_disagreement - 6, 0);
                auto length = std::min(arg.number_of_elements() - start, 12);

                std::cout << "-----------\nArg_prime[" << start << ":" << start + length << "] = " << std::endl;
                print_buffer((R*)Arg_prime + start,       length, loc_disagreement - start);
                std::cout << "-----------\n arg.dw[" << start << ":" << start + length << "] = " << std::endl;
                print_buffer((R*)arg.dw.memory()->readonly_data(memory::Device::cpu()) + start, length, loc_disagreement - start);
                if (arg.name != nullptr) {
                    std::cout << "arg.name = " << *arg.name << std::endl;
                }
                std::cout << "-----------" << std::endl;

                std::cout << "Largest disparity = "
                          << std::setw( 7 ) // keep 7 digits
                          << std::setprecision( 5 ) // use 3 decimals
                          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                          << max_disagreement << std::endl;


            }
            worked_out = worked_out && (bool)did_work_out;
            if (!worked_out) {
                break;
            }
            param_idx++;
        }
        return worked_out;
    }
}

#endif
