#include "top_k.h"
#include "dali/array/op/reducers.h"
#include "dali/utils/make_message.h"
#include "dali/utils/assert2.h"
#include "dali/array/expression/computation.h"
#include "dali/array/expression/buffer.h"
#include "dali/array/jit/array_view.h"
#include "dali/utils/topn.h"

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>
#include <vector>

namespace op {
    struct TopK : public Expression {
        int k_;
        bool sorted_;
        bool reversed_;
        TopK(const Array& array, int k, bool sorted, bool reversed) :
            Expression({array.shape()[0], k}, DTYPE_INT32, {array,}),
            k_(k), sorted_(sorted), reversed_(reversed) {}

        memory::Device preferred_device() const {
            return arguments_[0].preferred_device();
        }

        expression_ptr copy() const {
            return std::make_shared<TopK>(arguments_[0], k_, sorted_, reversed_);
        }
    };
}

namespace {
    struct CpuTopKImpl : public Computation {
        using Computation::Computation;

        virtual void run() {
            auto arg_dtype = right_.expression()->arguments()[0].dtype();
            if (right_.expression()->arguments()[0].strides().empty()) {
                if (arg_dtype == DTYPE_INT32) {
                    run_dtype_contig<int>();
                } else if (arg_dtype == DTYPE_FLOAT) {
                    run_dtype_contig<float>();
                } else if (arg_dtype == DTYPE_DOUBLE) {
                    run_dtype_contig<double>();
                } else {
                    ASSERT2(false, utils::make_message(
                        "TopK unsupported dtype ", arg_dtype, "."));
                }
            } else {
                if (arg_dtype == DTYPE_INT32) {
                    run_dtype_strided<int>();
                } else if (arg_dtype == DTYPE_FLOAT) {
                    run_dtype_strided<float>();
                } else if (arg_dtype == DTYPE_DOUBLE) {
                    run_dtype_strided<double>();
                } else {
                    ASSERT2(false, utils::make_message(
                        "TopK unsupported dtype ", arg_dtype, "."));
                }
            }
        }

        template<typename T>
        void run_dtype_contig() {
            Buffer* right = op::static_as_buffer(right_.expression()->arguments()[0]);
            auto right_arr = make_view<T, 2>(
                static_cast<T*>(right->memory_->readonly_data(memory::Device::cpu())),
                right->offset_,
                right->shape_.data());
            compute<T>(right_arr);
        }

        template<typename T>
        void run_dtype_strided() {
            Buffer* right = op::static_as_buffer(right_.expression()->arguments()[0]);
            auto right_arr = make_strided_view<T, 2>(
                static_cast<T*>(right->memory_->readonly_data(memory::Device::cpu())),
                right->offset_,
                right->shape_.data(),
                right->strides_.data());
            compute<T>(right_arr);
        }

        template<typename T, typename Source>
        void compute(const Source& right_arr) {
            Buffer* dest = op::static_as_buffer(left_);
            auto indices_arr = make_view<int, 2>(
                static_cast<int*>(dest->memory_->overwrite_data(memory::Device::cpu())),
                dest->offset_,
                dest->shape_.data());
            // Array values(dest->shape_, right_.dtype(), device);
            // auto values_arr = make_view<T, 2>(
            //     static_cast<T*>(values.memory()->overwrite_data(device)),
            //     values.offset(),
            //     values.shape().data());

            int num_cols = right_arr.shape()[1];
            auto op = static_cast<op::TopK*>(right_.expression().get());
            int k = op->k_;
            bool sorted = op->sorted_;
            bool reversed = op->reversed_;

            for (int batch_idx = 0; batch_idx < right_arr.shape()[0]; batch_idx++) {
                std::function<bool(const int, const int)> stable_comp;
                std::function<bool(const int, const int)> comp;
                if (reversed) {
                    stable_comp = [&right_arr, batch_idx](const int a, const int b) {
                        if (right_arr[{batch_idx, b}] > right_arr[{batch_idx, a}]) {
                            return true;
                        } else if (right_arr[{batch_idx, b}] < right_arr[{batch_idx, a}]) {
                            return false;
                        } else {
                            return a > b;
                        }
                    };
                    comp = [&right_arr, batch_idx](const int a, const int b) {
                        return right_arr[{batch_idx, b}] > right_arr[{batch_idx, a}];
                    };
                } else {
                    stable_comp = [&right_arr, batch_idx](const int a, const int b) {
                        if (right_arr[{batch_idx, b}] < right_arr[{batch_idx, a}]) {
                            return true;
                        } else if (right_arr[{batch_idx, b}] > right_arr[{batch_idx, a}]) {
                            return false;
                        } else {
                            return a < b;
                        }
                    };
                    comp = [&right_arr, batch_idx](const int a, const int b) {
                        return right_arr[{batch_idx, b}] < right_arr[{batch_idx, a}];
                    };
                }
                if (k == num_cols) {
                    auto* begin = &indices_arr[{batch_idx, 0}];
                    auto* end = &indices_arr[{batch_idx, k}];
                    // Set the initial array of indices 0 ... k - 1.
                    std::iota(begin, end, 0);
                    // We want an in-place sort, but we can cheat because we're sorting
                    // indices that started out sorted.  First, do a std::sort, which
                    // is notably faster than std::stable_sort.
                    std::sort(begin, end, comp);
                    // Then, for runs of adjacent elements that were equal, sort the
                    // indices in those runs in increasing order.
                    for (auto* run_begin = begin; run_begin != end;) {
                        auto* run_end = run_begin + 1;
                        if (run_end == end) break;
                        if (right_arr[{batch_idx, *run_begin}] == right_arr[{batch_idx, *run_end}]) {
                            while (++run_end != end) {
                                if (right_arr[{batch_idx, *run_begin}] != right_arr[{batch_idx, *run_end}]) break;
                            }
                            std::sort(run_begin, run_end);
                        }
                        run_begin = run_end;
                    }
                } else {
                    // Use the TopN heap object to sort.
                    utils::TopN<int, decltype(stable_comp)> filter(k, stable_comp);
                    filter.reserve(num_cols);
                    for (int c = 0; c < num_cols; ++c) {
                        filter.push(c);
                    }
                    int i = 0;
                    if (sorted) {
                        std::unique_ptr<std::vector<int>> top_k(filter.Extract());
                        for (auto top_k_it = top_k->begin(); top_k_it != top_k->end();
                             ++top_k_it, ++i) {
                            indices_arr[{batch_idx, i}] = *top_k_it;
                        }
                    } else {
                        for (auto top_k_it = filter.unsorted_begin();
                             top_k_it != filter.unsorted_end(); ++top_k_it, ++i) {
                            indices_arr[{batch_idx, i}] = *top_k_it;
                        }
                    }
                }
                // TODO(jonathan): add multiple outputs to enable sort and sort(axis):
                //
                // Now that the indices are sorted, copy the values over in
                // sorted order:
                // std::transform(&indices_arr[{b, 0}], &indices_arr[{b, k}], &values_arr[{b, 0}],
                //                [b, &right_arr](const int loc) { return right_arr[{b, loc}]; });
            }
        }
    };

    std::vector<int> top_k_shape(std::vector<int> shape, int k) {
        shape.back() = k;
        return shape;
    }
    int cpu_top_k_impl = register_implementation_default<op::TopK, CpuTopKImpl>();
}

namespace op {
    // TODO(jonathan): add gpu impl of topk
    Array top_k(const Array& array, int k, bool sorted) {
        ASSERT2(k >= 1, utils::make_message(
            "top_k's k argument must be >= 1, but got k = ", k, "."));
        if (array.number_of_elements() == 1) {
            return Array::zeros(array.shape(), DTYPE_INT32);
        }
        if (k == 1) return op::argmax(array, -1).expand_dims(-1);
        return Array(std::make_shared<TopK>(array.reshape({-1, array.shape().back()}), k, sorted, false)
        ).reshape(top_k_shape(array.shape(), k));
    }

    Array bottom_k(const Array& array, int k, bool sorted) {
        ASSERT2(k >= 1, utils::make_message(
            "bottom_k's k argument must be >= 1, but got k = ", k, "."));
        if (array.number_of_elements() == 1) {
            return Array::zeros(array.shape(), DTYPE_INT32);
        }
        if (k == 1) return op::argmin(array, -1).expand_dims(-1);
        return Array(std::make_shared<TopK>(array.reshape({-1, array.shape().back()}), k, sorted, true)
        ).reshape(top_k_shape(array.shape(), k));
    }
}
