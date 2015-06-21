#ifndef SHALLOW_COPY_MAT_H
#define SHALLOW_COPY_MAT_H
#include <vector>
#include <functional>

#include "dali/tensor/Mat.h"
/**
Header only shallow copy creator
for handling hogwild copies of models
for parallel asynchronous optimization.
**/
namespace utils {
    // as described here:
    // http://stackoverflow.com/questions/20373466/nested-c-template-parameters-for-functions
    template <typename R, template <typename> class M>
    std::tuple<std::vector<M<R>>, std::vector<std::vector<Mat<R>>>> shallow_copy(const M<R>& model, int num_copies) {
        std::tuple<std::vector<M<R>>, std::vector<std::vector<Mat<R>>>> copies;
        for (int i = 0; i < num_copies; i++) {
            // create a copy for each training thread
            // (shared memory mode = Hogwild)
            std::get<0>(copies).push_back(model.shallow_copy());

            auto tmodel_params = std::get<0>(copies).back().parameters();
            // take a slice of all the parameters except for embedding.
            std::get<1>(copies).emplace_back(
                tmodel_params.begin(),
                tmodel_params.end()
            );
        }
        return copies;
    }
    // pass a lamdba as third argument to decide which bin to place parameters in (split into 2 bins)
    template <typename R, template <typename> class M, typename UnaryPredicate>
    std::tuple<std::vector<M<R>>, std::vector<std::vector<Mat<R>>>, std::vector<std::vector<Mat<R>>>> shallow_copy_multi_params(
            const M<R>& model, int num_copies, UnaryPredicate bin_function) {
        std::tuple<std::vector<M<R>>, std::vector<std::vector<Mat<R>>>, std::vector<std::vector<Mat<R>>>> copies;
        auto bin_yang_function = [&bin_function](const Mat<R>& mat) {return !bin_function(mat);};
        for (int i = 0; i < num_copies; i++) {
            // create a copy for each training thread
            // (shared memory mode = Hogwild)
            std::get<0>(copies).push_back(model.shallow_copy());

            auto tmodel_params = std::get<0>(copies).back().parameters();
            // take a slice of all the parameters except for embedding.
            decltype(tmodel_params) tmodel_param_ying(tmodel_params.size());

            auto it_ying = std::copy_if(tmodel_params.begin(), tmodel_params.end(), tmodel_param_ying.begin(), bin_function);
            tmodel_param_ying.resize(std::distance(tmodel_param_ying.begin(), it_ying));

            decltype(tmodel_params) tmodel_param_yang(tmodel_params.size() - tmodel_param_ying.size());
            auto it_yang = std::copy_if(tmodel_params.begin(), tmodel_params.end(), tmodel_param_yang.begin(), bin_yang_function);
            assert(it_yang == tmodel_param_yang.end());

            std::get<1>(copies).emplace_back(tmodel_param_ying);
            std::get<2>(copies).emplace_back(tmodel_param_yang);
        }
        return copies;
    }
}
#endif
