#ifndef DALI_UTILS_CONCATENATE_H
#define DALI_UTILS_CONCATENATE_H

#include <initializer_list>
#include <vector>

namespace utils {
    template<typename T>
    std::vector<T> concatenate(std::initializer_list<std::vector<T>> lists) {
        std::vector<T> concatenated_list;
        for (auto& list: lists) {
            for (const T& el: list) {
                concatenated_list.emplace_back(el);
            }
        }
        return concatenated_list;
    }
}  // namespace utils

#endif
