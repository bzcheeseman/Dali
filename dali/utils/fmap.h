#ifndef DALI_UTILS_FMAP_H
#define DALI_UTILS_FMAP_H

#include <vector>

namespace utils {
    template<typename IN, typename Mapper>
    auto fmap(const std::vector<IN>& in_list, Mapper f) ->
            std::vector<decltype(f(std::declval<IN>()))> {
        std::vector<decltype(f(std::declval<IN>()))> out_list;
        out_list.reserve(in_list.size());
        for (const IN& in_element: in_list) {
            out_list.push_back(f(in_element));
        }
        return out_list;
    }
}  // namespace utils

#endif
