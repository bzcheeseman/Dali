#ifndef DALI_UTILS_UNFOLD_TUPLE_H
#define DALI_UTILS_UNFOLD_TUPLE_H

/* C++ equivalent of *args from Python */
#include <tuple>

// template<typename FunctT, typename... Args>
// auto unpack_tuple(FunctT f, const std::tuple<Args...>& params) -> typename std::result_of<FunctT>::type;

template<typename FunctT, typename... Args>
auto unpack_tuple(FunctT f, const std::tuple<Args...>& params) -> decltype(f(std::declval<Args>()...));

#include "dali/utils/unpack_tuple-impl.h"

#endif
