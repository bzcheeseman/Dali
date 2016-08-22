#include "functor.h"

namespace functor {
    const std::string Functor::name           = "unnamed_functor";
    const std::string isnotanumber<int>::name = "isnotanumber";
    const std::string isinfinity<int>::name   = "isinfinity";
}  // namespace functor
