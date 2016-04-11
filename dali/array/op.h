#include "dali/config.h"

#include "dali/array/op/binary.h"
#include "dali/array/op/elementwise.h"
#include "dali/array/op/other.h"


#if EXISTS_AND_TRUE(DALI_USE_LAZY)
    #include "dali/array/lazy_op/binary.h"
    namespace lazy {
        static bool ops_loaded = true;
    }

    template<typename T, typename T2>
    auto operator+(const T& left, const T2& right) -> decltype(lazy::add(left,right)) {
        return lazy::add(left,right);
    }

    template<typename T, typename T2>
    auto operator-(const T& left, const T2& right) -> decltype(lazy::sub(left,right)) {
        return lazy::sub(left,right);
    }

    template<typename T, typename T2>
    auto operator*(const T& left, const T2& right) -> decltype(lazy::eltmul(left,right)) {
        return lazy::eltmul(left,right);
    }

    template<typename T, typename T2>
    auto operator/(const T& left, const T2& right) -> decltype(lazy::eltdiv(left,right)) {
        return lazy::eltdiv(left,right);
    }
#else
    namespace lazy {
        static bool ops_loaded = false;
    }

    AssignableArray operator+(const Array& left, const Array& right);

    AssignableArray operator-(const Array& left, const Array& right);

    AssignableArray operator*(const Array& left, const Array& right);

    AssignableArray operator/(const Array& left, const Array& right);
#endif
