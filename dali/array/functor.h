#ifndef DALI_ARRAY_TENSOR_FUNCTIONS_H
#define DALI_ARRAY_TENSOR_FUNCTIONS_H

#include "dali/config.h"
#include "dali/macros.h"
#include <math.h>
#include <cmath>
#include <string>

#ifdef DALI_USE_CUDA
    #define TANH_F tanhf
    #define LOG_F  logf
    #define EXP_F  expf
    #define POW_F  powf
    #define ISINF_F isinf
    #define ISNAN_F isnan
#else
    #define TANH_F std::tanh
    #define LOG_F  std::log
    #define EXP_F  std::exp
    #define POW_F  pow
    #define ISINF_F std::isinf
    #define ISNAN_F std::isnan
#endif

#define EPS 1e-6

namespace functor_helper {
    template<typename ExpT> struct ExtractDType {typedef typename ExpT::exp_dtype_t value;};
    template<>              struct ExtractDType<float> {typedef float value;};
    template<>              struct ExtractDType<int> {typedef int value;};
    template<>              struct ExtractDType<double> {typedef double value;};

    template<typename T1, typename T2>
    struct BinaryExtractDType {
        typedef typename ExtractDType<T1>::value left_t;
        typedef typename ExtractDType<T2>::value right_t;
        typedef decltype(left_t(0) * right_t(0)) value;
    };

    template<typename T>
    struct UnaryExtractDType {
        typedef typename ExtractDType<T>::value value;
    };
}  // namespace functor_helper

namespace functor {

    struct Functor {
        const static std::string name;
    };

    template<typename T>
    struct near_equal : public Functor {
        const static std::string name;
        T tol;
        near_equal(T _tol) : tol(_tol) {}
        XINLINE bool operator()(const T& lhs, const T& rhs) const {
            return std::abs(lhs - rhs) < tol;
        }
    };

    template<typename R>
    struct square : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return a * a;
        }
    };

    template<typename R>
    struct cube : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return a * a * a;
        }
    };

    template<typename R>
    struct eye : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& diag, const int& y, const int& x) {
            return x == y ? diag : 0.0;
        }
    };

    template<typename R>
    struct fill : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& filler) {
            return filler;
        }
    };

    template<typename R>
    struct arange : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& step, const int& y, const int& x) {
            return a + step * x;
        }
    };

    template<typename R>
    struct add : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a + b;
        }
    };

    template<typename R>
    struct equals : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a == b ? 1 : 0;
        }
    };

    template<typename R>
    struct sub : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a - b;
        }
    };

    template<typename R>
    struct eltmul : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a * b;
        }
    };

    template<typename R>
    struct eltdiv : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a / b;
        }
    };

    template<>
    XINLINE
    int square<int>::Map(const int& a) {
        return a*a;
    }

    template<typename R>
    struct sqrt_f : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return sqrt(a);
        }
    };

    template<>
    XINLINE
    int sqrt_f<int>::Map(const int& a) {
        return (int)sqrt((double)a);
    }

    template<typename R>
    struct rsqrt : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return (1.0 / sqrt(a));
        }
    };

    template<>
    XINLINE
    int rsqrt<int>::Map(const int& a) {
        return (int)(1.0 / sqrt((double)a));
    }


    template<typename R>
    struct inv : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return ((R)1.0) / a;
        }
    };

    template<typename R>
    struct sigmoid : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return 1.0 / (1.0 + EXP_F(-a));
        }
    };

    template<typename R>
    struct identity : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return a;
        }
    };

    template<typename R>
    struct log : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return LOG_F(a);
        }
    };

    template<typename R>
    struct negative_log : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return -LOG_F(a);
        }
    };

    template<typename R>
    struct safe_entropy_log : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            R a_safe = a;
            const R lower_bound = (R)EPS;
            const R upper_bound = (R)(1.0 - EPS);
            if (a_safe > upper_bound) {
                a_safe = upper_bound;
            }
            if (a_safe < lower_bound) {
                a_safe = lower_bound;
            }
            return LOG_F(a_safe);
        }
    };

    template<typename R>
    struct exp : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return EXP_F(a);
        }
    };

    template<typename R>
    struct isnotanumber : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return ISNAN_F(a);
        }
    };

    // Every possible value of integer is a number
    // One can think of integer as floating point
    // number with only sign and mantissa bits,
    // (while nan/inf information is expressed
    // as a special value of exponent).
    template<>
    struct isnotanumber<int> : public Functor {
        const static std::string name;
        XINLINE static int Map(const int& a) {
            return 0;
        }
    };

    template<typename R>
    struct isinfinity : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return ISINF_F(a);
        }
    };

    template<>
    struct isinfinity<int> : public Functor {
        const static std::string name;
        XINLINE static int Map(const int& a) {
            return 0;
        }
    };

    template<typename R>
    struct div_grad : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a / (b * b);
        }
    };

    template<typename R>
    struct dsigmoid : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return a * (((R)1.0) - a);
        }
    };

    template<typename R>
    struct tanh : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            // if (a < -30.0f)
            //     return -1.0f;
            // if (a >  30.0f)
            //     return 1.0f;
            return TANH_F(a);
        }
    };

    // tanh^-1 (z) = 1/2 * (ln(1 + z) - ln(1 - z))
    // http://mathworld.wolfram.com/InverseHyperbolicTangent.html
    template<typename R>
    struct inverse_tanh : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return (LOG_F(1 + a) - LOG_F(1 - a)) * 0.5;
        }
    };

    template<typename R>
    struct dtanh : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return 1.0 - a * a;
        }
    };

    template<typename R>
    struct power : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return POW_F(a, b);
        }
    };

    template<typename R>
    struct abs : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            return std::abs(a);
        }
    };

    template<typename R>
    struct log_or_zero : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a) {
            if (a > 0) {
                return (R)LOG_F(a);
            } else {
                return (R)0;
            }
        }
    };

    template<typename R>
    struct sign : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x) {
            return x > 0.0 ? 1.0 : -1.0;
        }
    };

    template<typename R>
    struct threshold : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& a, const R& b) {
            return a < b ? 1.0 : 0.0;
        }
    };

    template<typename R>
    struct max_scalar : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& y) {
            return x > y ? x : y;
        }
    };

    template<typename R>
    struct min_scalar : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& y) {
            return x < y ? x : y;
        }
    };

    template<typename R>
    struct  steep_sigmoid : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& aggressiveness) {
            return 1.0 / (1.0 + EXP_F( - aggressiveness * x));
        }
    };

    template<typename R>
    struct  steep_sigmoid_backward : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& aggressiveness) {
            return aggressiveness * (x - x * x);
        }
    };

    template<typename R>
    struct relu : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x) {
            return x > 0.0 ? x : 0.0;
        }
    };

    template<typename R>
    struct clipped_relu : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& clipfactor) {
            return x > 0.0 ? ( x > clipfactor ? clipfactor : x) : 0.0;
        }
    };

    template<typename R>
    struct relu_backward : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x) {
            return x > 0.0 ? 1.0 : 0.0;
        }
    };

    template<typename R>
    struct clipped_relu_backward : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& clipfactor) {
            return x > 0.0 ? (x > clipfactor ? 0.0 : 1.0) : 0.0;
        }
    };

    template<typename R>
    struct clip : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& clipping_val) {
            if (x > clipping_val) {
                return clipping_val;
            } else if (x < -clipping_val) {
                return -clipping_val;
            } else {
                return x;
            }
        }
    };

    template<typename R>
    struct lessthanequal : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& upperbound) {
            return x <= upperbound ? 1 : 0;
        }
    };

    template<typename R>
    struct greaterthanequal : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& lowerbound) {
            return x >= lowerbound ? 1 : 0;
        }
    };

    template<typename R>
    struct binary_cross_entropy : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& t ) {
            R distance_from1 =        t  * LOG_F(x);
            R distance_from0 = (1.0 - t) * LOG_F(1. - x);
            return -(distance_from1 + distance_from0);
        }
    };

    template<typename R>
    struct binary_cross_entropy_grad : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& t ) {
            R numerator   = t - x;
            R denominator = (x * (x - 1.0));
            return numerator / denominator;
        }
    };

    template<typename R>
    struct softplus : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x) {
            if (x > 20.0) {
                return x;
            } else {
                return LOG_F((R)1.0 + EXP_F(x));
            }
        }
    };

    template<typename R>
    struct softplus_backward : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x) {
            if (x > 20.0) {
                return 1.0;
            } else {
                return EXP_F(x) / ((R)1.0 + EXP_F(x));
            }
        }
    };

    template<typename R>
    struct prelu : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& weight) {
            return x > 0 ? x : weight * x;
        }
    };

    template<typename R>
    struct prelu_backward_weights : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& grad) {
            return x > 0 ? 0 : x * grad;
        }
    };

    template<typename R>
    struct prelu_backward_inputs : public Functor {
        const static std::string name;
        XINLINE static R Map(const R& x, const R& weight) {
            return x > 0 ? 1.0 : weight;
        }
    };
} //namespace functor

#include "dali/array/functor-impl.h"

#endif
