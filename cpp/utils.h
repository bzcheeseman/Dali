#ifndef RECURRENT_MAT_UTILS_H
#define RECURRENT_MAT_UTILS_H

#include <iostream>
#include <iomanip>
#include <random>

namespace utils {
	template<typename T>
	struct sigmoid_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct tanh_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct relu_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct sign_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct dtanh_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct squared_operator {
		T operator() (T) const;
	};
	template<typename T>
	struct clip_operator {
		T min;
		T max;
		clip_operator(T, T);
		T operator() (T) const;
	};
	template <class T> inline void hash_combine(std::size_t &, const T &);
	std::size_t get_random_id();
	namespace ops {
		static const uint add                   = 0;
		static const uint eltmul                = 1;
		static const uint sigmoid               = 2;
		static const uint tanh                  = 3;
		static const uint mul                   = 4;
		static const uint relu                  = 5;
		static const uint row_pluck             = 6;
		static const uint add_broadcast         = 7;
		static const uint eltmul_broadcast      = 8;
		static const uint mul_with_bias         = 9;
		static const uint mul_add_mul_with_bias = 10;
		static const uint mul_add_broadcast_mul_with_bias = 11;
	}
}
#endif