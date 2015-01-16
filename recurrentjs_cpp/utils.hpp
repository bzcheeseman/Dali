#ifndef RECURRENT_MAT_UTILS_H
#define RECURRENT_MAT_UTILS_H

#include <iostream>
#include <iomanip>
#include <random>
// #include <time>
namespace utils {
	template<typename T>
	struct sigmoid_operator {
		double operator() (T x) const { return 1.0 / (1.0 + exp(-x)); }
	};
	template<typename T>
	struct tanh_operator {
		double operator() (T x) const { return std::tanh(x); }
	};
	template<typename T>
	struct relu_operator {
		double operator() (T x) const { return std::max(x, 0.0); }
	};
	template<typename T>
	struct sign_operator {
		T operator() (T x) const { return x > 0.0 ? 1.0 : 0.0; }
	};
	template<typename T>
	struct dtanh_operator {
		T operator() (T x) const { return 1.0 - x*x; }
	};
	template<typename T>
	struct exp_operator {
		T operator() (T x) const { return std::exp(x); }
	};
	template<typename T>
	struct squared_operator {
		T operator() (T x) const { return x * x; }
	};
	template<typename T>
	struct clip_operator {
		T min;
		T max;
		clip_operator(T _min, T _max) : min(_min), max(_max) {};
		T operator() (T x) const { return (x < min) ? min : (x > max ? max : x); }
	};
	template <class T> inline void hash_combine(std::size_t & seed, const T & v) {
	  std::hash<T> hasher;
	  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	std::size_t get_random_id() {
		std::size_t seed = 0;
		std::default_random_engine generator;
		std::random_device rd;
		std::uniform_int_distribution<long> randint(0, std::numeric_limits<long>::max());
		generator.seed(rd());
		hash_combine(seed, randint(generator));
		hash_combine(seed, std::time(NULL));
		return seed;
	}
	template<typename T> void print_matrix(int n, int d, T * ptr) {
		T * end = ptr + n * d;
		int i = 0;
		std::cout << "["; // print starting bracket
		while (ptr != end) {
			std::cout << std::fixed
					  << std::setw( 7 ) // keep 7 digits
					  << std::setprecision( 3 ) // use 3 decimals
					  << std::setfill( ' ' ) // pad values with blanks
					  << *ptr;
			i++;
			if (i == n * d) {
				std::cout << "]" << std::endl; // end with bracket
			} else {
				std::cout << " "; // or add a space between numbers
			}
			if (i % d == 0 && i < n * d)
				std::cout << "\n "; // skip a line between rows
			ptr++;
		}
	}
	namespace ops {
		static const uint add     = 0;
		static const uint eltmul  = 1;
		static const uint sigmoid = 2;
		static const uint tanh    = 3;
		static const uint mul     = 4;
		static const uint relu    = 5;
		static const uint row_pluck = 6;
	}
}
#endif