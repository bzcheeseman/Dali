#include <iostream>
#include <iomanip>
#include <random>

namespace utils {
	template<typename T>
	struct sigmoid_operator {
		T operator() (T x) const { return 1.0 / (1.0 + exp(-x)); }
	};
	template<typename T>
	struct tanh_operator {
		T operator() (T x) const { return std::tanh(x); }
	};
	template<typename T>
	struct relu_operator {
		T operator() (T x) const { return std::max(x, (T) 0.0); }
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
	namespace ops {
		static const uint add              = 0;
		static const uint eltmul           = 1;
		static const uint sigmoid          = 2;
		static const uint tanh             = 3;
		static const uint mul              = 4;
		static const uint relu             = 5;
		static const uint row_pluck        = 6;
		static const uint add_broadcast    = 7;
		static const uint eltmul_broadcast = 8;
	}

	template struct sigmoid_operator<float>;
	template struct tanh_operator<float>;
	template struct relu_operator<float>;
	template struct sign_operator<float>;
	template struct dtanh_operator<float>;
	template struct squared_operator<float>;
	template struct clip_operator<float>;

	template struct sigmoid_operator<double>;
	template struct tanh_operator<double>;
	template struct relu_operator<double>;
	template struct sign_operator<double>;
	template struct dtanh_operator<double>;
	template struct squared_operator<double>;
	template struct clip_operator<double>;
}

