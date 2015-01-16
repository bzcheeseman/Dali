#include <iostream>
#include <iomanip>
#include <random>

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
			if (i % d == 0 && i < n * d) {
				std::cout << "\n "; // skip a line between rows
			}
			ptr++;
		}
	}
	
	template<typename T> void fill_random(int n, T std, T * ptr) {
		std::default_random_engine generator;
		std::random_device rd;
		generator.seed(rd());
		std::normal_distribution<T> distribution(0.0,std);
		for (T * end = ptr + n; ptr != end;ptr++) *ptr = distribution(generator);
	}

	template<typename T> T* element_sum(int n, T* ptr1, T* ptr2) {
		T* out_array = (T * ) malloc(n * sizeof(T));
		T* iter = out_array;
		for (T* end = ptr1 + n; ptr1 != end; ptr1++, ptr2++, iter++) *iter = (*ptr1) + (*ptr2);
		return out_array;
	}
	template<typename T> void element_sum(int n, T* ptr1, T* ptr2, T* out_array) {
		T* iter = out_array;
		for (T* end = ptr1 + n; ptr1 != end; ptr1++, ptr2++, iter++) *iter = (*ptr1) + (*ptr2);
		return out_array;
	}
	template<typename T> T* element_mult(int n, T* ptr1, T* ptr2, T* out_array) {
		T* iter = out_array;
		for (T* end = ptr1 + n; ptr1 != end; ptr1++, ptr2++, iter++) *iter = (*ptr1) * (*ptr2);
		return out_array;
	}
	template<typename T> void element_mult_add(int n, T* ptr1, T* ptr2, T* out_array) {
		T* iter = out_array;
		for (T* end = ptr1 + n; ptr1 != end; ptr1++, ptr2++, iter++) *iter += (*ptr1) * (*ptr2);
	}
	template<typename T> void add_inplace(int n, T* ptr1, T* ptr2) {
		// add ptr1 to ptr2
		T* iter = ptr2;
		for (T* end = ptr1 + n; ptr1 != end; ptr1++, iter++) *iter += (*ptr1);
	}
	template<typename T> T* element_mult(int n, T* ptr1, T* ptr2) {
		T* out_array = (T * ) malloc(n * sizeof(T));
		T* iter = out_array;
		for (T* end = ptr1 + n; ptr1 != end; ptr1++, ptr2++, iter++) *iter = (*ptr1) * (*ptr2);
		return out_array;
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