#include "utils.h"
#include <iostream>
#include <iomanip>
#include <random>



namespace utils {
	template<typename T>
	T sigmoid_operator<T>::operator () (T x) const { return 1.0 / (1.0 + exp(-x)); }

	template<typename T>
	T tanh_operator<T>::operator() (T x) const { return std::tanh(x); }

	template<typename T>
	T relu_operator<T>::operator() (T x) const { return std::max(x, (T) 0.0); }

	template<typename T>
	T sign_operator<T>::operator() (T x) const { return x > 0.0 ? 1.0 : 0.0; }

	template<typename T>
	T dtanh_operator<T>::operator() (T x) const { return 1.0 - x*x; }

	template<typename T>
	T squared_operator<T>::operator() (T x) const { return x * x; }

	template<typename T>
	clip_operator<T>::clip_operator(T _min, T _max) : min(_min), max(_max) {};

	template<typename T>
	T clip_operator<T>::operator() (T x) const { return (x < min) ? min : (x > max ? max : x); }

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

	template<typename T>
	void assign_cli_argument(char * source, T& target, T default_val, std::string variable_name ) {
		using std::cerr;
		using std::istringstream;
		// Takes an input, a default value, and tries to extract from a character sequence
		// an assignment. If it fails it notifies the user and switches back to the default.
		// Default is copied so a copy is an original is always available
		// for fallback (even if target and default originated from the same place).
		istringstream ss(source);
		if (!(ss >> target)) {
		    cerr << "Invalid " << variable_name << " => \""<< source << "\"\n"
		         << "Using default (" << default_val << ") instead\n";
		    target = default_val;
		}
	}

	template<typename T>
	void assign_cli_argument(char * source, T& target, std::string variable_name ) {
		using std::cerr;
		using std::istringstream;
		T default_val = target;
		// Takes an input, a default value, and tries to extract from a character sequence
		// an assignment. If it fails it notifies the user and switches back to the default.
		// Default is copied so a copy is an original is always available
		// for fallback (even if target and default originated from the same place).
		istringstream ss(source);
		if (!(ss >> target)) {
		    cerr << "Invalid " << variable_name << " => \""<< source << "\"\n"
		         << "Using default (" << default_val << ") instead\n";
		    target = default_val;
		}
	}

	template struct sigmoid_operator<float>;
	template struct tanh_operator<float>;
	template struct relu_operator<float>;
	template struct sign_operator<float>;
	template struct dtanh_operator<float>;
	template struct squared_operator<float>;
	template struct clip_operator<float>;

	template void assign_cli_argument<int>(char*,int&,int,std::string);
	template void assign_cli_argument<float>(char*,float&,float,std::string);
	template void assign_cli_argument<double>(char*,double&,double,std::string);
	template void assign_cli_argument<long>(char*,long&,long,std::string);
	template void assign_cli_argument<uint>(char*,uint&,uint,std::string);
	template void assign_cli_argument<std::string>(char*,std::string&, std::string, std::string);

	template void assign_cli_argument<int>(char*,int&,std::string);
	template void assign_cli_argument<float>(char*,float&,std::string);
	template void assign_cli_argument<double>(char*,double&,std::string);
	template void assign_cli_argument<long>(char*,long&,std::string);
	template void assign_cli_argument<uint>(char*,uint&,std::string);
	template void assign_cli_argument<std::string>(char*,std::string&,std::string);

	template struct sigmoid_operator<double>;
	template struct tanh_operator<double>;
	template struct relu_operator<double>;
	template struct sign_operator<double>;
	template struct dtanh_operator<double>;
	template struct squared_operator<double>;
	template struct clip_operator<double>;
}

