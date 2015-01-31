#include "utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <unordered_map>

using std::vector;
using std::string;

std::ostream &operator <<(std::ostream &os, const vector<string> &v) {
   if (v.size() == 0) return os << "[]";
   os << "[\"";
   std::copy(v.begin(), v.end() - 1, std::ostream_iterator<string>(os, "\", \""));
   return os << v.back() << "\"]";
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
	if (v.size() == 0) return os << "[]";
	os << "[";
	for (auto& f : v) 
		os << std::fixed
			          << std::setw( 7 ) // keep 7 digits
			          << std::setprecision( 3 ) // use 3 decimals
			          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
			          << f << " ";
	return os << "]";
}

template std::ostream& operator<< <double>(std::ostream& strm, const vector<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const vector<float>& a);
template std::ostream& operator<< <uint>(std::ostream& strm, const vector<uint>& a);
template std::ostream& operator<< <int>(std::ostream& strm, const vector<int>& a);

namespace utils {

	void Vocab::construct_word2index() {
		uint i = 0;
		for (auto& s : index2word) word2index[s] = i++;
	}
	void Vocab::add_unknown_word() {
		index2word.emplace_back(unknown_word_symbol);
		word2index[unknown_word_symbol] = index2word.size() - 1;
		unknown_word = index2word.size() - 1;
	}
	Vocab::Vocab() : unknown_word(-1) {};
	Vocab::Vocab(vector<string>& _index2word) : index2word(_index2word), unknown_word(-1) {
		construct_word2index();
	}
	Vocab::Vocab(vector<string>& _index2word, bool unknown_word) : index2word(_index2word), unknown_word(-1) {
		construct_word2index();
		if (unknown_word) add_unknown_word();
	}

	bool is_gzip(const std::string& filename) {
		const unsigned char gzip_code = 0x1f;
		const unsigned char gzip_code2 = 0x8b;
		unsigned char ch;
		std::ifstream file;
		file.open(filename);
		if (!file) return false;
		file.read(reinterpret_cast<char*>(&ch), 1);
		if (ch != gzip_code)
			return false;
		if (!file) return false;
		file.read(reinterpret_cast<char*>(&ch), 1);
		if (ch != gzip_code2)
			return false;
		return true;
	}

	template <typename T>
	vector<size_t> argsort(const vector<T> &v) {
		// initialize original index locations
		vector<size_t> idx(v.size());
		for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

		// sort indexes based on comparing values in v
		sort(idx.begin(), idx.end(),
		   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

		return idx;
	}
	template vector<size_t> argsort(const vector<size_t>&v);
	template vector<size_t> argsort(const vector<float>&v);
	template vector<size_t> argsort(const vector<double>&v);
	template vector<size_t> argsort(const vector<int>&v);
	template vector<size_t> argsort(const vector<uint>&v);
	template vector<size_t> argsort(const vector<std::string>&v);

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
	// template struct clip_operator<float>;

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
	// template struct clip_operator<double>;
}

