#include "Mat.h"
#include "utils.h"

using namespace Eigen;
using std::vector;
using std::string;
using std::stringstream;

template<typename T>
Mat<T>::Mat (int _n, int _d) : sparse_row_keys(NULL), sparse(false), name(NULL), w(NULL, _n, _d), dw(NULL, _n, _d),  n(_n), d(_d), random_id(utils::get_random_id()) {
    _w = eigen_mat::Zero(n,d);
    _dw = eigen_mat::Zero(n,d);
	new (&w) eigen_mat_view(_w.data(), n, d);
	new (&dw) eigen_mat_view(_dw.data(), n, d);

}
template<typename T>
Mat<T>::Mat (int _n, int _d, bool empty) : sparse_row_keys(NULL), sparse(false), name(NULL), w(NULL, _n, _d), dw(NULL, _n, _d), n(_n), d(_d), random_id(utils::get_random_id()) {
	_w  = empty ? eigen_mat(n,d) : eigen_mat::Zero(n,d);
	_dw = eigen_mat::Zero(n,d);
	new (&w) eigen_mat_view(_w.data(), n, d);
	new (&dw) eigen_mat_view(_dw.data(), n, d);
}

template<typename T>
Mat<T>::Mat (string fname) : sparse_row_keys(NULL), sparse(false), w(NULL, 0, 0), dw(NULL, 0, 0), random_id(utils::get_random_id()) {
	auto arr = cnpy::npy_load(fname);
	n = arr.shape[0];
	d = arr.shape.size() > 1 ? arr.shape[1] : 1;

	_w  = eigen_mat(n,d);
    _dw = eigen_mat::Zero(n,d);

	if (arr.word_size == sizeof(double)) {
		double* loaded_data_double = reinterpret_cast<double*>(arr.data);
		if (arr.fortran_order) {
			Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, n, d);
			_w = wrapped_mat_double_ft.cast<T>();
		} else {
			Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, n, d);
			_w = wrapped_mat_double.cast<T>();
		}
	} else if (arr.word_size == sizeof(float)) {
		float* loaded_data_float = reinterpret_cast<float*>(arr.data);
		if (arr.fortran_order) {
			Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, n, d);
			_w = wrapped_mat_float_ft.cast<T>();
		} else {
			Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, n, d);
			_w = wrapped_mat_float.cast<T>();
		}
	} else {
		stringstream error_msg;
		error_msg << "Could not load numpy matrix : \""
		   << fname << "\". File dtype (" << arr.word_size << ") not recognized as float or double.";
		throw std::invalid_argument(error_msg.str());
	}
	arr.destruct();
	new (&w) eigen_mat_view(_w.data(), n, d);
	new (&dw) eigen_mat_view(_dw.data(), n, d);

}

/*
Mat<T>::Mat<T>
--------------

Matrix constructor using a zero mean
normal distribution with a user provided 
standard deviation.

Inputs
------

int _n : number of rows
int _d : number of columns
 T std : standard deviation for normal distribution

Outputs
-------

Mat<T> out : the matrix filled with random numbers from ~ N(0, std^2)

See `Mat<T>::Mat(int, int, T, T)` for uniform distribution (below).

*/
template<typename T>
Mat<T>::Mat (int _n, int _d, T std) : sparse_row_keys(NULL), sparse(false), name(NULL), w(NULL, _n, _d), dw(NULL, _n, _d), n(_n), d(_d), random_id(utils::get_random_id()) {
	std::default_random_engine generator;
	std::normal_distribution<T> distribution(0.0, std);
	std::random_device rd;
	generator.seed(rd());
	auto randn = [&] (int) {return distribution(generator);};
	_w = eigen_mat::NullaryExpr(n,d, randn);
	_dw = eigen_mat::Zero(n,d);
	new (&w) eigen_mat_view(_w.data(), n, d);
	new (&dw) eigen_mat_view(_dw.data(), n, d);
}

/*
Mat<T>::Mat<T>
--------------

Matrix constructor using a uniform
distribution with user defined min
and max support.

Inputs
------

  int _n : number of rows
  int _d : number of columns
 T lower : minimum of uniform distribution
 T upper : maximum of uniform distribution

Outputs
-------

Mat<T> out : the matrix filled with random numbers from ~U(lower, upper)

See `Mat<T>::Mat(int, int, T)` for normal distribution (above)
*/
template<typename T>
Mat<T>::Mat (int _n, int _d, T lower, T upper) : sparse_row_keys(NULL), sparse(false), name(NULL), w(NULL, _n, _d), dw(NULL, _n, _d), n(_n), d(_d), random_id(utils::get_random_id()) {
	std::default_random_engine generator;
	std::uniform_real_distribution<T> distribution(lower, upper);
	std::random_device rd;
	generator.seed(rd());
	auto randn = [&] (int) {return distribution(generator);};
	_w = eigen_mat::NullaryExpr(n,d, randn);
	_dw = eigen_mat::Zero(n,d);
	//w = eigen_mat_view(_w.data(), n, d);
	//dw = eigen_mat_view(_dw.data(), n, d);
	new (&w) eigen_mat_view(_w.data(), n, d);
	new (&dw) eigen_mat_view(_dw.data(), n, d);
}

/**
Mat<T>::Mat<T>
--------------

A copy constructor that perform shallow and deep
copies of a Mat.

Key usage is for Hogwild style training of parameters
where different computation threads share memory for
the parameters but each compute their own gradients.
The gradients are kept in separate `dw` memory buffers
but `w` buffers are shared amongst threads.

For usage see `Solver::Adadelta`, `examples/character_prediction.cpp`

Inputs
------

const Mat<T>& m : matrix to copy or point to
    bool copy_w : whether matrix parameters should be copied over, or shared
                  between matrices
   bool copy_dw : whether matrix gradient parameters should be copied over,
                  or shared (Note: it is unclear when `dw` should be shared,
                  proceed with caution).

Outputs
-------

Mat<T> out : deep or shallow copy of m

**/
template<typename T>
Mat<T>::Mat (const Mat<T>& m, bool copy_w, bool copy_dw) : sparse_row_keys(NULL), sparse(m.sparse), name(m.name), w(NULL, m.n, m.d), dw(NULL, m.n, m.d), n(m.n), d(m.d), random_id(copy_w ? utils::get_random_id() : m.random_id) {
	if (copy_w) {
		_w = m.w;
		new (&w) eigen_mat_view(_w.data(), n, d);
	} else {
		new (&w) eigen_mat_view(m.w.data(), n, d);
	}

	if (copy_dw) {
		_dw = m.dw;
		new (&dw) eigen_mat_view(_dw.data(), n, d);
	} else {
		new (&dw) eigen_mat_view(m.dw.data(), n, d);
	}
}


/**
Shallow Copy
------------

A copy constructor that perform shallow copies of a Mat.

Key usage is for Hogwild style training of parameters
where different computation threads share memory for
the parameters but each compute their own gradients.
The gradients are kept in separate `dw` memory buffers
but `w` buffers are shared amongst threads.

For usage see `Mat<T>::Mat<T>(const Mat&, bool, bool)`, `examples/character_prediction.cpp`

Inputs
------

const Mat<T>& m : matrix that will own the underlying memory
                  for `w`  

Outputs
-------

Mat<T> out : shallow copy of m

**/

template<typename T>
Mat<T> Mat<T>::shallow_copy(const Mat<T>& m) {
	return Mat(m, false, true);
}

/*
Set Name
--------

Used for giving names to matrices for debugging or convenience purposes,
but the names have no bearing on computation or identification in
lookup tables;

Inputs
------

std::string& name : name the Mat should take on

*/

template<typename T>
void Mat<T>::set_name(string& _name) {
	name = std::make_shared<string>(_name);
}

/*
Set Name
--------
See `Mat<T>::set_name` above
*/
template<typename T>
void Mat<T>::set_name(char * _name) {
	name = std::make_shared<string>(_name);
}
/*
Set Name
--------
See `Mat<T>::set_name` above
*/
template<typename T>
void Mat<T>::set_name(const char * _name) {
	name = std::make_shared<string>(_name);
}

template<typename T>
void Mat<T>::print () {
	for (int i = 0; i < n ; ++i) {
		std::cout << (i == 0 ? "[" : " ");
		for (int j = 0; j < d; ++j) {
			std::cout << std::fixed
			          << std::setw( 7 ) // keep 7 digits
			          << std::setprecision( 3 ) // use 3 decimals
			          << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
			          << this->w(i,j) << " ";
		}
		std::cout << (i == n-1 ? "]" : "\n");
	}
	std::cout << std::endl;
}

template<typename T>
void Mat<T>::npy_save (string fname, string mode) {
	const unsigned int shape[] = {(unsigned int) n,(unsigned int) d};
	cnpy::npy_save(fname, w.data(), shape, 2, mode);
}

template<typename T>
void Mat<T>::npy_save (FILE * fp) {
	const unsigned int shape[] = {(unsigned int) n,(unsigned int) d};
	std::vector<char> header = cnpy::create_npy_header(w.data(),shape,2);
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fwrite(w.data(),sizeof(T),n*d,fp);
}

template<typename T>
void Mat<T>::npy_load(cnpy::NpyArray& arr) {
	n = arr.shape[0];
	d = arr.shape.size() > 1 ? arr.shape[1] : 1;

	if (arr.word_size == sizeof(double)) {
		double* loaded_data_double = reinterpret_cast<double*>(arr.data);
		if (arr.fortran_order) {
			Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, n, d);
			w = wrapped_mat_double_ft.cast<T>();
		} else {
			Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, n, d);
			w = wrapped_mat_double.cast<T>();
		}
	} else if (arr.word_size == sizeof(float)) {
		float* loaded_data_float = reinterpret_cast<float*>(arr.data);
		if (arr.fortran_order) {
			Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, n, d);
			w = wrapped_mat_float_ft.cast<T>();
		} else {
			Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, n, d);
			w = wrapped_mat_float.cast<T>();
		}
	} else {
		throw std::invalid_argument("Could not load numpy matrix : not recognized as float or double.");
	}
}

template<typename T>
void Mat<T>::npy_load(FILE * fp) {
	auto arr = cnpy::load_the_npy_file(fp);
	npy_load(arr);
	arr.destruct();
}

template<typename T>
void Mat<T>::npy_load(string fname) {
	auto arr = cnpy::npy_load(fname);
	npy_load(arr);
	arr.destruct();
}

template<typename T>
Mat<T>::~Mat() {}

template<typename T>
Mat<T> Mat<T>::RandMat(int n, int d, T std) {
	// is in fact using C++ 11 's rvalue, move operator,
	// so no copy is made.
	return Mat(n, d, std);
}

template<typename T>
Mat<T> Mat<T>::Empty(int n, int d) {
	// use an empty matrix and modify
	// it so as to not incur the filling
	// with zeros cost.
	return Mat(n, d, true);
}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Mat<T>& a) {
	if (a.name != NULL) {
		return strm << "<#Mat name=\"" << *a.name<< "\" n=" << a.n << ", d=" << a.d << ">";
	} else {
		return strm << "<#Mat n=" << a.n << ", d=" << a.d << ">";
	}
}

template std::ostream& operator<< <double>(std::ostream& strm, const Mat<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Mat<float>& a);

template <typename T>
std::size_t std::hash<Mat<T>>::operator()(const Mat<T>& k) const {
	return k.random_id;
}

template <typename T>
static bool operator!=(const Mat<T>& A, const Mat<T>& B) {
    return A.random_id != B.random_id;
}

template <typename T>
static bool operator==(const Mat<T>& A, const Mat<T>& B) {
    return A.random_id == B.random_id;
}

#define PARAM_KEY_FOR_LOOKUP_TABLE *param

template<typename T>
Solver::SGD<T>::SGD (T _clipval) :
        clipval(_clipval) {};

template<typename T>
void Solver::SGD<T>::step (vector<typename Solver::SGD<T>::shared_mat>& parameters,
	T step_size,
	T regc
	) {
	for (auto& param : parameters) {
		if (param->sparse) {
			for (auto& i : *(param->sparse_row_keys)) {
				if (regc > 0) {
					param->w.row(i) -= (step_size * param->dw.row(i).array().min(clipval).max(-clipval)).matrix() - (regc * param->w.row(i));
				} else {
					param->w.row(i) -= (step_size * param->dw.row(i).array().min(clipval).max(-clipval)).matrix();
				}
				// reset gradient
				param->dw.row(i).fill(0);
			}
		} else {
			if (regc > 0) {
				param->w -= (step_size * param->dw.array().min(clipval).max(-clipval)).matrix() - (regc * param->w);
			} else {
				param->w -= (step_size * param->dw.array().min(clipval).max(-clipval)).matrix();
			}
			// reset gradient
			param->dw.fill(0);
		}
	}
}

template<typename T>
Solver::AdaDelta<T>::AdaDelta (
	T _rho,
	T _smooth_eps,
	T _clipval) :
		smooth_eps(_smooth_eps),
		rho(_rho),
        clipval(_clipval) {};

template<typename T>
Solver::AdaDelta<T>::AdaDelta (
	vector<typename Solver::AdaDelta<T>::shared_mat>& parameters,
	T _rho,
	T _smooth_eps,
	T _clipval) :
		smooth_eps(_smooth_eps),
		rho(_rho),
        clipval(_clipval) {
    create_gradient_caches(parameters);
};

template<typename T>
void Solver::AdaDelta<T>::create_gradient_caches(
	vector<typename Solver::AdaDelta<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
			auto new_cache = this->gsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);

			new_cache = this->xsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
	}
}

template<typename T>
void Solver::AdaDelta<T>::step (vector<typename Solver::AdaDelta<T>::shared_mat>& parameters, T regc) {
	for (auto& param : parameters) {
		auto& gsum = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
		auto& xsum = xsums[PARAM_KEY_FOR_LOOKUP_TABLE];
		if (param->sparse) {
			for (auto& i : *(param->sparse_row_keys)) {
				if (regc > 0) {
					param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix() + (regc * param->w.row(i));
				} else {
					// param->dw = param->dw.array().min(clipval).max(-clipval).matrix();
				}
				// update gradient cache using decay rule:
				gsum.row(i) = (gsum.row(i) * rho) + ((1.0 - rho) * (param->dw.row(i).array().square()).matrix());

				auto dparam = -(((xsum.row(i).array() + smooth_eps) / (gsum.row(i).array() + smooth_eps)).sqrt() * param->dw.row(i).array()).matrix();

				xsum.row(i) = (xsum.row(i) * rho) + ((1.0 - rho) * (dparam.array().square())).matrix();
				// update gradient using AdaDelta rule
				param->w.row(i) += dparam;
				// reset gradient
				param->dw.row(i).fill(0);
			}
		} else {
			if (regc > 0) {
				param->dw = param->dw.array().min(clipval).max(-clipval).matrix() + (regc * param->w);
			} else {
				// param->dw = param->dw.array().min(clipval).max(-clipval).matrix();
			}
			// update gradient cache using decay rule:
			gsum = (gsum * rho) + ((1.0 - rho) * (param->dw.array().square()).matrix());

			auto dparam = -(((xsum.array() + smooth_eps) / (gsum.array() + smooth_eps)).sqrt() * param->dw.array()).matrix();

			xsum = (xsum * rho) + ((1.0 - rho) * (dparam.array().square())).matrix();
			// update gradient using AdaDelta rule
			param->w += dparam;
			// reset gradient
			param->dw.fill(0);
		}
	}
}

template<typename T>
int argmax(std::shared_ptr<Mat<T>> A) {
	int i = 0;
	T current_max = -std::numeric_limits<T>::infinity();
	auto ptr = A->w.data();
	for (int j = 0; j < A->n * A->d; j++) {
		if (*ptr > current_max) {
			current_max = *ptr;
			i = j;
		}
		ptr++;
	}
	return i;
}

template<typename T>
int argmax_slice(std::shared_ptr<Mat<T>> A, int min, int max) {
	int i = 0;
	T current_max = -std::numeric_limits<T>::infinity();
	auto ptr = A->w.data();
	for (int j = min; j < max; j++) {
		if (*ptr > current_max) {
			current_max = *ptr;
			i = j;
		}
		ptr++;
	}
	return i;
}

template int argmax(std::shared_ptr<Mat<float>>);
template int argmax(std::shared_ptr<Mat<double>>);
template int argmax_slice(std::shared_ptr<Mat<float>>, int, int);
template int argmax_slice(std::shared_ptr<Mat<double>>, int, int);

template<typename T>
Solver::AdaGrad<T>::AdaGrad(T _smooth_eps, T _clipval) : smooth_eps(_smooth_eps), clipval(_clipval) {}

template<typename T>
Solver::AdaGrad<T>::AdaGrad (vector<typename Solver::AdaGrad<T>::shared_mat>& parameters,
	T _smooth_eps,
	T _clipval) : smooth_eps(_smooth_eps), clipval(_clipval) {
	create_gradient_caches(parameters);
}

template<typename T>
void Solver::AdaGrad<T>::step(
	vector<typename Solver::AdaGrad<T>::shared_mat>& parameters,
	T step_size,
	T regc
	) {
	for (auto& param : parameters) {
		auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
		if (param->sparse) {
			for (auto& i : *(param->sparse_row_keys)) {
				param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix() + (regc * param->w.row(i));
				// update gradient cache using decay rule:
				s.row(i) += param->dw.row(i).array().square().matrix();
				// clip the gradient to prevent explosions:
				// update gradient using RMSprop rule
				param->w.row(i) -= step_size * (param->dw.row(i).array() / (s.row(i).array() + smooth_eps).sqrt() ).matrix();
				// reset gradient
				param->dw.row(i).fill(0);
			}
		} else {
			param->dw = param->dw.array().min(clipval).max(-clipval).matrix() + (regc * param->w);
			// update gradient cache using decay rule:
			s += param->dw.array().square().matrix();
			// clip the gradient to prevent explosions:
			// update gradient using RMSprop rule
			param->w -= step_size * (param->dw.array() / (s.array() + smooth_eps).sqrt() ).matrix();
			// reset gradient
			param->dw.fill(0);
		}
	}
}

template<typename T>
void Solver::AdaGrad<T>::reset_caches(
	vector<typename Solver::AdaGrad<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
		s.fill(0);
	}
}

template<typename T>
void Solver::AdaGrad<T>::create_gradient_caches(
	vector<typename Solver::AdaGrad<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
			auto new_cache = this->gsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
	}
}

template<typename T>
Solver::RMSProp<T>::RMSProp (
            T _decay_rate,
            T _smooth_eps,
            T _clipval) :
        decay_rate(_decay_rate),
        smooth_eps(_smooth_eps),
        clipval(_clipval) {};

template<typename T>
Solver::RMSProp<T>::RMSProp (
			vector<typename Solver::RMSProp<T>::shared_mat>& parameters,
            T _decay_rate,
            T _smooth_eps,
            T _clipval) :
        decay_rate(_decay_rate),
        smooth_eps(_smooth_eps),
        clipval(_clipval)
        {
    create_gradient_caches(parameters);
};

template<typename T>
void Solver::RMSProp<T>::create_gradient_caches(
	vector<typename Solver::RMSProp<T>::shared_mat>& parameters) {
	for (auto& param : parameters) {
		// this operation should be run once unless
		// we expect the parameters of the model
		// to change online (probably not the case)
		if (!(gsums.count(PARAM_KEY_FOR_LOOKUP_TABLE) > 0)) {
			auto new_cache = this->gsums.emplace(
				std::piecewise_construct,
	              std::forward_as_tuple(PARAM_KEY_FOR_LOOKUP_TABLE),
	              std::forward_as_tuple(param->n, param->d));
			// initialize values for step cache to zero:
			new_cache.first->second.fill(0);
		}
	}
}

template<typename T>
void Solver::RMSProp<T>::step(
	vector<typename Solver::RMSProp<T>::shared_mat>& parameters,
	T step_size,
	T regc
	) {
	for (auto& param : parameters) {
		auto& s = gsums[PARAM_KEY_FOR_LOOKUP_TABLE];
		if (param->sparse) {

			for (auto& i : *(param->sparse_row_keys)) {
				s.row(i) = s.row(i) * decay_rate + (1.0 - decay_rate) * param->dw.row(i).array().square().matrix();
				// clip the gradient to prevent explosions:
				param->dw.row(i) = param->dw.row(i).array().min(clipval).max(-clipval).matrix();
				// update gradient using RMSprop rule
				param->w.row(i) -= step_size * (param->dw.row(i).array() / (s.row(i).array() + smooth_eps).sqrt() ).matrix()  - (regc * param->w.row(i));
				// reset gradient
				param->dw.row(i).fill(0);
			}

		} else {
			s = s * decay_rate + (1.0 - decay_rate) * param->dw.array().square().matrix();
			// clip the gradient to prevent explosions:
			param->dw = param->dw.array().min(clipval).max(-clipval).matrix();
			// update gradient using RMSprop rule
			param->w -= step_size * (param->dw.array() / (s.array() + smooth_eps).sqrt() ).matrix()  - (regc * param->w);
			// reset gradient
			param->dw.fill(0);
		}
	}

}

template<typename T>
void utils::save_matrices(vector<std::shared_ptr<Mat<T>>>& parameters, string dirname) {
	utils::ensure_directory(dirname);
	const char * c_dirname = dirname.c_str();
	utils::makedirs(c_dirname);
	int i = 0;
	for (auto& param : parameters) {
		stringstream param_location;
		param_location << dirname << "/param_" << i << ".npy";
		param->npy_save(param_location.str());
		i++;
	}
}

template<typename T>
void utils::load_matrices(vector< std::shared_ptr<Mat<T>> >& parameters, string dirname) {
	utils::ensure_directory(dirname);
	int i = 0;
	for (auto& param : parameters) {
		stringstream param_location;
		param_location << dirname << "/param_" << i << ".npy";
		param->npy_load(param_location.str());
		i++;
	}
}

template void utils::save_matrices(vector< std::shared_ptr<Mat<float>> >&, string);
template void utils::save_matrices(vector< std::shared_ptr<Mat<double>> >&, string);
template void utils::load_matrices(vector< std::shared_ptr<Mat<float>> >&, string);
template void utils::load_matrices(vector< std::shared_ptr<Mat<double>> >&, string);

template class Mat<float>;
template class Mat<double>;

template class Solver::SGD<float>;
template class Solver::SGD<double>;

template class Solver::AdaGrad<float>;
template class Solver::AdaGrad<double>;

template class Solver::AdaDelta<float>;
template class Solver::AdaDelta<double>;

template class Solver::RMSProp<float>;
template class Solver::RMSProp<double>;
