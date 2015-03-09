#include "Mat.h"

using namespace Eigen;
using std::vector;
using std::string;
using std::stringstream;

DEFINE_bool(eigen_parallel, true, "Use Eigen's InitParallel Mode ?");

template<typename T>
std::atomic<int> Mat<T>::next_matrix(0);

template<typename T>
Mat<T>::Mat (dim_t _n, dim_t _d) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d),  dims({_n, _d}), random_id(next_matrix++) {
    _w = eigen_mat::Zero(dims[0], dims[1]);
    _dw = eigen_mat::Zero(dims[0], dims[1]);
    new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
    new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}
template<typename T>
Mat<T>::Mat (dim_t _n, dim_t _d, bool empty) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d), dims({_n, _d}), random_id(next_matrix++) {
    _w  = empty ? eigen_mat(dims[0], dims[1]) : eigen_mat::Zero(dims[0], dims[1]);
    _dw = eigen_mat::Zero(dims[0], dims[1]);
    new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
    new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}

template<typename T>
Mat<T>::Mat (string fname) : sparse_row_keys(nullptr), sparse(false), w(NULL, 0, 0), dw(NULL, 0, 0), random_id(next_matrix++) {
    auto arr = cnpy::npy_load(fname);
    dims = {arr.shape[0], arr.shape.size() > 1 ? arr.shape[1] : 1};
    _w  = eigen_mat(dims[0], dims[1]);
    _dw = eigen_mat::Zero(dims[0], dims[1]);

    if (arr.word_size == sizeof(double)) {
        double* loaded_data_double = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, dims[0], dims[1]);
            _w = wrapped_mat_double_ft.cast<T>();
        } else {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, dims[0], dims[1]);
            _w = wrapped_mat_double.cast<T>();
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data_float = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, dims[0], dims[1]);
            _w = wrapped_mat_float_ft.cast<T>();
        } else {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, dims[0], dims[1]);
            _w = wrapped_mat_float.cast<T>();
        }
    } else {
        stringstream error_msg;
        error_msg << "Could not load numpy matrix : \""
           << fname << "\". File dtype (" << arr.word_size << ") not recognized as float or double.";
        throw std::invalid_argument(error_msg.str());
    }
    arr.destruct();
    new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
    new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}

template<typename T>
Mat<T>::Mat (dim_t _n, dim_t _d, T std) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d),
dims({_n, _d}), random_id(next_matrix++) {
        std::default_random_engine generator;
        std::normal_distribution<T> distribution(0.0, std);
        std::random_device rd;
        generator.seed(rd());
        auto randn = [&] (int) {return distribution(generator);};
        _w = eigen_mat::NullaryExpr(dims[0], dims[1], randn);
        _dw = eigen_mat::Zero(dims[0], dims[1]);
        new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
        new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}

template<typename T>
Mat<T>::Mat (dim_t _n, dim_t _d, T lower, T upper) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d), dims({_n, _d}), random_id(next_matrix++) {
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(lower, upper);
        std::random_device rd;
        generator.seed(rd());
        auto randn = [&] (int) {return distribution(generator);};
        _w = eigen_mat::NullaryExpr(dims[0], dims[1], randn);
        _dw = eigen_mat::Zero(dims[0], dims[1]);
        //w = eigen_mat_view(_w.data(), dims[0], dims[1]);
        //dw = eigen_mat_view(_dw.data(), dims[0], dims[1]);
        new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
        new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}

template<typename T>
Mat<T>::Mat (const Mat<T>& m, bool copy_w, bool copy_dw) : sparse_row_keys(nullptr), sparse(m.sparse), name(m.name), w(NULL, m.dims[0], m.dims[1]), dw(NULL, m.dims[0], m.dims[1]), dims(m.dims), random_id(copy_w ? next_matrix++ : m.random_id) {
    if (copy_w) {
        _w = m.w;
        new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
    } else {
        new (&w) eigen_mat_view(m.w.data(), dims[0], dims[1]);
    }

    if (copy_dw) {
        _dw = m.dw;
        new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
    } else {
        new (&dw) eigen_mat_view(m.dw.data(), dims[0], dims[1]);
    }
}

template<typename T>
Mat<T> Mat<T>::shallow_copy(const Mat<T>& m) {
        return Mat(m, false, true);
}

template<typename T>
void Mat<T>::set_name(string& _name) {
        name = std::make_shared<string>(_name);
}
template<typename T>
void Mat<T>::set_name(char * _name) {
        name = std::make_shared<string>(_name);
}
template<typename T>
void Mat<T>::set_name(const char * _name) {
        name = std::make_shared<string>(_name);
}

template<typename T>
void Mat<T>::print() const {

    for (int i = 0; i < dims[0] ; ++i) {
            std::cout << (i == 0 ? "[" : " ");
            for (int j = 0; j < dims[1]; ++j) {
                    std::cout << std::fixed
                              << std::setw( 7 ) // keep 7 digits
                              << std::setprecision( 3 ) // use 3 decimals
                              << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                              << this->w(i,j) << " ";
            }
            std::cout << (i == dims[0]-1 ? "]" : "\n");
    }
    std::cout << std::endl;
}

template<typename T>
void Mat<T>::grad() {
    if (dims[0] != 1 || dims[1] != 1) {
        std::cout << *this << std::endl;
        throw std::invalid_argument("Grad only works on a \"scalar\" matrix, a 1x1 matrix. Call G.sum or G.mean before using grad.");
    }
    dw(0) += 1;
}

template<typename T>
void Mat<T>::npy_save (string fname, string mode) {
    cnpy::npy_save(fname, w.data(), dims.data(), dims.size(), mode);
}

template<typename T>
unsigned int Mat<T>::number_of_elements() const {
    unsigned int dim = 1;
    for (auto & n : dims)
        dim *= n;
    return dim;
}

template<typename T>
void Mat<T>::npy_save (FILE * fp) {
    std::vector<char> header = cnpy::create_npy_header(w.data(),dims.data(),dims.size());
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fwrite(w.data(),sizeof(T), number_of_elements(), fp);
}

template<typename T>
void Mat<T>::npy_load(cnpy::NpyArray& arr) {
    dims = {arr.shape[0], arr.shape.size() > 1 ? arr.shape[1] : 1};

    if (arr.word_size == sizeof(double)) {
        double* loaded_data_double = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, dims[0], dims[1]);
            w = wrapped_mat_double_ft.cast<T>();
        } else {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, dims[0], dims[1]);
            w = wrapped_mat_double.cast<T>();
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data_float = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, dims[0], dims[1]);
            w = wrapped_mat_float_ft.cast<T>();
        } else {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, dims[0], dims[1]);
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
Mat<T> Mat<T>::RandMat(dim_t n, dim_t d, T std) {
    // is in fact using C++ 11 's rvalue, move operator,
    // so no copy is made.
    return Mat(n, d, std);
}

template<typename T>
Mat<T> Mat<T>::Empty(dim_t n, dim_t d) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Mat(n, d, true);
}

template<typename T>
std::ostream& operator<<(std::ostream& strm, const Mat<T>& a) {
    if (a.name != 0) {
        return strm << "<#Mat name=\"" << *a.name<< "\" n=" << a.dims[0] << ", d=" << a.dims[1] << ">";
    } else {
        return strm << "<#Mat n=" << a.dims[0] << ", d=" << a.dims[1] << ">";
    }
}

template std::ostream& operator<< <double>(std::ostream& strm, const Mat<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Mat<float>& a);

template <typename T>
std::size_t std::hash<Mat<T>>::operator()(const Mat<T>& k) const {
    return k.random_id;
}

template std::size_t std::hash<Mat<float>>::operator()(const Mat<float>& k)   const;
template std::size_t std::hash<Mat<double>>::operator()(const Mat<double>& k) const;

template <typename T>
bool operator!=(const Mat<T>& A, const Mat<T>& B) {
    return A.random_id != B.random_id;
}

template bool operator!=(const Mat<float>&, const Mat<float>&);
template bool operator!=(const Mat<double>&, const Mat<double>&);

template <typename T>
bool operator==(const Mat<T>& A, const Mat<T>& B) {
    return A.random_id == B.random_id;
}

template bool operator==<float>(const Mat<float>&, const Mat<float>&);
template bool operator==<double>(const Mat<double>&, const Mat<double>&);

template<typename T>
int argmax(std::shared_ptr<Mat<T>> A) {
    int i = 0;
    T current_max = -std::numeric_limits<T>::infinity();
    auto ptr = A->w.data();
    for (int j = 0; j < A->number_of_elements(); j++) {
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
