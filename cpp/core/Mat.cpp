#include "Mat.h"

using namespace Eigen;
using std::vector;
using std::string;
using std::stringstream;

DEFINE_bool(eigen_parallel, true, "Use Eigen's InitParallel Mode ?");

template<typename R>
std::atomic<int> Mat<R>::next_matrix(0);

template<typename R>
Mat<R>::Mat (dim_t _n, dim_t _d) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d),  dims({_n, _d}), random_id(next_matrix++) {
    _w = eigen_mat::Zero(dims[0], dims[1]);
    _dw = eigen_mat::Zero(dims[0], dims[1]);
    new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
    new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}
template<typename R>
Mat<R>::Mat (dim_t _n, dim_t _d, bool empty) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d), dims({_n, _d}), random_id(next_matrix++) {
    _w  = empty ? eigen_mat(dims[0], dims[1]) : eigen_mat::Zero(dims[0], dims[1]);
    _dw = eigen_mat::Zero(dims[0], dims[1]);
    new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
    new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}

template<typename R>
Mat<R>::Mat (string fname) : sparse_row_keys(nullptr), sparse(false), w(NULL, 0, 0), dw(NULL, 0, 0), random_id(next_matrix++) {
    auto arr = cnpy::npy_load(fname);
    dims = {arr.shape[0], arr.shape.size() > 1 ? arr.shape[1] : 1};
    _w  = eigen_mat(dims[0], dims[1]);
    _dw = eigen_mat::Zero(dims[0], dims[1]);

    if (arr.word_size == sizeof(double)) {
        double* loaded_data_double = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, dims[0], dims[1]);
            _w = wrapped_mat_double_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, dims[0], dims[1]);
            _w = wrapped_mat_double.cast<R>();
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data_float = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, dims[0], dims[1]);
            _w = wrapped_mat_float_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, dims[0], dims[1]);
            _w = wrapped_mat_float.cast<R>();
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

template<typename R>
Mat<R>::Mat (dim_t _n, dim_t _d, R std) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d),
dims({_n, _d}), random_id(next_matrix++) {
        std::default_random_engine generator;
        std::normal_distribution<R> distribution(0.0, std);
        std::random_device rd;
        generator.seed(rd());
        auto randn = [&] (int) {return distribution(generator);};
        _w = eigen_mat::NullaryExpr(dims[0], dims[1], randn);
        _dw = eigen_mat::Zero(dims[0], dims[1]);
        new (&w) eigen_mat_view(_w.data(), dims[0], dims[1]);
        new (&dw) eigen_mat_view(_dw.data(), dims[0], dims[1]);
}

template<typename R>
Mat<R>::Mat (dim_t _n, dim_t _d, R lower, R upper) : sparse_row_keys(nullptr), sparse(false), name(nullptr), w(NULL, _n, _d), dw(NULL, _n, _d), dims({_n, _d}), random_id(next_matrix++) {
        std::default_random_engine generator;
        std::uniform_real_distribution<R> distribution(lower, upper);
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

template<typename R>
Mat<R>::Mat (const Mat<R>& m, bool copy_w, bool copy_dw) : sparse_row_keys(nullptr), sparse(m.sparse), name(m.name), w(NULL, m.dims[0], m.dims[1]), dw(NULL, m.dims[0], m.dims[1]), dims(m.dims), random_id(copy_w ? next_matrix++ : m.random_id) {
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

template<typename R>
Mat<R> Mat<R>::shallow_copy(const Mat<R>& m) {
        return Mat(m, false, true);
}

template<typename R>
void Mat<R>::set_name(string& _name) {
        name = std::make_shared<string>(_name);
}
template<typename R>
void Mat<R>::set_name(char * _name) {
        name = std::make_shared<string>(_name);
}
template<typename R>
void Mat<R>::set_name(const char * _name) {
        name = std::make_shared<string>(_name);
}

template<typename R>
void Mat<R>::print() const {

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

template<typename R>
void Mat<R>::grad() {
    if (dims[0] != 1 || dims[1] != 1) {
        std::cout << *this << std::endl;
        throw std::invalid_argument("Grad only works on a \"scalar\" matrix, a 1x1 matrix. Call G.sum or G.mean before using grad.");
    }
    dw(0) += 1;
}

template<typename R>
void Mat<R>::npy_save (string fname, string mode) {
    cnpy::npy_save(fname, w.data(), dims.data(), dims.size(), mode);
}

template<typename R>
unsigned int Mat<R>::number_of_elements() const {
    unsigned int dim = 1;
    for (auto & n : dims)
        dim *= n;
    return dim;
}

template<typename R>
void Mat<R>::npy_save (FILE * fp) {
    std::vector<char> header = cnpy::create_npy_header(w.data(),dims.data(),dims.size());
    fwrite(&header[0],sizeof(char),header.size(),fp);
    fwrite(w.data(),sizeof(R), number_of_elements(), fp);
}

template<typename R>
void Mat<R>::npy_load(cnpy::NpyArray& arr) {
    dims = {arr.shape[0], arr.shape.size() > 1 ? arr.shape[1] : 1};

    if (arr.word_size == sizeof(double)) {
        double* loaded_data_double = reinterpret_cast<double*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_double_ft(loaded_data_double, dims[0], dims[1]);
            w = wrapped_mat_double_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_double(loaded_data_double, dims[0], dims[1]);
            w = wrapped_mat_double.cast<R>();
        }
    } else if (arr.word_size == sizeof(float)) {
        float* loaded_data_float = reinterpret_cast<float*>(arr.data);
        if (arr.fortran_order) {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor> > wrapped_mat_float_ft(loaded_data_float, dims[0], dims[1]);
            w = wrapped_mat_float_ft.cast<R>();
        } else {
            Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic, Eigen::ColMajor> > wrapped_mat_float(loaded_data_float, dims[0], dims[1]);
            w = wrapped_mat_float.cast<R>();
        }
    } else {
        throw std::invalid_argument("Could not load numpy matrix : not recognized as float or double.");
    }
}

template<typename R>
void Mat<R>::npy_load(FILE * fp) {
    auto arr = cnpy::load_the_npy_file(fp);
    npy_load(arr);
    arr.destruct();
}

template<typename R>
void Mat<R>::npy_load(string fname) {
    auto arr = cnpy::npy_load(fname);
    npy_load(arr);
    arr.destruct();
}

template<typename R>
Mat<R>::~Mat() {}

template<typename R>
Mat<R> Mat<R>::RandMat(dim_t n, dim_t d, R std) {
    // is in fact using C++ 11 's rvalue, move operator,
    // so no copy is made.
    return Mat(n, d, std);
}

template<typename R>
Mat<R> Mat<R>::Empty(dim_t n, dim_t d) {
    // use an empty matrix and modify
    // it so as to not incur the filling
    // with zeros cost.
    return Mat(n, d, true);
}



template<typename R>
SHARED_MAT Mat<R>::eltmul_broadcast(
    SHARED_MAT matrix2) {
    if (dims[0] != matrix2->dims[0] || matrix2->dims[1] != 1) {
        stringstream error_msg;
        error_msg << "Matrices " << *this << " and "
                                 << *matrix2
                  << " cannot be element multiplied with broadcast,"
                     " they do not have the same dimensions.";
        throw std::invalid_argument(error_msg.str());
    }
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = (w.array().colwise() * matrix2->w.col(0).array()).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += ((out->dw).array().colwise() * (matrix2->w).col(0).array()).matrix();
            matrix2->dw.noalias() += (self->w.array() * (out->dw).array()).matrix().rowwise().sum();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::eltmul(
    SHARED_MAT matrix2) {
    if (dims[1] != matrix2->dims[1] && (dims[1] == 1 || matrix2->dims[1] == 1)) {
        if (dims[1] == 1) {
            return matrix2->eltmul_broadcast(this->shared_from_this());
        }
        return this->eltmul_broadcast(matrix2);
    }
    if (dims[0] != matrix2->dims[0] || dims[1] != matrix2->dims[1])
        throw std::invalid_argument("Matrices cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = (w.array() * matrix2->w.array()).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += ((matrix2->w).array() * (out->dw).array()).matrix();
            matrix2->dw.noalias() += ((self->w).array() * (out->dw).array()).matrix();
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::eltmul(
    R alpha) {

    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = (w.array() * alpha).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, alpha, out]() {
            self->dw.noalias() += (alpha * (out->dw).array()).matrix();
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::eltmul_broadcast_rowwise(
    SHARED_MAT row_vector) {
    if (dims[1] != row_vector->dims[1] || row_vector->dims[0] != 1)
        throw std::invalid_argument("Matrices A and B^T cannot be element multiplied with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<MAT>(
            dims[0],
            dims[1],
            true);
    out->w = (w.array().rowwise() * row_vector->w.row(0).array()).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, row_vector, out]() {
            self->dw.noalias() += ((out->dw).array().rowwise() * (row_vector->w).row(0).array()).matrix();
            row_vector->dw.noalias() += (((self->w).array() * (out->dw).array()).matrix().colwise().sum()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::eltmul_rowwise(
    SHARED_MAT matrix2) {

    if (dims[0] != matrix2->dims[1] || dims[1] != matrix2->dims[0])
        throw std::invalid_argument("Matrices A and B^T cannot be element-wise multiplied, they do not have the same dimensions.");
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = (w.array() * matrix2->w.transpose().array()).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += ((matrix2->w).transpose().array() * (out->dw).array()).matrix();
            matrix2->dw.noalias() += ((self->w).array() * (out->dw).array()).matrix().transpose();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::add(
        SHARED_MAT matrix2) {
    if (dims[1] != matrix2->dims[1] && (dims[1] == 1 || matrix2->dims[1] == 1)) {
        if (dims[1] == 1) {
            return matrix2->add_broadcast(this->shared_from_this());
        }
        return this->add_broadcast(matrix2);
    }
    if (dims[0] != matrix2->dims[0] || dims[1] != matrix2->dims[1])
        throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<R>>(
        dims[0],
        dims[1],
        true);
    out->w = w + matrix2->w;
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += out->dw;
            matrix2->dw.noalias() += out->dw;
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::sub(
        SHARED_MAT matrix2) {
    if (dims[1] != matrix2->dims[1] && (dims[1] == 1 || matrix2->dims[1] == 1)) {
        if (dims[1] == 1) {
            return matrix2->sub_broadcast_reversed(this->shared_from_this());
        }
        return this->sub_broadcast(matrix2);
    }
    if (dims[0] != matrix2->dims[0] || dims[1] != matrix2->dims[1])
        throw std::invalid_argument("Matrices cannot be added, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<R>>(
        dims[0],
        dims[1],
        true);
    out->w = w - matrix2->w;
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += out->dw;
            matrix2->dw.noalias() -= out->dw;
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::add_broadcast(SHARED_MAT matrix2) {
    // broadcast matrix 2:
    if (dims[0] != matrix2->dims[0] || matrix2->dims[1] != 1)
            throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<R>>(
            dims[0],
            dims[1],
            true);
    out->w = (w.colwise() + matrix2->w.col(0)).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += out->dw;
            matrix2->dw.noalias() += out->dw.rowwise().sum();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::sub_broadcast(SHARED_MAT matrix2) {
    // broadcast matrix 2:
    if (dims[0] != matrix2->dims[0] || matrix2->dims[1] != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<R>>(
        dims[0],
        dims[1],
        true);
    out->w = (w.colwise() - matrix2->w.col(0)).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out]() {
            self->dw.noalias() += out->dw;
            matrix2->dw.noalias() -= out->dw.rowwise().sum();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::sub_broadcast_reversed(SHARED_MAT matrix2) {
    // broadcast matrix 2:
    if (dims[0] != matrix2->dims[0] || matrix2->dims[1] != 1)
        throw std::invalid_argument("Matrices cannot be substracted with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<Mat<R>>(
        dims[0],
        dims[1],
        true);
    out->w = ((-w).colwise() + matrix2->w.col(0)).matrix();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out] () {
            self->dw.noalias() -= out->dw;
            matrix2->dw.noalias() += out->dw.rowwise().sum();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::add(std::initializer_list<SHARED_MAT> matrices) {
    auto out = std::make_shared<Mat<R>>(
        (*matrices.begin())->dims[0],
        (*matrices.begin())->dims[1],
        false);
    auto matrices_vector = vector<SHARED_MAT>(matrices);
    for (auto& matrix : matrices_vector)
        out->w += matrix->w;
    if (graph::backprop_enabled) {
        graph::emplace_back([matrices_vector, out]() {
            for (auto& matrix : matrices_vector) {
                matrix->dw.noalias() += out->dw;
            }
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::square() {
    auto out = std::make_shared<MAT>(
            dims[0],
            dims[1],
            true);
    out->w = w.array().square();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out]() {
            self->dw.noalias() += 2.0 * ((self->w).array() * (out->dw).array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::sigmoid() {
    auto out = std::make_shared<MAT>(
            dims[0],
            dims[1],
            true);
    out->w = w.unaryExpr(utils::sigmoid_operator<R>());
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out](){
            self->dw.noalias() += (((out->w).array() - out->w.array().square()) * out->dw.array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::softmax(SHARED_MAT matrix, R temperature) {
    auto out = std::make_shared<Mat<R>>(
            matrix->dims[0],
            matrix->dims[1],
            false);

    DEBUG_ASSERT_NOT_NAN(matrix->w);

    auto layer_max = matrix->w.colwise().maxCoeff().array().matrix();
    auto exped_distributions = (matrix->w.rowwise() - layer_max.row(0)).array().exp().matrix();

    auto total_distribution = exped_distributions.colwise().sum().array().matrix();
    out->w = (exped_distributions.array().rowwise() / total_distribution.row(0).array());

    DEBUG_ASSERT_POSITIVE(out->w);

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, temperature, out](){
            matrix->dw.noalias() += (((out->w).array() - out->w.array().square())/temperature * out->dw.array()).matrix();
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::steep_sigmoid(R aggressiveness) {
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = w.unaryExpr(utils::steep_sigmoid_operator<R>(aggressiveness));
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out, aggressiveness](){
            self->dw.noalias() += (aggressiveness * ((out->w).array() - out->w.array().square()) * out->dw.array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::sum() {
    auto out = std::make_shared<MAT>(1,1,true);
    out->w(0) = w.array().sum();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out]() {
            self->dw.array() += out->dw(0);
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::mean() {
    auto out = std::make_shared<MAT>(1,1,true);
    out->w(0) = w.array().mean();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out](){
            self->dw.array() += (1.0 / (self->number_of_elements())) * out->dw(0);
        });
    }
    return out;
}



template<typename R>
SHARED_MAT Mat<R>::sigmoid_binary_cross_entropy(SHARED_MAT matrix, R t) {
    assert(0 <= t && t <= 1);
    assert(matrix->dims.size() > 1);
    DEBUG_ASSERT_BOUNDS(matrix->w,0.0,1.0 + EPS);
    auto out =  std::make_shared<MAT>(
        matrix->dims[0],
        matrix->dims[1],
        true);

    auto x = matrix->w.array().unaryExpr(utils::sigmoid_operator<R>());

    out->w = (-(t * (x + EPS).log() + (1.0-t) * (1.0 - x + EPS).log())).matrix();

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, t, out, x](){
            auto x = matrix->w.array();
            matrix->dw.array() += (t - x) * out->dw.array();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::binary_cross_entropy(SHARED_MAT matrix, R t) {
    assert(0 <= t && t <= 1);
    assert(matrix->dims.size() > 1);
    auto out =  std::make_shared<MAT>(
        matrix->dims[0],
        matrix->dims[1],
        true);

    auto x = matrix->w.array();

    out->w = (-(t * (x + EPS).log() + (1.0-t) * (1.0 - x + EPS).log())).matrix();

    DEBUG_ASSERT_NOT_NAN(out->w);

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, t, out](){
            auto x = matrix->w.array();
            matrix->dw.array() += (t-x) / (x*(x- 1.0) + EPS)* out->dw.array();
            DEBUG_ASSERT_NOT_NAN(matrix->dw);
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::cross_entropy(SHARED_MAT matrix, uint answer_idx) {
    DEBUG_ASSERT_BOUNDS(matrix->w,0.0,1.0 + EPS);
    assert(matrix->dims.size() > 1);
    auto out =  std::make_shared<MAT>(1, 1, true);

    auto x = matrix->w.array();

    out->w(0,0) = - std::log(x(answer_idx, 0) + EPS);

    DEBUG_ASSERT_NOT_NAN(out->w);

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, answer_idx, out](){
            auto x = matrix->w.array();
            matrix->dw(answer_idx, 0) += -1.0/(x(answer_idx, 0) + EPS) * out->dw(0,0);
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::softmax_cross_entropy(SHARED_MAT matrix, uint answer_idx) {
    auto out =  std::make_shared<MAT>(1, 1, true);

    SHARED_MAT probs = softmax(matrix);
    out->w(0,0) = -std::log(probs->w(answer_idx, 0));

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, probs, answer_idx, out](){
            matrix->dw += probs->w * out->dw(0,0);
            // write gradients into log probabilities
            matrix->dw(answer_idx, 0) -= 1 * out->dw(0,0);
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::log() {
    assert(dims.size() > 1);
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = w.array().log();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out](){
            self->dw.noalias() += ((1.0 / (self->w).array()) * (out->dw).array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::exp() {
    assert(dims.size() > 1);
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = w.array().exp();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out]() {
            self->dw.noalias() += ((out->w).array() * (out->dw).array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::hstack(SHARED_MAT matrix1, SHARED_MAT matrix2) {
    if (matrix1->dims[0] != matrix2->dims[0])
        throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
    auto out = std::make_shared<MAT>(
        matrix1->dims[0],
        matrix1->dims[1] + matrix2->dims[1],
        true
    );
    out->w.block(0,0, matrix1->dims[0], matrix1->dims[1]) = matrix1->w;
    out->w.block(0,matrix1->dims[1], matrix2->dims[0], matrix2->dims[1]) = matrix2->w;
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw.block(0,0, matrix1->dims[0], matrix1->dims[1]);
            matrix2->dw.noalias() += out->dw.block(0,matrix1->dims[1], matrix2->dims[0], matrix2->dims[1]);
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::hstack(std::initializer_list<SHARED_MAT> matrices) {
    vector<SHARED_MAT> matrices_vector(matrices);
    return hstack(matrices_vector);
}

template<typename R>
SHARED_MAT Mat<R>::hstack(const std::vector<SHARED_MAT>& matrices) {
    int n = -1;
    int d_total = 0;
    for (auto& mat : matrices) {
        if (n == -1) {
            n = mat->dims[0];
        } else {
            if (mat->dims[0] != n) {
                throw std::invalid_argument("Matrices cannot be joined -- they do not have the same number of rows.");
            }
        }
        d_total+= mat->dims[1];
    }
    auto out = std::make_shared<MAT>(
        n,
        d_total,
        true
    );
    int offset = 0;
    for (auto& mat : matrices) {
        out->w.block(0, offset, mat->dims[0], mat->dims[1]) = mat->w;
        offset += mat->dims[1];
    }
    if (graph::backprop_enabled) {
        graph::emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                mat->dw.noalias() += out->dw.block(0, offset, mat->dims[0], mat->dims[1]);
                offset += mat->dims[1];
            }
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::vstack(SHARED_MAT matrix1, SHARED_MAT matrix2) {
    if (matrix1->dims[1] != matrix2->dims[1])
        throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
    auto out = std::make_shared<MAT>(
        matrix1->dims[0] + matrix2->dims[0],
        matrix1->dims[1],
        true
    );
    out->w.block(0,0, matrix1->dims[0], matrix1->dims[1]) = matrix1->w;
    out->w.block(matrix1->dims[0],0, matrix2->dims[0], matrix2->dims[1]) = matrix2->w;
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix1, matrix2, out]() {
            matrix1->dw.noalias() += out->dw.block(0,0, matrix1->dims[0], matrix1->dims[1]);
            matrix2->dw.noalias() += out->dw.block(matrix1->dims[0],0, matrix2->dims[0], matrix2->dims[1]);
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::vstack(std::initializer_list<SHARED_MAT> matrices) {
    vector<SHARED_MAT> matrices_vector(matrices);
    return vstack(matrices_vector);
}

template<typename R>
SHARED_MAT Mat<R>::vstack(const std::vector<SHARED_MAT>& matrices) {
    assert(matrices.size() > 0);
    assert(matrices[0]->dims.size() > 1);
    int d = matrices[0]->dims[1];
    int n_total = 0;
    for (auto& mat : matrices) {
        if (mat->dims[1] != d) {
            throw std::invalid_argument("Matrices cannot be horizontally stacked -- they do not have the same number of cols.");
        }
        n_total += mat->dims[0];
    }
    auto out = std::make_shared<MAT>(
        n_total,
        d,
        true
    );
    int offset = 0;
    for (auto& mat : matrices) {
        out->w.block(offset, 0, mat->dims[0], mat->dims[1]) = mat->w;
        offset += mat->dims[0];
    }
    if (graph::backprop_enabled) {
        graph::emplace_back([matrices, out]() {
            int offset = 0;
            for (auto & mat : matrices) {
                mat->dw.noalias() += out->dw.block(offset,0, mat->dims[0], mat->dims[1]);
                offset += mat->dims[0];
            }
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::T() {
    assert(dims.size() > 1);
    auto out = std::make_shared<MAT>(
        dims[1],
        dims[0],
        true);
    out->w = w.transpose();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out]() {
            self->dw.noalias() += (out->dw).transpose();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::tanh() {
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = w.unaryExpr(utils::tanh_operator<R>());
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out](){
            self->dw.noalias() += (out->w.unaryExpr(utils::dtanh_operator<R>()).array() * out->dw.array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::relu() {
    auto out = std::make_shared<MAT>(
        dims[0],
        dims[1],
        true);
    out->w = w.unaryExpr(utils::relu_operator<R>());
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out](){
            self->dw.noalias() += (out->w.unaryExpr(utils::sign_operator<R>()).array() * out->dw.array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::mul(
    SHARED_MAT matrix2) {
    if (dims[1] != matrix2->dims[0])
        throw std::invalid_argument("matmul dimensions misaligned.");
    auto out = std::make_shared<MAT>(
        dims[0],
        matrix2->dims[1],
        true);
    out->w = w * matrix2->w;
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, matrix2, out](){
            self->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
            matrix2->dw.noalias() += self->w.transpose() * (out->dw);
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::mul_with_bias(
    SHARED_MAT matrix1,
    SHARED_MAT matrix2,
    SHARED_MAT bias) {
    if (matrix1->dims[1] != matrix2->dims[0])
            throw std::invalid_argument("matmul dimensions misaligned.");
    if (matrix1->dims[0] != bias->dims[0] || bias->dims[1] != 1)
            throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<MAT>(
            matrix1->dims[0],
            matrix2->dims[1],
            true);
    out->w = ((matrix1->w * matrix2->w).colwise() + bias->w.col(0)).matrix();
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix1, matrix2, bias, out]() {
            matrix1->dw.noalias() += (out->dw) * ((matrix2->w).transpose());
            matrix2->dw.noalias() += matrix1->w.transpose() * (out->dw);
            bias->dw.noalias()    += out->dw.rowwise().sum().matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::mul_add_broadcast_mul_with_bias(
    SHARED_MAT matrix1,
    SHARED_MAT input_to_1,
    SHARED_MAT matrix2,
    SHARED_MAT input_to_2,
    SHARED_MAT bias) {
    if (matrix1->dims[1] != input_to_1->dims[0])
        throw std::invalid_argument("matmul 1 dimensions misaligned.");
    if (matrix2->dims[1] != input_to_2->dims[0])
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2->dims[0] != bias->dims[0] || matrix1->dims[0] != bias->dims[0] || input_to_1->dims[1] != 1 || bias->dims[1] != 1)
        throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    auto out = std::make_shared<MAT>(
            matrix1->dims[0],
            input_to_2->dims[1],
            true);
    // both input to 1 and bias are columns,
    // so we add both of those before adding the true matrix
    // product in broadcasted form
    out->w = (
          (
              (
                  (matrix2->w * input_to_2->w)
              )
          ).colwise() + (bias->w + (matrix1->w * input_to_1->w)).col(0)
      ).matrix();
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out] () {
            // first multiply:
            // broadcasting input means taking outer product here:
            matrix1->dw += ((out->dw).rowwise().sum() * ((input_to_1->w).transpose()));
            // broadcasting output means sum after the reverse product here:
            input_to_1->dw.noalias() += (matrix1->w.transpose() * (out->dw)).rowwise().sum();
            // second multiply:
            matrix2->dw.noalias() += (out->dw) * ((input_to_2->w).transpose());

            input_to_2->dw.noalias() += matrix2->w.transpose() * (out->dw);
            // bias vector:
            bias->dw.noalias() += out->dw.rowwise().sum();
        });
    }
    return out;
}


template<typename R>
SHARED_MAT Mat<R>::mul_add_mul_with_bias(std::initializer_list<SHARED_MAT> matrices) {
    vector<SHARED_MAT> matrices_vector(matrices);
    return mul_add_mul_with_bias(matrices_vector);
}

template<typename R>
SHARED_MAT Mat<R>::mul_add_mul_with_bias(const vector<SHARED_MAT>& matrices) {
    auto out = std::make_shared<MAT>(
            matrices[0]->dims[0],
            matrices[1]->dims[1],
            false);
    auto matrices_ptr = matrices.begin();
    while (matrices_ptr != (matrices.end() - 1)) {
        out->w += (*matrices_ptr)->w * (*(matrices_ptr + 1))->w;
        DEBUG_ASSERT_MAT_NOT_NAN(out);
        DEBUG_ASSERT_MAT_NOT_NAN((*matrices_ptr));
        DEBUG_ASSERT_MAT_NOT_NAN((*(matrices_ptr + 1)));
        matrices_ptr+=2;
    }

    DEBUG_ASSERT_NOT_NAN((*(matrices.begin() + matrices.size() - 1))->w);
    out->w.colwise() += (*(matrices.begin() + matrices.size() - 1))->w.col(0);
    if (graph::backprop_enabled) {
        graph::emplace_back([matrices, out](){
            auto matrices_ptr = matrices.begin();
            while (matrices_ptr != (matrices.end() - 1)) {
                (*matrices_ptr)->dw.noalias()     += (out->dw) * (*(matrices_ptr+1))->w.transpose();
                (*(matrices_ptr+1))->dw.noalias() += (*matrices_ptr)->w.transpose() * (out->dw);
                matrices_ptr+=2;
            }
            auto bias = *(matrices.begin() + matrices.size() - 1);
            bias->dw.noalias() += out->dw.rowwise().sum();
        });
    }

    DEBUG_ASSERT_NOT_NAN(out->w);
    return out;
}

// operation of the form (A * x + B * y) + C, called with mul_add_mul_with_bias(A, x, B, y, C)
template<typename R>
SHARED_MAT Mat<R>::mul_add_mul_with_bias(
    SHARED_MAT matrix1,
    SHARED_MAT input_to_1,
    SHARED_MAT matrix2,
    SHARED_MAT input_to_2,
    SHARED_MAT bias) {
    if (matrix1->dims[1] != input_to_1->dims[0])
        throw std::invalid_argument("matmul 1 dimensions misaligned.");
    if (matrix2->dims[1] != input_to_2->dims[0])
        throw std::invalid_argument("matmul 2 dimensions misaligned.");
    if (matrix2->dims[0] != bias->dims[0] || matrix1->dims[0] != bias->dims[0] || bias->dims[1] != 1)
        throw std::invalid_argument("Matrices cannot be added with broadcast, they do not have the same dimensions.");
    if (input_to_1->dims[1] != input_to_2->dims[1]) {
        if (input_to_1->dims[1] == 1) {
            return mul_add_broadcast_mul_with_bias(matrix1, input_to_1, matrix2, input_to_2, bias);
        }
        return mul_add_broadcast_mul_with_bias(matrix2, input_to_2, matrix1, input_to_1, bias);
    }
    auto out = std::make_shared<MAT>(
            matrix1->dims[0],
            input_to_1->dims[1],
            true);
    out->w = (
              (
                  (
                      (matrix1->w * input_to_1->w) +
                      (matrix2->w * input_to_2->w)
                  )
              ).colwise() + bias->w.col(0)
          ).matrix();
    if (graph::backprop_enabled) {
        graph::emplace_back([matrix1, input_to_1, matrix2, input_to_2, bias, out](){
            // first multiply:
            // broadcasting input means taking outer product here:
            matrix1->dw += (out->dw * (input_to_1->w).transpose());
            // broadcasting output means sum after the reverse product here:
            input_to_1->dw.noalias() += matrix1->w.transpose() * (out->dw);
            // second multiply:
            matrix2->dw.noalias() += (out->dw) * (input_to_2->w).transpose();

            input_to_2->dw.noalias() += matrix2->w.transpose() * (out->dw);
            // bias vector:
            bias->dw.noalias() += out->dw.rowwise().sum();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::rows_pluck(
        Indexing::Index indices
        ) {
    auto out = std::make_shared<MAT>(
        dims[1],
        indices.size(),
        true);

    for (std::size_t offset = 0; offset < indices.size(); ++offset) {
        out->w.col(offset) = w.row(indices[offset]).transpose();
    }
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out, indices](){
            auto index_ptr = indices.data();
            for (std::size_t i = 0; i < out->dims[1]; ++i) {
                // for each row do the same operation as for row_pluck:
                self->dw.row(*index_ptr).noalias() += out->dw.col(i).transpose();
                index_ptr++;
            }
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::dropout(
    SHARED_MAT matrix,
    R drop_prob) {

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = std::make_shared<MAT>(
        matrix->dims[0],
        matrix->dims[1],
        true);

    auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(matrix->dims[0], matrix->dims[1]);

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix->w.data();
    auto out_ptr  = out->w.data();
    auto bool_ptr = bool_mat->data();

    for (int i = 0; i < matrix->number_of_elements();++i) {
        (*bool_ptr) = distribution(generator) ? 1.0 : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, bool_mat](){
            matrix->dw += (out->dw.array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::dropout_normalized(
    SHARED_MAT matrix,
    R drop_prob) {

    assert(0.0 <= drop_prob && drop_prob <= 1.0);

    // no dropout happens.
    if (drop_prob < 1e-6)
        return matrix;

    auto out = std::make_shared<MAT>(
        matrix->dims[0],
        matrix->dims[1],
        true);

    auto bool_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(matrix->dims[0], matrix->dims[1]);

    std::default_random_engine generator;
    std::bernoulli_distribution distribution(1.0 - drop_prob);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix->w.data();
    auto out_ptr  = out->w.data();
    auto bool_ptr = bool_mat->data();

    R normalized_drop_prob = 1.0 / (1.0 - drop_prob);
    for (unsigned int i = 0; i < matrix->number_of_elements();++i) {
        (*bool_ptr) = distribution(generator) ? normalized_drop_prob : 0.0;
        (*out_ptr) = (*bool_ptr) > 0 ? *data_ptr : 0.0;
        out_ptr++;
        data_ptr++;
        bool_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, bool_mat](){
            matrix->dw += (out->dw.array() * (*bool_mat).array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::fast_dropout(SHARED_MAT matrix) {
    auto out = std::make_shared<MAT>(
        matrix->dims[0],
        matrix->dims[1],
        true);

    auto randn_mat = std::make_shared<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>>(matrix->dims[0], matrix->dims[1]);

    std::default_random_engine generator;
    std::normal_distribution<R> distribution(1.0, 1.0);
    std::random_device rd;
    generator.seed(rd());

    auto data_ptr = matrix->w.data();
    auto out_ptr  = out->w.data();
    auto randn_ptr = randn_mat->data();

    for (unsigned int i = 0; i < matrix->number_of_elements();++i) {
        (*randn_ptr) = distribution(generator);
        (*out_ptr) = (*randn_ptr) * *data_ptr;
        out_ptr++;
        data_ptr++;
        randn_ptr++;
    }

    if (graph::backprop_enabled) {
        graph::emplace_back([matrix, out, randn_mat](){
            matrix->dw += (out->dw.array() * (*randn_mat).array()).matrix();
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::rows_cols_pluck(
        Indexing::Index row_indices,
        Indexing::Index col_indices) {
    if (row_indices.size() != col_indices.size())
        throw std::invalid_argument("Cannot pluck column row pairs, not the same amount of row and column indices.");
    auto out = std::make_shared<MAT>(
        1,
        row_indices.size(),
        true);
    for (int offset = 0; offset < row_indices.size(); ++offset)
        out->w(offset) = w(row_indices[offset], col_indices[offset]);
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out, row_indices, col_indices](){
            auto row_index_ptr = row_indices.data();
            auto col_index_ptr = col_indices.data();
            for (int i = 0; i < out->dims[1]; ++i) {
                // for each row do the same operation as for row_pluck:
                self->dw(*row_index_ptr, *col_index_ptr) += out->dw(i);
                row_index_ptr++;
                col_index_ptr++;
            }
        });
    }
    return out;
}

template<typename R>
SHARED_MAT Mat<R>::row_pluck(
        int row) {
    auto out = std::make_shared<MAT>(dims[1], 1, true);
    out->w = w.row(row).transpose();
    if (graph::backprop_enabled) {
        SHARED_MAT self = this->shared_from_this();
        graph::emplace_back([self, out, row]() {
            self->dw.row(row).noalias() += out->dw.col(0).transpose();
        });
    }
    return out;
}



template<typename R>
std::ostream& operator<<(std::ostream& strm, const Mat<R>& a) {
    if (a.name != 0) {
        return strm << "<#Mat name=\"" << *a.name<< "\" n=" << a.dims[0] << ", d=" << a.dims[1] << ">";
    } else {
        return strm << "<#Mat n=" << a.dims[0] << ", d=" << a.dims[1] << ">";
    }
}

template std::ostream& operator<< <double>(std::ostream& strm, const Mat<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const Mat<float>& a);

template <typename R>
std::size_t std::hash<Mat<R>>::operator()(const Mat<R>& k) const {
    return k.random_id;
}

template std::size_t std::hash<Mat<float>>::operator()(const Mat<float>& k)   const;
template std::size_t std::hash<Mat<double>>::operator()(const Mat<double>& k) const;

template <typename R>
bool operator!=(const Mat<R>& A, const Mat<R>& B) {
    return A.random_id != B.random_id;
}

template bool operator!=(const Mat<float>&, const Mat<float>&);
template bool operator!=(const Mat<double>&, const Mat<double>&);

template <typename R>
bool operator==(const Mat<R>& A, const Mat<R>& B) {
    return A.random_id == B.random_id;
}

template bool operator==<float>(const Mat<float>&, const Mat<float>&);
template bool operator==<double>(const Mat<double>&, const Mat<double>&);

template<typename R>
int argmax(std::shared_ptr<Mat<R>> A) {
    int i = 0;
    R current_max = -std::numeric_limits<R>::infinity();
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

template<typename R>
int argmax_slice(std::shared_ptr<Mat<R>> A, int min, int max) {
    int i = 0;
    R current_max = -std::numeric_limits<R>::infinity();
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

template<typename R>
void utils::save_matrices(vector<std::shared_ptr<Mat<R>>>& parameters, string dirname) {
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

template<typename R>
void utils::load_matrices(vector< std::shared_ptr<Mat<R>> >& parameters, string dirname) {
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
