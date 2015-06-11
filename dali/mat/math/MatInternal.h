#ifndef DALI_MAT_MATH_MAT_INTERNAL_H
#define DALI_MAT_MATH_MAT_INTERNAL_H

#include <atomic>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Eigen>


typedef unsigned int dim_t;

template<typename R>
class MatInternal {
    private:
        static std::atomic<int> next_matrix;
    public:
        typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
        typedef Eigen::Map<eigen_mat> eigen_mat_view;

        eigen_mat w;
        std::vector<dim_t> dims;
        const size_t id;

        MatInternal(dim_t n, dim_t d, bool empty=false);
        MatInternal(const MatInternal<R>& m);

        R& operator()(int i, int j);
        R operator()(int i, int j) const;

        R& operator()(int i);
        R operator()(int i) const;

        const R* data() const;
        R* data();

        void print() const;

        operator eigen_mat();
};

template<typename R>
class GradInternal {
    public:
        typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;
        typedef Eigen::Map<eigen_mat> eigen_mat_view;

        eigen_mat dw;

        GradInternal(dim_t n, dim_t d, bool empty=true);
        GradInternal(const GradInternal<R>& g);

        R& operator()(int i, int j);
        R operator()(int i, int j) const;

        R& operator()(int i);
        R operator()(int i) const;

        const R* data() const;
        R* data();

        operator eigen_mat();
};

#endif
