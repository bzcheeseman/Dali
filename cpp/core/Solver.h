#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "core/Mat.h"

namespace Solver {

    template<typename T> class RMSProp {
        T decay_rate;
        T smooth_eps;
        T clipval;
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            std::unordered_map<mat, eigen_mat> gsums;
            RMSProp (T decay_rate= 0.999, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            RMSProp (std::vector<shared_mat>&, T decay_rate= 0.999, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            void step( std::vector<shared_mat>&, T, T);
            void create_gradient_caches(std::vector<shared_mat>&);
    };

    template<typename T> class AdaDelta {
        T rho;
        T smooth_eps;
        T clipval;
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            std::unordered_map<mat, eigen_mat> gsums;
            std::unordered_map<mat, eigen_mat> xsums;
            AdaDelta (T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            AdaDelta (std::vector<shared_mat>&, T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            void step( std::vector<shared_mat>&, T);
            void create_gradient_caches(std::vector<shared_mat>&);
    };

    template<typename T> class AdaGrad {
        T smooth_eps;
        T clipval;
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            std::unordered_map<mat, eigen_mat> gsums;
            AdaGrad (T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            AdaGrad (std::vector<shared_mat>&, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            void step( std::vector<shared_mat>&, T, T);
            void reset_caches( std::vector<shared_mat>&);
            void create_gradient_caches(std::vector<shared_mat>&);
    };

    template<typename T> class SGD {
        T clipval;
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        public:
            SGD (T clipval = 5.0);
            SGD (std::vector<shared_mat>&, T clipval = 5.0);
            void step( std::vector<shared_mat>&, T, T = 0.0);
    };
}
#endif
