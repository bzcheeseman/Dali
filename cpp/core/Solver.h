#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "core/Mat.h"

namespace Solver {

    template<typename T> class SGD {
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        public:
            T clipval;
            T smooth_eps;
            SGD (T clipval = 5.0);
            SGD (std::vector<shared_mat>&, T clipval = 5.0);
            virtual void step( std::vector<shared_mat>&);
            virtual void step( std::vector<shared_mat>&, T step_size, T regc = 0.0);
            virtual void reset_caches( std::vector<shared_mat>&);
    };

    template<typename T> class RMSProp : public SGD {
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            T decay_rate;
            std::unordered_map<mat, eigen_mat> gsums;
            RMSProp (T decay_rate= 0.999, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            RMSProp (std::vector<shared_mat>&, T decay_rate= 0.999, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0);
            void step( std::vector<shared_mat>&, T, T);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches( std::vector<shared_mat>&);
    };

    template<typename T> class AdaDelta : public RMSProp {
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            T rho;
            std::unordered_map<mat, eigen_mat> xsums;
            AdaDelta (T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            AdaDelta (std::vector<shared_mat>&, T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            virtual void step( std::vector<shared_mat>&, T regc);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches( std::vector<shared_mat>&);
    };

    template<typename T> class AdaGrad {
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            std::unordered_map<mat, eigen_mat> gsums;
            AdaGrad (T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            AdaGrad (std::vector<shared_mat>&, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            virtual void step( std::vector<shared_mat>&, T, T regc);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches( std::vector<shared_mat>&);
    };

    template<typename T> class Adam : public AdaDelta {
        typedef Mat<T>                      mat;
        typedef std::shared_ptr<mat> shared_mat;
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

        public:
            Adam (T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0);
            Adam (std::vector<shared_mat>&, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0);
            virtual void step( std::vector<shared_mat>&, T, T regc);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches( std::vector<shared_mat>&);
    };
}
#endif
