#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "core/Mat.h"

#define SMOOTH_DEFAULT 1e-9

#define SOLVER_MAT_TYPEDEF_H typedef Mat<T>                      mat; \
        typedef std::shared_ptr<mat> shared_mat; \
        typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

#define SOLVER_MAT_DEFAULT_STEP_SIZE_H 0.035

namespace Solver {

    template<typename T> class AbstractSolver {
        SOLVER_MAT_TYPEDEF_H
        public:
            T clipval;
            T smooth_eps;
            T regc;
            AbstractSolver();
            AbstractSolver(T clipval, T smooth_eps, T regc);
            virtual void step( std::vector<shared_mat>& ) = 0;
            virtual void reset_caches( std::vector<shared_mat>&);
    };

    template<typename T> class SGD : public AbstractSolver<T> {
        SOLVER_MAT_TYPEDEF_H
        public:
            SGD (T clipval = 5.0, T regc = 0.0);
            SGD (std::vector<shared_mat>&, T clipval = 5.0, T regc = 0.0);
            virtual void step( std::vector<shared_mat>&);
            virtual void step( std::vector<shared_mat>&, T step_size);
    };

    template<typename T> class AdaGrad : public AbstractSolver<T> {
        SOLVER_MAT_TYPEDEF_H
        public:
            std::unordered_map<mat, eigen_mat> gsums;
            AdaGrad (T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            AdaGrad (std::vector<shared_mat>&, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            virtual void step( std::vector<shared_mat>&);
            virtual void step( std::vector<shared_mat>&, T step_size);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches(std::vector<shared_mat>&);
    };

    template<typename T> class RMSProp : public AdaGrad<T> {
        SOLVER_MAT_TYPEDEF_H
        public:
            T decay_rate;
            std::unordered_map<mat, eigen_mat> gsums;
            RMSProp (T _decay_rate= 0.999, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            RMSProp (std::vector<shared_mat>&, T _decay_rate= 0.999, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            virtual void step(std::vector<shared_mat>&);
            virtual void step(std::vector<shared_mat>&, T step_size);
    };

    template<typename T> class AdaDelta : public AbstractSolver<T> {
        SOLVER_MAT_TYPEDEF_H
        public:
            T rho;
            std::unordered_map<mat, eigen_mat> xsums;
            std::unordered_map<mat, eigen_mat> gsums;
            AdaDelta (T rho= 0.95, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            AdaDelta (std::vector<shared_mat>&, T rho= 0.95, T smooth_eps =SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            virtual void step(std::vector<shared_mat>&);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches(std::vector<shared_mat>&);
    };

    template<typename T> class Adam : public AbstractSolver<T> {
        SOLVER_MAT_TYPEDEF_H
        public:
            T b1;
            T b2;
            // This is a large integer:
            unsigned long long epoch;
            std::unordered_map<mat, eigen_mat> xsums;
            std::unordered_map<mat, eigen_mat> gsums;
            Adam (T b1 = 0.1, T b2 = 0.001, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            Adam (std::vector<shared_mat>&, T b1 = 0.1, T b2 = 0.001, T smooth_eps = SMOOTH_DEFAULT, T clipval = 5.0, T regc = 0.0);
            virtual void step(std::vector<shared_mat>&);
            virtual void step(std::vector<shared_mat>&, T step_size);
            virtual void create_gradient_caches(std::vector<shared_mat>&);
            virtual void reset_caches(std::vector<shared_mat>&);
    };
}
#endif
