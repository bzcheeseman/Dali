#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "core/Mat.h"

#define SMOOTH_DEFAULT 1e-9

#define SOLVER_MAT_TYPEDEF_H typedef Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> eigen_mat;

#define SOLVER_MAT_DEFAULT_STEP_SIZE_H 0.035

namespace Solver {

    template<typename R> class AbstractSolver {
        SOLVER_MAT_TYPEDEF_H
        public:
            R clipval;
            R smooth_eps;
            R regc;
            AbstractSolver();
            AbstractSolver(R clipval, R smooth_eps, R regc);
            virtual void step( std::vector<Mat<R>>& ) = 0;
            virtual void reset_caches( std::vector<Mat<R>>&);
    };

    template<typename R> class SGD : public AbstractSolver<R> {
        SOLVER_MAT_TYPEDEF_H
        public:
            SGD (R clipval = 5.0, R regc = 0.0);
            SGD (std::vector<Mat<R>>&, R clipval = 5.0, R regc = 0.0);
            virtual void step( std::vector<Mat<R>>&);
            virtual void step( std::vector<Mat<R>>&, R step_size);
    };

    template<typename R> class AdaGrad : public AbstractSolver<R> {
        SOLVER_MAT_TYPEDEF_H
        public:
            std::unordered_map<Mat<R>, eigen_mat> gsums;
            AdaGrad (R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            AdaGrad (std::vector<Mat<R>>&, R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            virtual void step( std::vector<Mat<R>>&);
            virtual void step( std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class RMSProp : public AdaGrad<R> {
        SOLVER_MAT_TYPEDEF_H
        public:
            R decay_rate;
            std::unordered_map<Mat<R>, eigen_mat> gsums;
            RMSProp (R _decay_rate= 0.999, R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            RMSProp (std::vector<Mat<R>>&, R _decay_rate= 0.999, R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
    };

    template<typename R> class AdaDelta : public AbstractSolver<R> {
        SOLVER_MAT_TYPEDEF_H
        public:
            R rho;
            std::unordered_map<Mat<R>, eigen_mat> xsums;
            std::unordered_map<Mat<R>, eigen_mat> gsums;
            AdaDelta (R rho= 0.95, R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            AdaDelta (std::vector<Mat<R>>&, R rho= 0.95, R smooth_eps =SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };

    template<typename R> class Adam : public AbstractSolver<R> {
        SOLVER_MAT_TYPEDEF_H
        public:
            R b1;
            R b2;
            // This is a large integer:
            unsigned long long epoch;
            std::unordered_map<Mat<R>, eigen_mat> xsums;
            std::unordered_map<Mat<R>, eigen_mat> gsums;
            Adam (R b1 = 0.1, R b2 = 0.001, R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            Adam (std::vector<Mat<R>>&, R b1 = 0.1, R b2 = 0.001, R smooth_eps = SMOOTH_DEFAULT, R clipval = 5.0, R regc = 0.0);
            virtual void step(std::vector<Mat<R>>&);
            virtual void step(std::vector<Mat<R>>&, R step_size);
            virtual void create_gradient_caches(std::vector<Mat<R>>&);
            virtual void reset_caches(std::vector<Mat<R>>&);
    };
}
#endif
