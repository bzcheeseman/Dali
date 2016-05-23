#ifndef SOLVER_MAT_H
#define SOLVER_MAT_H

#include "dali/tensor/tensor.h"

#define SOLVER_MAT_DEFAULT_STEP_SIZE_H 0.035

namespace solver {
    typedef void* cache_key_t;
    typedef Array cache_t;

    enum Method {
        METHOD_UNINITIALIZED,
        METHOD_ADAGRAD,
        METHOD_ADADELTA,
        METHOD_SGD,
        METHOD_RMSPROP,
        METHOD_ADAM,
        METHOD_RMSPROPMOMENTUM,
    };

    extern bool nan_protection;
    const double SMOOTH_DEFAULT = 1e-4;

    class AbstractSolver {
        public:
            Method method;

            double clip_norm;
            double clip_abs;

            double smooth_eps;
            double regc;

            AbstractSolver();
            AbstractSolver(const double& clip_norm,
                           const double& smooth_eps,
                           const double& regc,
                           Method method);
            virtual void step(std::vector<Tensor>&) = 0;
            virtual void reset_caches(const std::vector<Tensor>&);
            virtual void create_gradient_caches(const std::vector<Tensor>&);
    };

    class SGD : public AbstractSolver {
        public:
            // This can be overriden by parameter passed to step function.
            double step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;

            SGD(const double& clip_norm=5.0,
                const double& regc=0.0);
            SGD(const std::vector<Tensor>& params,
                const double& clip_norm=5.0,
                const double& regc=0.0);
            virtual void step(std::vector<Tensor>&);
            virtual void step(std::vector<Tensor>&, const double& step_size);
    };

    class AdaGrad : public AbstractSolver {
        public:
            // This can be overriden by parameter passed to step function.
            double step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;
            std::unordered_map<cache_key_t, cache_t> gsums;
            AdaGrad(const double& smooth_eps=SMOOTH_DEFAULT,
                    const double& clip_norm=100.0,
                    const double& regc=0.0);
            AdaGrad(const std::vector<Tensor>& params,
                    const double& smooth_eps = SMOOTH_DEFAULT,
                    const double& clip_norm=100.0,
                    const double& regc=0.0);
            virtual void step(std::vector<Tensor>&);
            virtual void step(std::vector<Tensor>&, const double& step_size);
            virtual void create_gradient_caches(const std::vector<Tensor>& params);
            virtual void reset_caches(const std::vector<Tensor>& params);
    };

    class RMSProp : public AdaGrad {
        public:
            double step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;
            double decay_rate;

            RMSProp(const double& _decay_rate=0.999,
                    const double& smooth_eps=SMOOTH_DEFAULT,
                    const double& clip_norm=100.0,
                    const double& regc=0.0);
            RMSProp(const std::vector<Tensor>& params,
                    const double& _decay_rate=0.999,
                    const double& smooth_eps=SMOOTH_DEFAULT,
                    const double& clip_norm=100.0,
                    const double& regc=0.0);
            virtual void step(std::vector<Tensor>&);
            virtual void step(std::vector<Tensor>&, const double& step_size);
    };

    class RMSPropMomentum : public AbstractSolver {
        public:
            double step_size = SOLVER_MAT_DEFAULT_STEP_SIZE_H;
            double decay_rate;
            double momentum;

            std::unordered_map<cache_key_t, cache_t> n_cache;
            std::unordered_map<cache_key_t, cache_t> g_cache;
            std::unordered_map<cache_key_t, cache_t> momentum_cache;

            RMSPropMomentum(const double& decay_rate=0.95,
                            const double& momentum=0.9,
                            const double& step_size=1e-4,
                            const double& smooth_eps=1e-4,
                            const double& clip_norm=100.0,
                            const double& regc=0.0);
            RMSPropMomentum(const std::vector<Tensor>&,
                            const double& decay_rate=0.95,
                            const double& momentum=0.9,
                            const double& step_size=1e-4,
                            const double& smooth_eps=1e-4,
                            const double& clip_norm=100.0,
                            const double& regc=0.0);
            virtual void step(std::vector<Tensor>&);
            virtual void step(std::vector<Tensor>&, const double& step_size);
            virtual void create_gradient_caches(const std::vector<Tensor>&);
            virtual void reset_caches(const std::vector<Tensor>&);
    };

    class AdaDelta : public AbstractSolver {
        public:
            double rho;
            std::unordered_map<cache_key_t, cache_t> xsums;
            std::unordered_map<cache_key_t, cache_t> gsums;
            AdaDelta(const double& rho=0.95,
                      const double& smooth_eps=1e-4,
                      const double& clip_norm=100.0,
                      const double& regc=0.0);
            AdaDelta(const std::vector<Tensor>&,
                     const double& rho=0.95,
                     const double& smooth_eps=1e-4,
                     const double& clip_norm=100.0,
                     const double& regc = 0.0);
            virtual void step(std::vector<Tensor>&);
            virtual void create_gradient_caches(const std::vector<Tensor>&);
            virtual void reset_caches(const std::vector<Tensor>&);
    };

    class Adam : public AbstractSolver {
        public:
            double b1;
            double b2;
            double step_size;
            // This is a large integer:
            unsigned long long epoch;
            std::unordered_map<cache_key_t, cache_t> xsums;
            std::unordered_map<cache_key_t, cache_t> gsums;
            Adam(const double& step_size=0.0002,
                 const double& b1=0.5,
                 const double& b2=1e-6,
                 const double& smooth_eps = SMOOTH_DEFAULT,
                 const double& clip_norm=100.0,
                 const double& regc=0.0);
            Adam(const std::vector<Tensor>&,
                 const double& step_size=0.0002,
                 const double& b1=0.5,
                 const double& b2=1e-6,
                 const double& smooth_eps=SMOOTH_DEFAULT,
                 const double& clip_norm=100.0,
                 const double& regc=0.0);
            virtual void step(std::vector<Tensor>&);
            virtual void step(std::vector<Tensor>&,
                              const double& step_size);
            virtual void create_gradient_caches(const std::vector<Tensor>&);
            virtual void reset_caches(const std::vector<Tensor>&);
    };

    std::shared_ptr<AbstractSolver> construct(std::string solver_name,
                                              std::vector<Tensor>& params,
                                              const double& step_size = 0.01,
                                              const double& regc = 0.0);

} // namespace solver
#endif
