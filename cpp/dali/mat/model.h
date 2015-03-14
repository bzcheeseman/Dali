#ifndef MAT_MODEL_H
#define MAT_MODEL_H

#include <memory>

#include "dali/utils/configuration.h"

class Model {
    private:
        mutable std::shared_ptr<Conf> _conf;
    public:
        // returns default configuration.
        // Default configuration acts like a schema for what the configuration
        // for this model should actually be. See examples from other models before use.
        virtual  std::shared_ptr<Conf> default_conf() const;
        // Returns mutable reference to a configuration. Should not be subclassed,
        // to ensure integrity of configuration.
        virtual Conf& conf() const final;
};

#endif
