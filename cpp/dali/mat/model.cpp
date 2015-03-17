#include "model.h"


Conf Model::default_conf() const {
    return Conf();
}

Conf& Model::conf()  const {
    if(!_conf)
        _conf = std::make_shared<Conf>(default_conf());
    return *_conf;
}
