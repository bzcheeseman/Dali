#include "model.h"


std::shared_ptr<Conf> Model::default_conf() const {
    return std::make_shared<Conf>();
}

Conf& Model::conf()  const {
    if(!_conf)
        _conf = default_conf();
    return *_conf;
}
