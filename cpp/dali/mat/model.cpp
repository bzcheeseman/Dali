#include "model.h"

Model::Model() {
    _conf = std::make_shared<Conf>(default_conf());
}

Model::Model (Conf conf) {
    _conf = std::make_shared<Conf>(conf);
}

Conf Model::default_conf() {
    return Conf();
}

Conf& Model::c()  const {
    return *_conf;
}
