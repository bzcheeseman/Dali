#ifndef NER_DALI_H
#define NER_DALI_H

#include <string>
#include <vector>
#include "dali/utils.h"

namespace NER {
    typedef std::pair< std::vector<std::string>, std::vector<std::string> > example_t;
    std::vector<example_t> load(std::string path, std::string start_symbol = "-DOCSTART-");
}


#endif
