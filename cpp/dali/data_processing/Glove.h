#ifndef DALI_CORE_DATA_PROCESSING_GLOVE_H
#define DALI_CORE_DATA_PROCESSING_GLOVE_H

#include <string>
#include <vector>
#include "dali/mat/Mat.h"
#include "dali/utils/core_utils.h"

namespace glove {
    /**
    Loads a text file with Glove word vectors. Returns a tuple
    with a Matrix containing the embeddings (one row per word)
    and a Vocab object containing a mapping from index to word
    and from word to index. Matrix is dynamically resized.
    **/
    template<typename T>
    std::tuple<Mat<T>, utils::Vocab> load(std::string fname);
}

#endif
