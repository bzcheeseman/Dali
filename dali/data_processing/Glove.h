#ifndef DALI_CORE_DATA_PROCESSING_GLOVE_H
#define DALI_CORE_DATA_PROCESSING_GLOVE_H

#include <string>
#include <vector>
#include "dali/tensor/Mat.h"
#include "dali/utils/core_utils.h"
#include "dali/utils/assert2.h"
#include "dali/utils/ParseUtils.h"

namespace glove {
    /**
    Loads a text file with Glove word vectors. Returns a tuple
    with a Matrix containing the embeddings (one row per word)
    and a Vocab object containing a mapping from index to word
    and from word to index. Matrix is dynamically resized.
    **/

    template<typename T>
    std::tuple<Mat<T>, utils::Vocab> load(std::string fname, int threshold = -1);

    template<typename T>
    void load(std::string fname, Mat<T>* mat, utils::Vocab* vocab, int threshold = -1);

    // Loads relevant embeddings from glove vector fname,
    // stores the in matrix target (resized if necessary)
    // Reads at most threshold words from glove (if not -1).
    // Returns number of matched words.
    template<typename T>
    int load_relevant_vectors(std::string fname,
                              Mat<T>* target,
                              const utils::Vocab& vocab,
                              int threshold=-1);
}

#endif
