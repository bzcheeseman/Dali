#include "dali/data_processing/Glove.h"

using std::string;
using utils::Vocab;
using std::make_tuple;
using std::vector;
using utils::from_string;

namespace glove {

    template<typename T>
    std::tuple<Mat<T>, Vocab> load(string fname) {
        if (!utils::file_exists(fname)) {
            throw std::runtime_error("Cannot open file with glove vectors.");
        }

        std::fstream fp(fname);
        std::string line;

        int observed_size = 0;
        int capacity      = 100;
        int vocab_size    = 0;
        auto pair = make_tuple<Mat<T>, Vocab>(Mat<T>(capacity, observed_size, false), Vocab());
        vector<string> vocabulary;
        std::string item;
        // use mat for assigning elements
        auto& mat = std::get<0>(pair).w();
        // notify Mat of resizes using underlying mat
        auto& underlying_mat = std::get<0>(pair);


        while (std::getline(fp, line)) {
            bool found_name = false;
            int i = 0; // count how many numbers are in this row
            std::stringstream tokenizer(line);
            while (std::getline(tokenizer, item, ' ')) {
                if (!found_name) {
                    vocabulary.emplace_back(item);
                    vocab_size+=1;
                    if (vocab_size > capacity) {
                        // increase matrix by 10%
                        underlying_mat.resize(capacity * 1.1, observed_size);
                        capacity = capacity * 1.1;
                    }
                    found_name = true;
                } else {
                    if (!item.empty()) {
                        if (observed_size == 0 && i+1 > mat.cols()) {
                            underlying_mat.resize(capacity, std::max(100, (int) (mat.cols() * 1.5)));
                            mat(vocab_size-1, i) = from_string<T>(item);
                        } else {
                            mat(vocab_size-1, i) = from_string<T>(item);
                        }
                        i += 1;
                    }
                }
            }
            if (observed_size == 0) {
                // actualize size of matrix
                observed_size = i;
                underlying_mat.resize(capacity, observed_size);
            }
            if (i != observed_size) {
                std::runtime_error("Vectors in Glove file are of different sizes.");
            }
        }
        // now final update is made to matrix
        underlying_mat.resize(vocab_size  + 1, observed_size);
        mat.row(vocab_size).fill(0.0);
        std::get<1>(pair) = Vocab(vocabulary);
        return pair;
    }

    template std::tuple<Mat<float>, Vocab> load(string);
    template std::tuple<Mat<double>, Vocab> load(string);
}
