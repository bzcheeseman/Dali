#include "Batch.h"

template<typename R>
void Batch<R>::insert_example(const std::vector<std::string>& example,
                              const utils::Vocab& vocab,
                              size_t example_idx,
                              int offset) {
    auto example_length = example.size();
    ASSERT2(example_idx < data.dims(1),
            utils::MS() << "Inserting at position " << example_idx
                        << " which is beyond maximum number of examples in batch ("
                        << data.dims(1) << ")");
    ASSERT2(offset >= 0,
            "Offset cannot be negative");
    ASSERT2(example_length + offset <= data.dims(0),
            utils::MS() << "Insert example's length + offset = " << example_length
            << " + " << offset << " > max example length ("
            << data.dims(0) << ")");

    for (size_t j = 0; j < example_length; j++) {

        data.w(offset + j, example_idx) = (
            vocab.word2index.find(example[j]) != vocab.word2index.end()
            ? vocab.word2index.at(example[j])
            : vocab.unknown_word
        );
    }
}

template class Batch<float>;
template class Batch<double>;
