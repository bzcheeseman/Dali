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
        data.w(offset + j, example_idx) = vocab[example[j]];
    }
}

template<typename R>
int Batch<R>::example_length(const int& idx) const {
    ASSERT2(idx < code_lengths.size(),
        utils::MS() << "Asking for length of an example outside of the batch ("
        << idx << "), batch size = "
        <<  code_lengths.size() <<".");
    return code_lengths[idx];
}

template<typename R>
size_t Batch<R>::size() const {
    return data.dims(1);
}

template<typename R>
size_t Batch<R>::max_length() const {
    return data.dims(0);
}

template class Batch<float>;
template class Batch<double>;
