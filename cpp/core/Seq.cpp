#include "core/Seq.h"

using std::vector;

template<typename T>
void Seq<T>::push_back(T el) {
    seq.push_back(el);
}

template<typename T>
std::size_t  Seq<T>::size() const {
    return seq.size();
}

template<typename T>
T& Seq<T>::operator[](std::size_t idx) {
    return seq[idx];
};

template<typename T>
T Seq<T>::operator[](std::size_t idx) const {
    return seq[idx];
}

template<typename T>
typename Seq<T>::SeqIter Seq<T>::begin() {
    return seq.begin();
}

template<typename T>
typename Seq<T>::SeqIter Seq<T>::end() {
    return seq.end();
}

template<typename T>
typename Seq<T>::SeqConstIter Seq<T>::begin() const {
    return seq.begin();
}

template<typename T>
typename Seq<T>::SeqConstIter Seq<T>::end() const {
    return seq.end();
}

template<typename T>
typename Seq<T>::SeqConstIter Seq<T>::cbegin() const {
    return seq.cbegin();
}

template<typename T>
typename Seq<T>::SeqConstIter Seq<T>::cend() const {
    return seq.cend();
}


template class Seq<std::shared_ptr<Mat<float>>>;
template class Seq<std::shared_ptr<Mat<double>>>;
