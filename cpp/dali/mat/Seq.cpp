#include "Seq.h"

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

/* begin, end */

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

/* rbegin, rend */

template<typename T>
typename Seq<T>::RSeqIter Seq<T>::rbegin() {
    return seq.rbegin();
}

template<typename T>
typename Seq<T>::RSeqIter Seq<T>::rend() {
    return seq.rend();
}

template<typename T>
typename Seq<T>::RSeqConstIter Seq<T>::rbegin() const {
    return seq.rbegin();
}

template<typename T>
typename Seq<T>::RSeqConstIter Seq<T>::rend() const {
    return seq.rend();
}

/* cbegin, cend */


template<typename T>
typename Seq<T>::SeqConstIter Seq<T>::cbegin() const {
    return seq.cbegin();
}

template<typename T>
typename Seq<T>::SeqConstIter Seq<T>::cend() const {
    return seq.cend();
}


template<typename T>
void Seq<T>::insert(SeqIter where, SeqConstIter begin, SeqConstIter end ) {
    seq.insert(where, begin, end);
}

template<typename T>
void Seq<T>::insert(SeqIter where, RSeqConstIter begin, RSeqConstIter end ) {
    seq.insert(where, begin, end);
}


template class Seq<Mat<float>>;
template class Seq<Mat<double>>;
