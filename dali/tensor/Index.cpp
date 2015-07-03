#include "Index.h"

using std::make_shared;
using std::vector;

std::ostream& operator<<(std::ostream& strm, const Indexing::Index& a) {
    if (a.size() < 10) {
        strm << "<Index [";
        for (size_t i = 0; i < a.size() - 1; ++i) {
            strm << a[i] << ", ";
        }
        if (a.size() > 0) {
            strm << a[a.size() - 1];
        }
        strm << "]>";
    } else {
        strm << "<Index [";
        for (size_t i = 0; i < 5; ++i) {
            strm << a[i] << ", ";
        }
        strm << "...,";
        for (size_t i = a.size() - 5; i < a.size() - 1; ++i) {
            strm << a[i] << ", ";
        }
        strm << a[a.size() - 1] << "]>";
    }
    return strm;
}

namespace Indexing {

    // IndexInteranl is purely virtual

    // Index wraps around a shared point to Index Internal

    const ind_t* Index::data() const {
        return w->data();
    }

    ind_t* Index::data() {
        return w->data();
    }

    size_t Index::size() const {
        return w->size();
    }

    ind_t& Index::operator[](std::size_t idx) {
        return (*w)[idx];
    }

    ind_t Index::operator[](std::size_t idx) const {
        return (*w)[idx];
    }

    Index::Index(const Index& original) : w(original.w) {}

    Index::Index(index_std_vector* vec) {
        w = make_shared<VectorIndex>(*vec);
    }
    Index::Index(std::initializer_list<ind_t> vec) {
        w = make_shared<OwnershipVectorIndex>(vec);
    }
    Index::Index(std::shared_ptr<OwnershipVectorIndex> internal) {
        w = internal;
    }
    Index Index::arange(uint start, uint end_non_inclusive) {
        assert(start < end_non_inclusive);
        auto internal = make_shared<OwnershipVectorIndex>(std::initializer_list<uint>({}));
        internal->w.reserve(end_non_inclusive - start);
        for (uint s = start; s < end_non_inclusive; ++s) {
            internal->w.emplace_back(s);
        }
        return Index(internal);
    }

    // Finally different datatypes get the royal treatment:

    const ind_t* VectorIndex::data() const {
        return w.data();
    }

    ind_t* VectorIndex::data() {
        return w.data();
    }

    size_t VectorIndex::size() const {
        return w.size();
    }

    ind_t& VectorIndex::operator[](std::size_t idx) {
        return w[idx];
    }

    ind_t VectorIndex::operator[](std::size_t idx) const {
        return w[idx];
    }

    VectorIndex::VectorIndex(index_std_vector& vec) : w(vec) {
    }

    const ind_t* OwnershipVectorIndex::data() const {
        return w.data();
    }

    ind_t* OwnershipVectorIndex::data() {
        return w.data();
    }

    size_t OwnershipVectorIndex::size() const {
        return w.size();
    }

    ind_t& OwnershipVectorIndex::operator[](std::size_t idx) {
        return w[idx];
    }

    ind_t OwnershipVectorIndex::operator[](std::size_t idx) const {
        return w[idx];
    }

    OwnershipVectorIndex::OwnershipVectorIndex(std::initializer_list<ind_t> vec) : w(vec) {
    }

    Index::iterator Index::begin() {
        return iterator(w->data());
    }

    Index::iterator Index::end() {
        return iterator(w->data() + size());
    }

    Index::const_iterator Index::begin() const {
        return const_iterator(w->data());
    }

    Index::const_iterator Index::end() const {
        return const_iterator(w->data() + size());
    }

    Index::iterator::iterator(pointer ptr) : ptr_(ptr) { }
    typename Index::iterator::self_type Index::iterator::operator++() { self_type i = *this; ptr_++; return i; }
    typename Index::iterator::self_type Index::iterator::operator++(int junk) { ptr_++; return *this; }
    typename Index::iterator::reference Index::iterator::operator*() { return *ptr_; }
    typename Index::iterator::pointer   Index::iterator::operator->() { return ptr_; }
    bool Index::iterator::operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
    bool Index::iterator::operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }

    Index::const_iterator::const_iterator(const pointer ptr) : ptr_(ptr) { }
    typename Index::const_iterator::self_type       Index::const_iterator::operator++() { self_type i = *this; ptr_++; return i; }
    typename Index::const_iterator::self_type       Index::const_iterator::operator++(int junk) { ptr_++; return *this; }
    typename Index::const_iterator::reference Index::const_iterator::operator*() { return *ptr_; }
    typename Index::const_iterator::pointer   Index::const_iterator::operator->() { return ptr_; }
    bool Index::const_iterator::operator==(const self_type& rhs) {
        return ptr_ == rhs.ptr_;
    }
    bool Index::const_iterator::operator!=(const self_type& rhs) {
        return ptr_ != rhs.ptr_;
    }
}
