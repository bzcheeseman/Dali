#include "core/Index.h"

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

    Index::Index(index_std_vector& vec) {
        w = make_shared<VectorIndex>(vec);
    }
    Index::Index(std::initializer_list<ind_t> vec) {
        w = make_shared<OwnershipVectorIndex>(vec);
    }
    Index::Index(std::shared_ptr<OwnershipVectorIndex> internal) {
        w = internal;
    }
    Index::Index(eigen_index_vector& vec) {
        w = make_shared<EigenIndexVectorIndex>(vec);
    }
    Index::Index(eigen_index_block_scalar vec) {
        w = make_shared<EigenIndexBlockIndex>(vec);
    }
    Index::Index(eigen_index_block vec) {
        w = make_shared<EigenIndexBlockIndex>(vec);
    }
    Index::Index(eigen_segment vec) {
        w = make_shared<EigenIndexBlockIndex>(vec);
    }
    Index::Index(eigen_segment_scalar vec) {
        w = make_shared<EigenIndexBlockIndex>(vec);
    }
    Index::Index(eigen_index_block_scalar_from_row vec) {
        w = make_shared<EigenIndexBlockIndex>(vec);
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

    const ind_t* EigenIndexVectorIndex::data() const {
        return w.data();
    }

    size_t EigenIndexVectorIndex::size() const {
        return w.rows();
    }

    ind_t& EigenIndexVectorIndex::operator[](std::size_t idx) {
        return w[idx];
    }

    ind_t EigenIndexVectorIndex::operator[](std::size_t idx) const {
        return w[idx];
    }

    EigenIndexVectorIndex::EigenIndexVectorIndex(eigen_index_vector& vec) : w(vec) {
    }


    const ind_t* EigenIndexBlockIndex::data() const {
        return w.data();
    }

    size_t EigenIndexBlockIndex::size() const {
        return w.rows();
    }

    ind_t& EigenIndexBlockIndex::operator[](std::size_t idx) {
        return w[idx];
    }

    ind_t EigenIndexBlockIndex::operator[](std::size_t idx) const {
        return w[idx];
    }

    EigenIndexBlockIndex::EigenIndexBlockIndex(eigen_index_block_scalar vec)          : w(vec) {}
    EigenIndexBlockIndex::EigenIndexBlockIndex(eigen_index_block vec)                 : w(vec) {}
    EigenIndexBlockIndex::EigenIndexBlockIndex(eigen_segment vec)                     : w(vec) {}
    EigenIndexBlockIndex::EigenIndexBlockIndex(eigen_segment_scalar vec)              : w(vec) {}
    EigenIndexBlockIndex::EigenIndexBlockIndex(eigen_index_block_scalar_from_row vec) : w(vec) {}

}
