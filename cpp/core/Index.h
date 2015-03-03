#ifndef INDEX_MAT_H
#define INDEX_MAT_H

#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <ostream>
#include <initializer_list>

typedef Eigen::MatrixBase<Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic> >::ColXpr eigen_index_block;
typedef Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> eigen_index_vector;
typedef Eigen::MatrixWrapper<
        Eigen::CwiseUnaryOp< Eigen::internal::scalar_add_op<unsigned int>, Eigen::ArrayWrapper<eigen_index_block> const > const > eigen_index_block_scalar;

typedef Eigen::MatrixWrapper<
        Eigen::CwiseUnaryOp< Eigen::internal::scalar_add_op<unsigned int>, Eigen::ArrayWrapper<Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>> const > const > eigen_index_block_scalar_from_row;

typedef std::vector<uint> index_std_vector;
typedef std::shared_ptr<eigen_index_vector> shared_eigen_index_vector;

typedef Eigen::VectorBlock<eigen_index_vector> eigen_segment;
typedef Eigen::VectorBlock<Eigen::Block<Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>, 1, Eigen::Dynamic>> eigen_segment_scalar;

namespace Indexing {

    typedef uint ind_t;

    class IndexInternal {
        public:
            virtual const ind_t* data() const = 0;
            virtual size_t size() const = 0;
            virtual ind_t& operator[](std::size_t idx) = 0;
            virtual ind_t operator[](std::size_t idx) const = 0;
    };

    class OwnershipVectorIndex : public IndexInternal {
        public:
            index_std_vector w;
            OwnershipVectorIndex(std::initializer_list<ind_t> vec);
            virtual const ind_t* data() const;
            virtual size_t size() const;
            virtual ind_t& operator[](std::size_t idx);
            virtual ind_t operator[](std::size_t idx) const;
    };

    class Index {
        private:
            std::shared_ptr<IndexInternal> w;
        public:
            const ind_t* data() const;
            size_t size() const;
            ind_t& operator[](std::size_t idx);
            ind_t operator[](std::size_t idx) const;
            Index(index_std_vector&);
            Index(std::initializer_list<ind_t>);
            Index(std::shared_ptr<OwnershipVectorIndex>);
            Index(eigen_index_vector&);
            Index(const Index&);
            Index(eigen_segment);
            Index(eigen_segment_scalar);
            Index(eigen_index_block_scalar_from_row);
            /**
            Index constructor from Eigen views and blocks
            ---------------------------------------------

            Cause copies, wrap around the column and rows.
            **/
            Index(eigen_index_block_scalar);
            Index(eigen_index_block);
            static Index arange(uint start, uint end_non_inclusive);
    };

    class EigenIndexBlockIndex : public IndexInternal {
        public:
            eigen_index_vector w;
            EigenIndexBlockIndex(eigen_index_block_scalar vec);
            EigenIndexBlockIndex(eigen_index_block vec);
            EigenIndexBlockIndex(eigen_segment vec);
            EigenIndexBlockIndex(eigen_segment_scalar vec);
            EigenIndexBlockIndex(eigen_index_block_scalar_from_row vec);

            virtual const ind_t* data() const;
            virtual size_t size() const;
            virtual ind_t& operator[](std::size_t idx);
            virtual ind_t operator[](std::size_t idx) const;
    };

    class EigenIndexVectorIndex : public IndexInternal {
        public:
            eigen_index_vector& w;
            EigenIndexVectorIndex(eigen_index_vector& vec);
            virtual const ind_t* data() const;
            virtual size_t size() const;
            virtual ind_t& operator[](std::size_t idx);
            virtual ind_t operator[](std::size_t idx) const;
    };

    class VectorIndex : public IndexInternal {
        public:
            index_std_vector& w;
            VectorIndex(index_std_vector& vec);
            virtual const ind_t* data() const;
            virtual size_t size() const;
            virtual ind_t& operator[](std::size_t idx);
            virtual ind_t operator[](std::size_t idx) const;
    };
}

std::ostream& operator<<(std::ostream&, const Indexing::Index&);

#endif
