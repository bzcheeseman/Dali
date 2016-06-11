////////////////////////////////////////////////////////////////////////////////
//                                   UTILS                                    //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<int dstdim>
    mshadow::Shape<dstdim> canonical_reshape(const std::vector<int>& src_shape,
                                             bool collapse_leading) {
        mshadow::Shape<dstdim> res;
        for (int i = 0; i < dstdim; i++) res[i] = 1;

        int srcdim = src_shape.size();

        if (collapse_leading) {
            for (int i = 0; i < srcdim; ++i) {
                res[std::max(dstdim - 1 - i, 0)] *= src_shape[srcdim - 1 - i];
            }
        } else {
            for (int i = 0; i < srcdim; ++i) {
                res[std::min(i, dstdim - 1)] *= src_shape[i];
            }
        }

        return res;
    }
}
////////////////////////////////////////////////////////////////////////////////
//                            TYPED ARRAY SHARED                              //
//                                   ---                                      //
//  Common to both CPU and GPU implementations of TypedArray below.           //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<typename MDevT, typename T>
    template<int dim>
    mshadow::Tensor<MDevT, dim, T> TypedArrayShared<MDevT,T>::mtensor(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::Tensor<MDevT, dim, T>(
            ptr_internal(access_mode),
            internal::canonical_reshape<dim>(array.shape(), collapse_leading)
        );
    }

    template<typename MDevT, typename T>
    template<int dim>
    mshadow::Tensor<MDevT, dim, T> TypedArrayShared<MDevT,T>::contiguous_d(memory::AM access_mode, bool collapse_leading) const {
        ASSERT2(this->array.contiguous_memory(),
            "contiguous_d can only be called on a TypedArray that has contiguous memory.");
        return mtensor<dim>(access_mode, collapse_leading);
    }

    template<typename MDevT, typename T>
    template<int dim>
    DaliWrapperExp<MDevT, dim, T> TypedArrayShared<MDevT,T>::d(memory::AM access_mode, bool collapse_leading) const {
        return MakeDaliWrapperExp(mtensor<dim>(access_mode, collapse_leading), array);
    }

////////////////////////////////////////////////////////////////////////////////
//                            TYPED SUBTENSOR SHARED                          //
//                                   ---                                      //
//  Common to both CPU and GPU implementations of TypedArray below.           //
////////////////////////////////////////////////////////////////////////////////


    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, dim, IndexT>,
                                   mshadow::Tensor<MDevT, dim + 1, T>,
                                   T,
                                   IndexT>
    TypedArraySubtensorShared<MDevT,T,IndexT>::contiguous_d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take_from_rows(
            indices.template contiguous_d<dim>(access_mode, collapse_leading),
            source.template contiguous_d<dim + 1>(access_mode, collapse_leading)
        );
    }

    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, dim, IndexT>,
                                   DaliWrapperExp<MDevT, dim+1, T>,
                                   T,
                                   IndexT>
    TypedArraySubtensorShared<MDevT,T,IndexT>::d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take_from_rows(
            indices.template d<dim>(access_mode, collapse_leading),
            source.template d<dim + 1>(access_mode, collapse_leading)
        );
    }
}  // namespace internal
