////////////////////////////////////////////////////////////////////////////////
//                            TYPED ARRAY SHARED                              //
//                                   ---                                      //
//  Common to both CPU and GPU implementations of TypedArray below.           //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeFromRowsExp<mshadow::Tensor<MDevT, dim - 1, IndexT>, mshadow::Tensor<MDevT, dim, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T,IndexT>::contiguous_d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take_from_rows(
            indices.template contiguous_d<dim - 1>(access_mode, collapse_leading),
            source.template contiguous_d<dim>(access_mode, collapse_leading)
        );
    }

    template<typename MDevT, typename T, typename IndexT>
    template<int dim>
    mshadow::expr::TakeFromRowsExp<DaliWrapperExp<MDevT, dim - 1, IndexT>, DaliWrapperExp<MDevT, dim, T>, T, IndexT> TypedArraySubtensorShared<MDevT,T,IndexT>::d(memory::AM access_mode, bool collapse_leading) const {
        return mshadow::expr::take_from_rows(
            indices.template d<dim - 1>(access_mode, collapse_leading),
            source.template d<dim>(access_mode, collapse_leading)
        );
    }
}  // namespace internal
