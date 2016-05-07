////////////////////////////////////////////////////////////////////////////////
//                                   UTILS                                    //
////////////////////////////////////////////////////////////////////////////////

namespace internal {
    template<int dstdim>
    mshadow::Shape<dstdim> canonical_reshape(const std::vector<int>& src_shape) {
        mshadow::Shape<dstdim> res;
        for (int i = 0; i < dstdim; i++) res[i] = 1;

        int residual_shape = 1;
        for (int i = 0; i < src_shape.size(); ++i) {
            residual_shape *= src_shape[i];
            int dst_index = i - src_shape.size() + dstdim;
            if (dst_index >= 0) {
                res[dst_index] = residual_shape;
                residual_shape = 1;
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
    mshadow::Tensor<MDevT, dim, T> TypedArrayShared<MDevT,T>::mtensor(memory::AM access_mode) const {
        return mshadow::Tensor<MDevT, dim, T>(
            ptr_internal(access_mode),
            internal::canonical_reshape<dim>(array.shape())
        );
    }

    template<typename MDevT, typename T>
    template<int dim>
    mshadow::Tensor<MDevT, dim, T> TypedArrayShared<MDevT,T>::contiguous_d(memory::AM access_mode) const {
        ASSERT2(this->array.contiguous_memory(),
                "This function is only supported for contiguous_memory");
        return mtensor<dim>(access_mode);
    }

    template<typename MDevT, typename T>
    template<int dim>
    DaliWrapperExp<MDevT, dim, T> TypedArrayShared<MDevT,T>::d(memory::AM access_mode) const {
        return MakeDaliWrapperExp(mtensor<dim>(access_mode), array);
    }

}  // namespace internal
