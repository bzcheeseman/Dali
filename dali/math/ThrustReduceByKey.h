#ifndef DALI_MATH_THRUST_REDUCE_BY_KEY_H
#define DALI_MATH_THRUST_REDUCE_BY_KEY_H
#include "dali/math/TensorOps.h"
#include "dali/math/memory_bank/MemoryBank.h"


#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/iterator/is_output_iterator.h>
#include <thrust/detail/type_traits/function_traits.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/iterator/zip_iterator.h>
#include <limits>
#include <thrust/detail/internal_functional.h>
#include <thrust/scan.h>
#include <thrust/detail/temporary_array.h>

#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/detail/allocator/allocator_traits.h>

namespace thrust {
namespace system {
namespace detail {
namespace generic {

/**
A version of Thrust's reduce_by_key which uses the MemoryBank instead
of allocating its own memory each time
**/
template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction,
         typename ValueType,
         typename FlagType>
__host__ __device__
thrust::pair<OutputIterator1,OutputIterator2>
dali_reduce_by_key(thrust::execution_policy<ExecutionPolicy> &exec,
                  InputIterator1 keys_first,
                  InputIterator1 keys_last,
                  InputIterator2 values_first,
                  OutputIterator1 keys_output,
                  OutputIterator2 values_output,
                  BinaryPredicate binary_pred,
                  BinaryFunction binary_op,
                  FlagType* head_flags,
                  ValueType* scanned_values,
                  FlagType* tail_flags,
                  FlagType* scanned_tail_flags) {
    typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;

    if (keys_first == keys_last)
        return thrust::make_pair(keys_output, values_output);

    // input size
    difference_type n = keys_last - keys_first;

    InputIterator2 values_last = values_first + n;

    // compute head flags
    auto head_flags_begin = thrust::detail::make_normal_iterator(
      thrust::device_pointer_cast(head_flags)
    );
    auto head_flags_end = head_flags_begin + n;

    thrust::transform(exec, keys_first, keys_last - 1, keys_first + 1, head_flags_begin + 1, thrust::detail::not2(binary_pred));
    head_flags_begin[0] = 1;

    // compute tail flags
    auto tail_flags_begin = thrust::detail::make_normal_iterator(
      thrust::device_pointer_cast(tail_flags)
    );
    auto tail_flags_end   = tail_flags_begin + n;

    thrust::transform(exec, keys_first, keys_last - 1, keys_first + 1, tail_flags_begin, thrust::detail::not2(binary_pred));
    tail_flags_begin[n-1] = 1;

    // scan the values by flag
    auto scanned_values_begin = thrust::detail::make_normal_iterator(
      thrust::device_pointer_cast(scanned_values)
    );
    auto scanned_values_end   = scanned_values_begin + n;

    auto scanned_tail_flags_begin = thrust::detail::make_normal_iterator(
      thrust::device_pointer_cast(scanned_tail_flags)
    );
    auto scanned_tail_flags_end   = scanned_tail_flags_begin + n;

    thrust::inclusive_scan
        (exec,
         thrust::make_zip_iterator(thrust::make_tuple(values_first,           head_flags_begin)),
         thrust::make_zip_iterator(thrust::make_tuple(values_last,            head_flags_end)),
         thrust::make_zip_iterator(thrust::make_tuple(scanned_values_begin,   scanned_tail_flags_begin)),
         detail::reduce_by_key_functor<ValueType, FlagType, BinaryFunction>(binary_op));

    thrust::exclusive_scan(exec, tail_flags_begin, tail_flags_end, scanned_tail_flags_begin, FlagType(0), thrust::plus<FlagType>());

    // number of unique keys
    FlagType N = scanned_tail_flags_begin[n - 1] + 1;

    // scatter the keys and accumulated values
    thrust::scatter_if(exec, keys_first,            keys_last,             scanned_tail_flags_begin, head_flags_begin, keys_output);
    thrust::scatter_if(exec, scanned_values_begin, scanned_values_end, scanned_tail_flags_begin, tail_flags_begin, values_output);

    return thrust::make_pair(keys_output + N, values_output + N);
} // end dali_reduce_by_key()

} // end generic
} // end detail
} // end system

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator1,
         typename OutputIterator2,
         typename BinaryPredicate,
         typename BinaryFunction>
  thrust::pair<OutputIterator1,OutputIterator2>
  dali_reduce_by_key(InputIterator1 keys_first,
                InputIterator1 keys_last,
                InputIterator2 values_first,
                OutputIterator1 keys_output,
                OutputIterator2 values_output,
                BinaryPredicate binary_pred,
                BinaryFunction binary_op) {
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<InputIterator1>::type  System1;
    typedef typename thrust::iterator_system<InputIterator2>::type  System2;
    typedef typename thrust::iterator_system<OutputIterator1>::type System3;
    typedef typename thrust::iterator_system<OutputIterator2>::type System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;


    typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
    // input size
    difference_type n = keys_last - keys_first;

    // the pseudocode for deducing the type of the temporary used below:
    //
    // if BinaryFunction is AdaptableBinaryFunction
    //   TemporaryType = AdaptableBinaryFunction::result_type
    // else if OutputIterator2 is a "pure" output iterator
    //   TemporaryType = InputIterator2::value_type
    // else
    //   TemporaryType = OutputIterator2::value_type
    //
    // XXX upon c++0x, TemporaryType needs to be:
    // result_of<BinaryFunction>::type
    typedef typename thrust::detail::eval_if<
      thrust::detail::has_result_type<BinaryFunction>::value,
      thrust::detail::result_type<BinaryFunction>,
      thrust::detail::eval_if<
        thrust::detail::is_output_iterator<OutputIterator2>::value,
        thrust::iterator_value<InputIterator2>,
        thrust::iterator_value<OutputIterator2>
      >
    >::type ValueType;
    typedef unsigned int FlagType;


    auto head_flags_temp         = temporary_array<int>(n,n);
    auto scanned_values_temp     = temporary_array<int>( n * sizeof(ValueType)/sizeof(int) + 1, n * sizeof(ValueType)/sizeof(int) + 1 );
    auto tail_flags_temp         = temporary_array<int>(n,n);
    auto scanned_tail_flags_temp = temporary_array<int>(n,n);

    return system::detail::generic::dali_reduce_by_key(
        select_system(system1,system2,system3,system4),
        keys_first,
        keys_last,
        values_first,
        keys_output,
        values_output,
        binary_pred,
        binary_op,
        (FlagType*) head_flags_temp.dptr_,
        (ValueType*) scanned_values_temp.dptr_,
        (FlagType*) tail_flags_temp.dptr_,
        (FlagType*) scanned_tail_flags_temp.dptr_
    );
}

} // end thrust

#endif
