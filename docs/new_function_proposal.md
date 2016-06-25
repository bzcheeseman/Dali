# New Function API proposal

## Requirements

1. Single abstraction for Array, ArrayGather, LazyExp, (TypedArray), which exposes some simple Array like API. Note: ArrayGather should be returned by both `Array::operator[]` and `op::take` / `op::gather`.

*Motivation: Ease of use*

2. Templates for lazy exp should not be overly general.

*Motivation: Ease of debugging*

3. All the array "preprocessing", i.e. reshaping, type casting and broadcasting, should happen before computation.

*Motivation: async*

4. API should be able to clearly express the conditions which it can and cannot handle, i.e. which shapes / broadcasts / types / devices are supported. Those check should happen at the preprocessing stage.

*Motivation: ease of use, getting exception in the line they were written, not when async executes them.*

5. Different return types. Can we get away with just `Array<bool>` and no other return types?

*Motivation: isnan and more.*

6. Returning multiple arrays. What does it mean for async?

*Motivation: concat?.*

7. Support for n-axis reduction.

*Motivation: comes up for example in conv bias.*

8. All above requirements also apply to lazy expressions.

9. All the code related to particular function should ideally reside in one file, or as few files as possible.
