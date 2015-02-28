#ifdef BLA_BLA_BLA
#ifndef BACKWARD_MAT_H
#define BACKWARD_MAT_H

#include "Mat.h"
#include <iostream>
#include <initializer_list>
#include <string>
#include <sstream>

template<typename T> class Backward {
        int ix;
        uint* indices;
        int num_indices;
        uint type;
        typedef Mat<T> mat;
        typedef std::shared_ptr<mat> shared_mat;
        void backward_rows_pluck();
        void backward_mul_add_mul_with_bias();
        void backward_rows_pluck(T);
        void backward_mul_add_mul_with_bias(T);
        public:
                std::vector<shared_mat> matrices;
                shared_mat out;
                Backward(shared_mat, shared_mat, uint);
                Backward(shared_mat, shared_mat, int, uint);
                Backward(shared_mat, shared_mat, index_std_vector&, uint);
                Backward(shared_mat, shared_mat, eigen_index_block, uint);
                Backward(std::initializer_list<shared_mat>, shared_mat, uint);
                Backward(const std::vector<shared_mat>&, shared_mat, uint);
                operator std::string() const;
                /**
                Operation Type
                --------------

                Returns the name of the mathematical operation that this step
                of backpropagation is responsible for computing (e.g. sigmoid,
                tanh, mul, eltmul, etc..)

                Outputs
                -------

                std::string name : name of operation
                **/
                std::string op_type () const;
                /**
                Operator()
                ----------

                Perform the backpropagation step taking the gradients from the
                matrix referenced by `out` and passing those using the adjoint
                method to the input `matrices`.

                See method definition for specific implementation details, and
                look at `Graph<T>::backward` for specific usage of
                `Backward<T>::operator()`.
                **/
                void operator ()();
                /**
                Operator(T)
                ----------

                Perform the backpropagation step taking the gradients from the
                matrix referenced by `out` and passing those using the adjoint
                method to the input `matrices`.

                See method definition for specific implementation details, and
                look at `Graph<T>::backward` for specific usage of
                `Backward<T>::operator()`.

                Inputs
                ------

                T clip_val : how much to elementwise clip the gradients during
                             back propagation.

                **/
                void operator ()(T);
};

template<typename T>
std::ostream& operator<<(std::ostream&, const Backward<T>&);

#endif
#endif
