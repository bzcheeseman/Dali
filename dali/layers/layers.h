#ifndef CORE_LAYERS_H
#define CORE_LAYERS_H

#include <initializer_list>

#include "dali/tensor/tensor.h"

class AbstractLayer {
    public:
        virtual std::vector<Tensor> parameters() const = 0;
};

class AbstractMultiInputLayer : public AbstractLayer {
    public:
        Tensor b;
        virtual Tensor activate(Tensor) const = 0;
        virtual Tensor activate(const std::vector<Tensor>&) const;
        virtual Tensor activate(Tensor, const std::vector<Tensor>&) const;
};

class Layer : public AbstractMultiInputLayer {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted):

        > y = A * x + b

    */
    void create_variables();
    public:
        Tensor W;
        int hidden_size;
        int input_size;

        Layer();
        Layer(int input_size, int hidden_size);
        Layer(const Layer&, bool, bool);

        Tensor activate(Tensor) const;
        Layer shallow_copy() const;

        virtual std::vector<Tensor> parameters() const;
};

class StackedInputLayer : public AbstractMultiInputLayer {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted), taking as inputs many different
    vectors of disparate sizes:

        > y = [A_1, ..., A_n] * [x_1, ..., x_n]^T + b

    */
    void create_variables();
    std::vector<int> input_sizes;
  public:
    mutable std::vector<Tensor> matrices;
    int hidden_size;

    virtual std::vector<Tensor> parameters() const;
    StackedInputLayer();
    StackedInputLayer(std::initializer_list<int> input_sizes, int output_size);
    StackedInputLayer(std::vector<int> input_sizes,           int output_size);
    StackedInputLayer(int input_size,                         int output_size);
    StackedInputLayer(const StackedInputLayer& other, bool copy_w, bool copy_dw);
    // getter
    const std::vector<int>& input_sizes() const;
    // settter
    void input_sizes(std::vector<int> new_sizes);

    Tensor activate(std::initializer_list<Tensor>) const;
    Tensor activate(const std::vector<Tensor>&) const;
    Tensor activate(Tensor) const;
    Tensor activate(Tensor, const std::vector<Tensor>&) const;

    StackedInputLayer shallow_copy() const;
};

class MultiLayerPerceptron {
    /* Multi Layer Perceptron.

       If there are N-1 layers an each maps for vector of hidden_sizes[i] to hidden_sizes[i+1].
       At the output of each layer function activations[i] is applied.
    */
    public:
        typedef std::function<Tensor(Tensor)> activation_t;

        std::vector<int> hidden_sizes;
        std::vector<activation_t> activations;

        std::vector<Layer> layers;

        MultiLayerPerceptron();

        MultiLayerPerceptron(std::vector<int> hidden_sizes, std::vector<activation_t> activations);

        MultiLayerPerceptron(const MultiLayerPerceptron& other, bool copy_w, bool copy_dw);

        MultiLayerPerceptron shallow_copy() const;

        Tensor activate(Tensor input) const;

        std::vector<Tensor> parameters() const;

        static Tensor identity(Tensor m);
};

class SecondOrderCombinator : public AbstractLayer {
    /* Implements the following expression

           y = (i1 * W1) * (i2 * W2) + b
           (the middle * is elementwise, others are matrix
            multiplies)

       The purpose is to capture only a few relevant terms from:

           y = i1 * W * i2' + B

        without having to evaluate big matrix equation.
    */
    public:
        int input1_size;
        int input2_size;
        int output_size;

        Tensor W1;
        Tensor W2;
        Tensor b;

        SecondOrderCombinator();

        SecondOrderCombinator(int input1_size, int input2_size, int output_size);

        SecondOrderCombinator(const SecondOrderCombinator& m, bool copy_w, bool copy_dw);

        virtual std::vector<Tensor> parameters() const;

        Tensor activate(Tensor i1, Tensor i2) const;
};

class RNN : public AbstractLayer {
    /*
    Combine the input of a hidden vector and an input vector
    into a single matrix product sum:

        > y = A * [x, h] + b

    */
    void create_variables();
    public:
        Tensor Wx;
        Tensor Wh;
        Tensor b;
        int hidden_size;
        int input_size;
        int output_size;
        virtual std::vector<Tensor> parameters() const;
        RNN();
        /**
        By default the RNN constructor sets the output size
        equal to the hidden size, creating the recurrence
        relation.
        **/
        RNN (int input_size, int hidden_size);
        /**
        An RNN can also output a different size output that
        its inputs and hiddens, e.g. in the case of a gate.
        **/
        RNN (int input_size, int hidden_size, int output_size);

        RNN (const RNN&, bool, bool);
        Tensor activate(Tensor input_vector, Tensor prev_hidden) const;

        RNN shallow_copy() const;
};

class DelayedRNN : public AbstractLayer {
    /*
        h' = A * [h,x] + b
        o  = B * [h,x] + d
    */
    public:
        RNN hidden_rnn;
        RNN output_rnn;
        virtual std::vector<Tensor> parameters() const;

        DelayedRNN (int input_size, int hidden_size, int output_size);
        DelayedRNN (const DelayedRNN&, bool, bool);

        Tensor initial_states() const;

        // output (next_hidden, output)
        std::tuple<Tensor,Tensor> activate(Tensor input_vector, Tensor prev_hidden) const;
        DelayedRNN shallow_copy() const;
};

#endif
