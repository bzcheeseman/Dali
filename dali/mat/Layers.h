#ifndef CORE_LAYERS_H
#define CORE_LAYERS_H

#include <initializer_list>

#include "dali/mat/Mat.h"
#include "dali/mat/MatOps.h"

template<typename R>
class AbstractLayer {
    public:
        virtual std::vector<Mat<R>> parameters() const = 0;
};

template<typename R>
class AbstractMultiInputLayer : public AbstractLayer<R> {
    public:
        Mat<R> b;
        virtual Mat<R> activate(Mat<R>) const = 0;
        virtual Mat<R> activate(const std::vector<Mat<R>>&) const;
        virtual Mat<R> activate(Mat<R>, const std::vector<Mat<R>>&) const;
};

template<typename R>
class Layer : public AbstractMultiInputLayer<R> {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted):

        > y = A * x + b

    */
    void create_variables();
    public:
        typedef R value_t;
        Mat<R> W;
        int hidden_size;
        int input_size;
        virtual std::vector<Mat<R>> parameters() const;

        Layer();
        Layer(int input_size, int hidden_size);
        Layer(const Layer&, bool, bool);

        Mat<R> activate(Mat<R>) const;
        Layer<R> shallow_copy() const;
};

template<typename R>
class StackedInputLayer : public AbstractMultiInputLayer<R> {
    /*
    Linear output layer of the form with affine offset bias
    vector b (broadcasted), taking as inputs many different
    vectors of disparate sizes:

        > y = [A_1, ..., A_n]^T * [x_1, ..., x_n] + b

    */
    void create_variables();
    std::vector<Mat<R>> zip_inputs_with_matrices_and_bias(const std::vector<Mat<R>>&) const;
    std::vector<Mat<R>> zip_inputs_with_matrices_and_bias(Mat<R>, const std::vector<Mat<R>>&) const;
    std::vector<int> _input_sizes;
    public:
        typedef R value_t;
        std::vector<Mat<R>> matrices;
        int hidden_size;

        virtual std::vector<Mat<R>> parameters() const;
        StackedInputLayer();
        StackedInputLayer(std::initializer_list<int> input_sizes, int output_size);
        StackedInputLayer(std::vector<int> input_sizes,           int output_size);
        StackedInputLayer(int input_size,                         int output_size);
        StackedInputLayer(const StackedInputLayer& other, bool copy_w, bool copy_dw);
        // getter
        const std::vector<int>& input_sizes() const;
        // settter
        void input_sizes(std::vector<int> new_sizes);

        Mat<R> activate(std::initializer_list<Mat<R>>) const;
        Mat<R> activate(const std::vector<Mat<R>>&) const;
        Mat<R> activate(Mat<R>) const;
        Mat<R> activate(Mat<R>, const std::vector<Mat<R>>&) const;

        StackedInputLayer<R> shallow_copy() const;
};

template<typename R>
class MultiLayerPerceptron {
    /* Multi Layer Perceptron.

       If there are N-1 layers an each maps for vector of hidden_sizes[i] to hidden_sizes[i+1].
       At the output of each layer function activations[i] is applied.
    */
    public:
        typedef std::function<Mat<R>(Mat<R>)> activation_t;

        std::vector<int> hidden_sizes;
        std::vector<activation_t> activations;

        std::vector<Layer<R>> layers;

        MultiLayerPerceptron();

        MultiLayerPerceptron(std::vector<int> hidden_sizes, std::vector<activation_t> activations);

        MultiLayerPerceptron(const MultiLayerPerceptron& other, bool copy_w, bool copy_dw);

        MultiLayerPerceptron shallow_copy();

        Mat<R> activate(Mat<R> input);

        std::vector<Mat<R>> parameters();

        static Mat<R> identity(Mat<R> m);
};

template<typename R>
class SecondOrderCombinator : public AbstractLayer<R> {
    /* Implements the following expression

           y = (W1 * i1) * (W2 * i2) + b
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

        Mat<R> W1;
        Mat<R> W2;
        Mat<R> b;

        SecondOrderCombinator();

        SecondOrderCombinator(int input1_size, int input2_size, int output_size);

        SecondOrderCombinator(const SecondOrderCombinator& m, bool copy_w, bool copy_dw);

        virtual std::vector<Mat<R>> parameters() const;

        Mat<R> activate(Mat<R> i1, Mat<R> i2) const;
};

template<typename R>
class RNN : public AbstractLayer<R> {
    /*
    Combine the input of a hidden vector and an input vector
    into a single matrix product sum:

        > y = A * [x, h] + b

    */
    void create_variables();
    public:
        typedef R value_t;
        Mat<R> Wx;
        Mat<R> Wh;
        Mat<R> b;
        int hidden_size;
        int input_size;
        int output_size;
        virtual std::vector<Mat<R>> parameters() const;
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
        Mat<R> activate(Mat<R> input_vector, Mat<R> prev_hidden) const;

        RNN<R> shallow_copy() const;
};

template<typename R>
class DelayedRNN : public AbstractLayer<R> {
    /*
        h' = A * [h,x] + b
        o  = B * [h,x] + d
    */
    public:
        RNN<R> hidden_rnn;
        RNN<R> output_rnn;
        virtual std::vector<Mat<R>> parameters() const;

        DelayedRNN (int input_size, int hidden_size, int output_size);
        DelayedRNN (const DelayedRNN&, bool, bool);

        Mat<R> initial_states() const;

        // output (next_hidden, output)
        std::tuple<Mat<R>,Mat<R>> activate(Mat<R> input_vector, Mat<R> prev_hidden) const;
        DelayedRNN<R> shallow_copy() const;
};

template<typename R>
class GatedInput : public RNN<R> {
    public:
        GatedInput (int, int);
        GatedInput (const GatedInput&, bool, bool);
        GatedInput<R> shallow_copy() const;
};

#endif
