#include "dali/layers/Layers.h"

using std::vector;

template<typename R>
Mat<R> AbstractMultiInputLayer<R>::activate(const vector<Mat<R>>& inputs) const {
    assert(inputs.size() > 0);
    return activate(inputs[0]);
};

template<typename R>
Mat<R> AbstractMultiInputLayer<R>::activate(Mat<R> first_input, const vector<Mat<R>>& inputs) const {
    if (inputs.size() > 0) {
        return activate(inputs.back());
    } else {
        return activate(first_input);
    }
};

template<typename R>
void Layer<R>::create_variables() {
    W = Mat<R>(input_size, hidden_size, weights<R>::uniform(2.0 / sqrt(input_size)));
    this->b = Mat<R>(1, hidden_size, weights<R>::uniform(2.0 / sqrt(input_size)));
}

template<typename R>
Layer<R>::Layer() {
}

template<typename R>
Layer<R>::Layer (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
    create_variables();
}

template<typename R>
Mat<R> Layer<R>::activate(Mat<R> input_vector) const {
    return MatOps<R>::mul_with_bias(W, input_vector, this->b);
}

template<typename R>
Layer<R>::Layer (const Layer<R>& layer, bool copy_w, bool copy_dw) : hidden_size(layer.hidden_size), input_size(layer.input_size) {
    W = Mat<R>(layer.W, copy_w, copy_dw);
    this->b = Mat<R>(layer.b, copy_w, copy_dw);
}

template<typename R>
Layer<R> Layer<R>::shallow_copy() const {
    return Layer<R>(*this, false, true);
}

template<typename R>
vector<Mat<R>> Layer<R>::parameters() const{
    return vector<Mat<R>>({W, this->b});
}

// StackedInputLayer:
template<typename R>
void StackedInputLayer<R>::create_variables() {
    int total_input_size = 0;
    for (auto& input_size : _input_sizes) total_input_size += input_size;
    matrices.reserve(_input_sizes.size());
    auto U = weights<R>::uniform(2.0 / sqrt(total_input_size));
    for (auto& input_size : _input_sizes) {
        matrices.emplace_back(input_size, hidden_size, U);
    }
    this->b = Mat<R>(1, hidden_size, U);
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer() {
}

template<typename R>
const vector<int>& StackedInputLayer<R>::input_sizes() const {
    return _input_sizes;
}

template<typename R>
void StackedInputLayer<R>::input_sizes(vector<int> new_sizes) {
    ASSERT2(new_sizes.size() > 0, "StackedInputLayer must have at least one input (0 provided)");
    // optimistic exit
    if (new_sizes == _input_sizes)
        return;

    // new random numbers
    int total_input_size = 0;
    for (auto& input_size : new_sizes) total_input_size += input_size;
    auto U = weights<R>::uniform(2.0 / sqrt(total_input_size));

    // construct matrices
    matrices = vector<Mat<R>>();
    for (auto& input_size : new_sizes) {
        matrices.emplace_back(input_size, hidden_size, U);
    }

    // save new size
    _input_sizes = new_sizes;
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (vector<int> input_sizes,
                                         int _hidden_size) :
        hidden_size(_hidden_size),
        _input_sizes(input_sizes) {
    create_variables();
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (int input_size,
                                         int output_size) :
        hidden_size(output_size),
        _input_sizes({input_size}) {
    create_variables();
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (std::initializer_list<int> input_sizes,
                                         int _hidden_size) :
        hidden_size(_hidden_size),
        _input_sizes(input_sizes) {
    create_variables();
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(std::initializer_list<Mat<R>> inputs) const {
    vector<Mat<R>> temp(inputs);
    return activate(temp);
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(const vector<Mat<R>>& inputs) const {
    return MatOps<R>::mul_add_mul_with_bias(matrices, inputs, this->b);
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(
        Mat<R> input_vector) const {
    if (matrices.size() == 1) {
        return MatOps<R>::mul_with_bias(matrices.front(), input_vector, this->b);
    } else {
        throw std::runtime_error("Error: Stacked Input Layer parametrized with more than 1 inputs only received 1 input vector.");
    }
}

template<typename R>
Mat<R> StackedInputLayer<R>::activate(
        Mat<R> input,
        const vector<Mat<R>>& inputs) const {
    DEBUG_ASSERT_MAT_NOT_NAN(input)

    vector<Mat<R>> zipped;
    zipped.emplace_back(input);
    for (auto& an_input : inputs )
        zipped.emplace_back(an_input);

    auto out = MatOps<R>::mul_add_mul_with_bias(matrices, zipped, this->b);

    DEBUG_ASSERT_MAT_NOT_NAN(out)

    return out;
}

template<typename R>
StackedInputLayer<R>::StackedInputLayer (const StackedInputLayer<R>& layer, bool copy_w, bool copy_dw) : hidden_size(layer.hidden_size), _input_sizes(layer.input_sizes()) {
    matrices.reserve(layer.matrices.size());
    for (auto& matrix : layer.matrices)
        matrices.emplace_back(matrix, copy_w, copy_dw);
    this->b = Mat<R>(layer.b, copy_w, copy_dw);
}

template<typename R>
StackedInputLayer<R> StackedInputLayer<R>::shallow_copy() const {
    return StackedInputLayer<R>(*this, false, true);
}

template<typename R>
std::vector<Mat<R>> StackedInputLayer<R>::parameters() const{
    auto params = vector<Mat<R>>(matrices);
    params.emplace_back(this->b);
    return params;
}

/* MultiLayerPerceptron */
template<typename R>
MultiLayerPerceptron<R>::MultiLayerPerceptron() {
}

template<typename R>
MultiLayerPerceptron<R>::MultiLayerPerceptron(vector<int> hidden_sizes, vector<activation_t> activations) :
        hidden_sizes(hidden_sizes),
        activations(activations) {
    ASSERT2(activations.size() == hidden_sizes.size() - 1,
            "Wrong number of activations for MultiLayerPerceptron");

    for (int lidx = 0; lidx < hidden_sizes.size() - 1; ++lidx) {
        layers.push_back(Layer<R>(hidden_sizes[lidx], hidden_sizes[lidx + 1]));
    }
}

template<typename R>
MultiLayerPerceptron<R>::MultiLayerPerceptron(const MultiLayerPerceptron& other, bool copy_w, bool copy_dw) :
        hidden_sizes(other.hidden_sizes),
        activations(other.activations) {
    for (auto& other_layer: other.layers) {
        layers.emplace_back(other_layer, copy_w, copy_dw);
    }
}

template<typename R>
MultiLayerPerceptron<R> MultiLayerPerceptron<R>::shallow_copy() const {
    return MultiLayerPerceptron(*this, false, true);
}

template<typename R>
Mat<R> MultiLayerPerceptron<R>::activate(Mat<R> input) const {
    Mat<R> last_output = input;
    for (int i = 0; i < hidden_sizes.size() - 1; ++i)
        last_output = activations[i](layers[i].activate(last_output));

    return last_output;
}

template<typename R>
vector<Mat<R>> MultiLayerPerceptron<R>::parameters() const {
    vector<Mat<R>> params;
    for (auto& layer: layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}
template<typename R>
Mat<R> MultiLayerPerceptron<R>::identity(Mat<R> m) { return m; }


template class MultiLayerPerceptron<float>;
template class MultiLayerPerceptron<double>;


template<typename R>
void RNN<R>::create_variables() {
    Wx = Mat<R>(input_size,  output_size,  weights<R>::uniform(2.0 / sqrt(input_size)));
    Wh = Mat<R>(hidden_size, output_size,  weights<R>::uniform(2.0 / sqrt(hidden_size)));
    b  = Mat<R>(1, output_size, weights<R>::uniform(2.0 / sqrt(hidden_size)));
}

template<typename R>
vector<Mat<R>> RNN<R>::parameters() const {
    return vector<Mat<R>>({Wx, Wh, b});
}

/* DelayedRNN */

template<typename R>
DelayedRNN<R>::DelayedRNN(int input_size, int hidden_size, int output_size) :
        hidden_rnn(input_size, hidden_size),
        output_rnn(input_size, hidden_size, output_size) {
}

template<typename R>
DelayedRNN<R>::DelayedRNN (const DelayedRNN<R>& rnn, bool copy_w, bool copy_dw) :
        hidden_rnn(rnn.hidden_rnn, copy_w, copy_dw),
        output_rnn(rnn.output_rnn, copy_w, copy_dw) {
}

template<typename R>
vector<Mat<R>> DelayedRNN<R>::parameters() const {
    vector<Mat<R>> ret;
    for (auto& model: {hidden_rnn, output_rnn}) {
        auto params = model.parameters();
        ret.insert(ret.end(), params.begin(), params.end());
    }
    return ret;
}

template<typename R>
Mat<R> DelayedRNN<R>::initial_states() const {
    return Mat<R>(hidden_rnn.hidden_size, 1, weights<R>::uniform(2.0 / sqrt(hidden_rnn.hidden_size)));
}

template<typename R>
std::tuple<Mat<R>,Mat<R>> DelayedRNN<R>::activate(
        Mat<R> input_vector,
        Mat<R> prev_hidden) const {

    return std::make_tuple(
        hidden_rnn.activate(input_vector, prev_hidden),
        output_rnn.activate(input_vector, prev_hidden)
    );
}

template<typename R>
DelayedRNN<R> DelayedRNN<R>::shallow_copy() const {
    return DelayedRNN<R>(*this, false, true);
}

template class DelayedRNN<float>;
template class DelayedRNN<double>;

/* StackedInputLayer */
/* SecondOrderCombinator */
template<typename R>
SecondOrderCombinator<R>::SecondOrderCombinator() {
}

template<typename R>
SecondOrderCombinator<R>::SecondOrderCombinator(int input1_size, int input2_size, int output_size) :
        input1_size(input1_size), input2_size(input2_size), output_size(output_size) {
    W1 = Mat<R>(input1_size, output_size,  weights<R>::uniform(2.0/sqrt(input1_size)));
    W2 = Mat<R>(input2_size, output_size,  weights<R>::uniform(2.0/sqrt(input2_size)));
    b =  Mat<R>(1, output_size,            weights<R>::uniform(2.0/sqrt(input1_size)));
}
template<typename R>
SecondOrderCombinator<R>::SecondOrderCombinator(const SecondOrderCombinator& m,
                                                bool copy_w,
                                                bool copy_dw) :
        input1_size(m.input1_size),
        input2_size(m.input2_size),
        output_size(m.output_size),
        W1(m.W1, copy_w, copy_dw),
        W2(m.W2, copy_w, copy_dw),
        b(m.b, copy_w, copy_dw) {
}

template<typename R>
vector<Mat<R>> SecondOrderCombinator<R>::parameters() const {
    return { W1, W2, b };
}

template<typename R>
Mat<R> SecondOrderCombinator<R>::activate(Mat<R> i1, Mat<R> i2) const {
    // TODO(jonathan): should be replaced with mul_mul_mul_with_mul
    return i1.dot(W1) * i2.dot(W2) + b;
}

template class SecondOrderCombinator<float>;
template class SecondOrderCombinator<double>;

/* RNN */
template<typename R>
RNN<R>::RNN() {
}

template<typename R>
RNN<R>::RNN (int _input_size, int _hidden_size) :
        hidden_size(_hidden_size),
        input_size(_input_size),
        output_size(_hidden_size) {
    create_variables();
}

template<typename R>
RNN<R>::RNN (int _input_size, int _hidden_size, int _output_size) :\
        hidden_size(_hidden_size),
        input_size(_input_size),
        output_size(_output_size) {
    create_variables();
}

template<typename R>
RNN<R>::RNN (const RNN<R>& rnn, bool copy_w, bool copy_dw) :
        hidden_size(rnn.hidden_size),
        input_size(rnn.input_size),
        output_size(rnn.output_size) {
    Wx = Mat<R>(rnn.Wx, copy_w, copy_dw);
    Wh = Mat<R>(rnn.Wh, copy_w, copy_dw);
    b = Mat<R>(rnn.b, copy_w, copy_dw);
}

template<typename R>
RNN<R> RNN<R>::shallow_copy() const {
    return RNN<R>(*this, false, true);
}

template<typename R>
Mat<R> RNN<R>::activate(
    Mat<R> input_vector,
    Mat<R> prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    DEBUG_ASSERT_MAT_NOT_NAN(Wx)
    DEBUG_ASSERT_MAT_NOT_NAN(input_vector)
    DEBUG_ASSERT_MAT_NOT_NAN(Wh)
    DEBUG_ASSERT_MAT_NOT_NAN(prev_hidden)
    DEBUG_ASSERT_MAT_NOT_NAN(b)
    return MatOps<R>::mul_add_mul_with_bias({Wx, Wh},  {input_vector, prev_hidden}, b);
}

template class Layer<float>;
template class Layer<double>;

template class StackedInputLayer<float>;
template class StackedInputLayer<double>;

template class RNN<float>;
template class RNN<double>;

