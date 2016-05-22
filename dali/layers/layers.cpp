#include "dali/layers/Layers.h"

using std::vector;

////////////////////////////////////////////////////////////////////////////////
//                        AbstractMultiInputLayer                             //
////////////////////////////////////////////////////////////////////////////////

Tensor AbstractMultiInputLayer::activate(const vector<Tensor>& inputs) const {
    assert(inputs.size() > 0);
    return activate(inputs[0]);
};

Tensor AbstractMultiInputLayer::activate(Tensor first_input, const vector<Tensor>& inputs) const {
    if (inputs.size() > 0) {
        return activate(inputs.back());
    } else {
        return activate(first_input);
    }
};

////////////////////////////////////////////////////////////////////////////////
//                               Layer                                        //
////////////////////////////////////////////////////////////////////////////////

void Layer::create_variables() {
    double bound = 1.0 / sqrt(input_size);
    W       = Tensor::uniform(-bound, bound, {input_size, hidden_size});
    this->b = Tensor::zeros({1, hidden_size});
}

Layer::Layer() {
}

Layer::Layer (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
    create_variables();
}

Tensor Layer::activate(Tensor input_vector) const {
    return tensor_ops::mul_with_bias(W, input_vector, this->b);
}

Layer::Layer (const Layer& layer, bool copy_w, bool copy_dw) :
        hidden_size(layer.hidden_size),
        input_size(layer.input_size) {
    W       = Tensor(layer.W, copy_w, copy_dw);
    this->b = Tensor(layer.b, copy_w, copy_dw);
}

Layer Layer::shallow_copy() const {
    return Layer(*this, false, true);
}

vector<Tensor> Layer::parameters() const{
    return vector<Tensor>({W, this->b});
}


////////////////////////////////////////////////////////////////////////////////
//                        StackedInputLayer                                   //
////////////////////////////////////////////////////////////////////////////////

void StackedInputLayer::create_variables() {
    int total_input_size = 0;
    for (auto& input_size : input_sizes) {
        total_input_size += input_size;
    }
    tensors.reserve(input_sizes.size());

    double bound = 1.0 / sqrt(total_input_size);
    for (auto& input_size : input_sizes) {
        tensors.emplace_back(Tensor::uniform(-bound, bound, {input_size, hidden_size});
    }
    this->b = Tensor::zeros({1, hidden_size});
}

StackedInputLayer::StackedInputLayer() {
}

const vector<int>& StackedInputLayer::input_sizes() const {
    return input_sizes;
}

void StackedInputLayer::input_sizes(vector<int> new_sizes) {
    ASSERT2(new_sizes.size() > 0, "StackedInputLayer must have at least one input (0 provided)");
    // optimistic exit
    if (new_sizes == input_sizes)
        return;

    // new random numbers
    int total_input_size = 0;
    for (auto& input_size : new_sizes) {
        total_input_size += input_size;
    }
    double bound = 1.0 / sqrt(total_input_size);

    // construct tensors
    tensors = vector<Tensor>();
    for (auto& input_size : new_sizes) {
        tensors.emplace_back(Tensor::uniform(-bound, bound, {input_size, hidden_size});
    }

    // save new size
    input_sizes = new_sizes;
}

StackedInputLayer::StackedInputLayer (vector<int> input_sizes_, int hidden_size_) :
        input_sizes(input_sizes_),
        hidden_size(hidden_size_) {
    create_variables();
}

StackedInputLayer::StackedInputLayer (int input_size, int output_size) :
        hidden_size(output_size),
        input_sizes({input_size}) {
    create_variables();
}

StackedInputLayer::StackedInputLayer (std::initializer_list<int> input_sizes_,
                                      int hidden_size_) :
        hidden_size(hidden_size_),
        input_sizes(input_sizes_) {
    create_variables();
}

Tensor StackedInputLayer::activate(std::initializer_list<Tensor> inputs) const {
    vector<Tensor> temp(inputs);
    return activate(temp);
}

Tensor StackedInputLayer::activate(const vector<Tensor>& inputs) const {
    return tensor_ops::mul_add_mul_with_bias(tensors, inputs, this->b);
}

Tensor StackedInputLayer::activate(
        Tensor input_vector) const {
    if (tensors.size() == 1) {
        return tensor_ops::mul_with_bias(tensors.front(), input_vector, this->b);
    } else {
        throw std::runtime_error("Error: Stacked Input Layer parametrized with more than 1 inputs only received 1 input vector.");
    }
}

Tensor StackedInputLayer::activate(
        Tensor input,
        const vector<Tensor>& inputs) const {
    DEBUG_ASSERT_MAT_NOT_NAN(input)

    vector<Tensor> zipped;
    zipped.emplace_back(input);
    for (auto& an_input : inputs )
        zipped.emplace_back(an_input);

    auto out = tensor_ops::mul_add_mul_with_bias(tensors, zipped, this->b);

    DEBUG_ASSERT_MAT_NOT_NAN(out)

    return out;
}

StackedInputLayer::StackedInputLayer (const StackedInputLayer& layer, bool copy_w, bool copy_dw) :
        hidden_size(layer.hidden_size),
        input_sizes(layer.input_sizes()) {
    tensors.reserve(layer.tensors.size());
    for (auto& tensor : layer.tensors)
        tensors.emplace_back(tensor, copy_w, copy_dw);
    this->b = Tensor(layer.b, copy_w, copy_dw);
}

StackedInputLayer StackedInputLayer::shallow_copy() const {
    return StackedInputLayer(*this, false, true);
}

std::vector<Tensor> StackedInputLayer::parameters() const{
    auto params = vector<Tensor>(tensors);
    params.emplace_back(this->b);
    return params;
}


////////////////////////////////////////////////////////////////////////////////
//                        MultiLayerPerceptron                                //
////////////////////////////////////////////////////////////////////////////////

MultiLayerPerceptron::MultiLayerPerceptron() {
}

MultiLayerPerceptron::MultiLayerPerceptron(vector<int> hidden_sizes, vector<activation_t> activations) :
        hidden_sizes(hidden_sizes),
        activations(activations) {
    ASSERT2(activations.size() == hidden_sizes.size() - 1,
            "Wrong number of activations for MultiLayerPerceptron");

    for (int lidx = 0; lidx < hidden_sizes.size() - 1; ++lidx) {
        layers.push_back(Layer(hidden_sizes[lidx], hidden_sizes[lidx + 1]));
    }
}

MultiLayerPerceptron::MultiLayerPerceptron(const MultiLayerPerceptron& other, bool copy_w, bool copy_dw) :
        hidden_sizes(other.hidden_sizes),
        activations(other.activations) {
    for (auto& other_layer: other.layers) {
        layers.emplace_back(other_layer, copy_w, copy_dw);
    }
}

MultiLayerPerceptron MultiLayerPerceptron::shallow_copy() const {
    return MultiLayerPerceptron(*this, false, true);
}

Tensor MultiLayerPerceptron::activate(Tensor input) const {
    Tensor last_output = input;
    for (int i = 0; i < hidden_sizes.size() - 1; ++i)
        last_output = activations[i](layers[i].activate(last_output));

    return last_output;
}

vector<Tensor> MultiLayerPerceptron::parameters() const {
    vector<Tensor> params;
    for (auto& layer: layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}
Tensor MultiLayerPerceptron::identity(Tensor m) { return m; }


////////////////////////////////////////////////////////////////////////////////
//                             DelayedRNN                                     //
////////////////////////////////////////////////////////////////////////////////

DelayedRNN::DelayedRNN(int input_size, int hidden_size, int output_size) :
        hidden_rnn(input_size, hidden_size),
        output_rnn(input_size, hidden_size, output_size) {
}

DelayedRNN::DelayedRNN (const DelayedRNN& rnn, bool copy_w, bool copy_dw) :
        hidden_rnn(rnn.hidden_rnn, copy_w, copy_dw),
        output_rnn(rnn.output_rnn, copy_w, copy_dw) {
}

vector<Tensor> DelayedRNN::parameters() const {
    vector<Tensor> ret;
    for (auto& model: {hidden_rnn, output_rnn}) {
        auto params = model.parameters();
        ret.insert(ret.end(), params.begin(), params.end());
    }
    return ret;
}

Tensor DelayedRNN::initial_states() const {
    double bound = 1.0 / sqrt(hidden_rnn.hidden_size)
    return Tensor(-bound, bound, {hidden_rnn.hidden_size, 1})
}

std::tuple<Tensor,Tensor> DelayedRNN::activate(
        Tensor input_vector,
        Tensor prev_hidden) const {

    return std::make_tuple(
        hidden_rnn.activate(input_vector, prev_hidden),
        output_rnn.activate(input_vector, prev_hidden)
    );
}

DelayedRNN DelayedRNN::shallow_copy() const {
    return DelayedRNN(*this, false, true);
}


////////////////////////////////////////////////////////////////////////////////
//                       SecondOrderCombinator                                //
////////////////////////////////////////////////////////////////////////////////

SecondOrderCombinator::SecondOrderCombinator() {
}

SecondOrderCombinator::SecondOrderCombinator(int input1_size, int input2_size, int output_size) :
        input1_size(input1_size), input2_size(input2_size), output_size(output_size) {

    W1 = Tensor::uniform(-1.0 / sqrt(input1_size), 1.0 / sqrt(input1_size), {input1_size, output_size});
    21 = Tensor::uniform(-1.0 / sqrt(input1_size), 1.0 / sqrt(input2_size), {input2_size, output_size});
    b =  Tensor::zeros({1, output_size});
}
SecondOrderCombinator::SecondOrderCombinator(const SecondOrderCombinator& m,
                                                bool copy_w,
                                                bool copy_dw) :
        input1_size(m.input1_size),
        input2_size(m.input2_size),
        output_size(m.output_size),
        W1(m.W1, copy_w, copy_dw),
        W2(m.W2, copy_w, copy_dw),
        b(m.b, copy_w, copy_dw) {
}

vector<Tensor> SecondOrderCombinator::parameters() const {
    return { W1, W2, b };
}

Tensor SecondOrderCombinator::activate(Tensor i1, Tensor i2) const {
    // TODO(jonathan): should be replaced with mul_mul_mul_with_mul
    // (to:szymon, from:jonathan, I'll try)
    return i1.dot(W1) * i2.dot(W2) + b;
}



////////////////////////////////////////////////////////////////////////////////
//                                RNN                                         //
////////////////////////////////////////////////////////////////////////////////


void RNN::create_variables() {
    Wx = Tensor::uniform(-1.0 / sqrt(input_size), 1.0 / sqrt(input_size), {input_size, output_size});
    Wx = Tensor::uniform(-1.0 / sqrt(hidden_size), 1.0 / sqrt(hidden_size), {hidden_size, output_size});

    b  = Tensor::zeros({1, output_size});
}

vector<Tensor> RNN::parameters() const {
    return vector<Tensor>({Wx, Wh, b});
}

RNN::RNN() {
}

RNN::RNN (int input_size_, int hidden_size_) :
        hidden_size(hidden_size_),
        input_size(input_size_),
        output_size(hidden_size_) {
    create_variables();
}

RNN::RNN (int input_size_, int hidden_size_, int output_size_) :\
        hidden_size(hidden_size_),
        input_size(input_size_),
        output_size(output_size_) {
    create_variables();
}

RNN::RNN (const RNN& rnn, bool copy_w, bool copy_dw) :
        hidden_size(rnn.hidden_size),
        input_size(rnn.input_size),
        output_size(rnn.output_size) {
    Wx = Tensor(rnn.Wx, copy_w, copy_dw);
    Wh = Tensor(rnn.Wh, copy_w, copy_dw);
    b = Tensor(rnn.b, copy_w, copy_dw);
}

RNN RNN::shallow_copy() const {
    return RNN(*this, false, true);
}

Tensor RNN::activate(
    Tensor input_vector,
    Tensor prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.mul_with_bias(Wh, prev_hidden, b));
    DEBUG_ASSERT_MAT_NOT_NAN(Wx)
    DEBUG_ASSERT_MAT_NOT_NAN(input_vector)
    DEBUG_ASSERT_MAT_NOT_NAN(Wh)
    DEBUG_ASSERT_MAT_NOT_NAN(prev_hidden)
    DEBUG_ASSERT_MAT_NOT_NAN(b)
    return tensor_ops::mul_add_mul_with_bias({Wx, Wh},  {input_vector, prev_hidden}, b);
}
