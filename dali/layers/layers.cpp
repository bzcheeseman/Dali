#include "layers.h"

#include "dali/tensor/op.h"

using std::vector;


////////////////////////////////////////////////////////////////////////////////
//                             AbstractLayer                                  //
////////////////////////////////////////////////////////////////////////////////

AbstractLayer::AbstractLayer() :
        dtype(DTYPE_FLOAT),
        device(memory::default_preferred_device) {
}


AbstractLayer::AbstractLayer(DType dtype, memory::Device device) :
        dtype(dtype),
        device(device) {
}



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
    W       = Tensor::uniform(-bound, bound, {input_size, hidden_size}, dtype, device);
    this->b = Tensor::zeros({hidden_size}, dtype, device)[Broadcast()];
}

Layer::Layer() {
}

Layer::Layer (int input_size_, int hidden_size_, DType dtype_, memory::Device device_) :
        hidden_size(hidden_size_),
        input_size(input_size_),
        AbstractMultiInputLayer(dtype_, device_) {
    create_variables();
}

Tensor Layer::activate(Tensor input_vector) const {
    return tensor_ops::dot_with_bias(input_vector, W, this->b);
}

Layer::Layer (const Layer& other, bool copy_w, bool copy_dw) :
        hidden_size(other.hidden_size),
        input_size(other.input_size),
        AbstractMultiInputLayer(other.dtype, other.device) {
    W       = Tensor(other.W, copy_w, copy_dw);
    this->b = Tensor(other.b, copy_w, copy_dw);
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
        tensors.emplace_back(Tensor::uniform(-bound, bound, {input_size, hidden_size}, dtype, device));
    }
    this->b = Tensor::zeros({hidden_size}, dtype, device)[Broadcast()];
}

StackedInputLayer::StackedInputLayer() {
}

const vector<int>& StackedInputLayer::get_input_sizes() const {
    return input_sizes;
}

void StackedInputLayer::set_input_sizes(vector<int> new_sizes) {
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
        tensors.emplace_back(Tensor::uniform(-bound, bound,
                                             {input_size, hidden_size},
                                             dtype,
                                             device));
    }

    // save new size
    input_sizes = new_sizes;
}

StackedInputLayer::StackedInputLayer (vector<int> input_sizes_,
                                      int hidden_size_,
                                      DType dtype,
                                      memory::Device device) :
        input_sizes(input_sizes_),
        hidden_size(hidden_size_),
        AbstractMultiInputLayer(dtype, device) {
    create_variables();
}

StackedInputLayer::StackedInputLayer (int input_size,
                                      int output_size,
                                      DType dtype,
                                      memory::Device device) :
        hidden_size(output_size),
        input_sizes({input_size}),
        AbstractMultiInputLayer(dtype, device) {
    create_variables();
}

Tensor StackedInputLayer::activate(std::initializer_list<Tensor> inputs) const {
    vector<Tensor> temp(inputs);
    return activate(temp);
}

Tensor StackedInputLayer::activate(const vector<Tensor>& inputs) const {
    return tensor_ops::multiple_dot_with_bias(inputs, tensors, this->b);
}

Tensor StackedInputLayer::activate(
        Tensor input_vector) const {
    if (tensors.size() == 1) {
        return tensor_ops::dot_with_bias(input_vector, tensors.front(), this->b);
    } else {
        throw std::runtime_error("Error: Stacked Input Layer parametrized with more than 1 inputs only received 1 input vector.");
    }
}

Tensor StackedInputLayer::activate(
        Tensor input,
        const vector<Tensor>& inputs) const {

    vector<Tensor> zipped;
    zipped.emplace_back(input);
    for (auto& an_input : inputs )
        zipped.emplace_back(an_input);

    auto out = tensor_ops::multiple_dot_with_bias(zipped, tensors, this->b);

    return out;
}

StackedInputLayer::StackedInputLayer (const StackedInputLayer& other, bool copy_w, bool copy_dw) :
        hidden_size(other.hidden_size),
        input_sizes(other.get_input_sizes()),
        AbstractMultiInputLayer(other.dtype, other.device) {
    tensors.reserve(other.tensors.size());
    for (auto& tensor : other.tensors)
        tensors.emplace_back(tensor, copy_w, copy_dw);
    this->b = Tensor(other.b, copy_w, copy_dw);
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
//                       SecondOrderCombinator                                //
////////////////////////////////////////////////////////////////////////////////

SecondOrderCombinator::SecondOrderCombinator() {
}

SecondOrderCombinator::SecondOrderCombinator(int input1_size,
                                             int input2_size,
                                             int output_size,
                                             DType dtype_,
                                             memory::Device device_) :
        input1_size(input1_size),
        input2_size(input2_size),
        output_size(output_size),
        AbstractLayer(dtype_, device_) {
    W1 = Tensor::uniform(-1.0 / sqrt(input1_size), 1.0 / sqrt(input1_size),
                         {input1_size, output_size},
                         dtype,
                         device);
    W2 = Tensor::uniform(-1.0 / sqrt(input1_size), 1.0 / sqrt(input2_size),
                         {input2_size, output_size},
                         dtype,
                         device);
    b =  Tensor::zeros({output_size}, dtype, device)[Broadcast()];
}

SecondOrderCombinator::SecondOrderCombinator(const SecondOrderCombinator& other,
                                             bool copy_w,
                                             bool copy_dw) :
        input1_size(other.input1_size),
        input2_size(other.input2_size),
        output_size(other.output_size),
        W1(other.W1, copy_w, copy_dw),
        W2(other.W2, copy_w, copy_dw),
        b(other.b, copy_w, copy_dw),
        AbstractLayer(other.dtype, other.device) {
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
    Wx = Tensor::uniform(-1.0 / sqrt(input_size), 1.0 / sqrt(input_size),
                         {input_size, output_size},
                         dtype,
                         device);
    Wh = Tensor::uniform(-1.0 / sqrt(hidden_size), 1.0 / sqrt(hidden_size),
                         {hidden_size, output_size},
                         dtype,
                         device);
    b  = Tensor::zeros({output_size}, dtype, device)[Broadcast()];
}

vector<Tensor> RNN::parameters() const {
    return vector<Tensor>({Wx, Wh, b});
}

RNN::RNN() {
}

RNN::RNN (int input_size_, int hidden_size_, DType dtype_, memory::Device device_) :
        hidden_size(hidden_size_),
        input_size(input_size_),
        output_size(hidden_size_),
        AbstractLayer(dtype_, device_) {
    create_variables();
}

RNN::RNN (int input_size_, int hidden_size_, int output_size_, DType dtype_, memory::Device device_) :
        hidden_size(hidden_size_),
        input_size(input_size_),
        output_size(output_size_),
        AbstractLayer(dtype_, device_) {
    create_variables();
}

RNN::RNN (const RNN& other, bool copy_w, bool copy_dw) :
        hidden_size(other.hidden_size),
        input_size(other.input_size),
        output_size(other.output_size),
        AbstractLayer(other.dtype, other.device) {
    Wx = Tensor(other.Wx, copy_w, copy_dw);
    Wh = Tensor(other.Wh, copy_w, copy_dw);
    b = Tensor(other.b, copy_w, copy_dw);
}

RNN RNN::shallow_copy() const {
    return RNN(*this, false, true);
}

Tensor RNN::activate(
        Tensor input_vector,
        Tensor prev_hidden) const {
    // takes 5% less time to run operations when grouping them (no big gains then)
    // 1.118s with explicit (& temporaries) vs 1.020s with grouped expression & backprop
    // return G.add(G.mul(Wx, input_vector), G.dot_with_bias(Wh, prev_hidden, b));
    return tensor_ops::multiple_dot_with_bias({input_vector, prev_hidden}, {Wx, Wh}, b);
}


Tensor RNN::initial_states() const {
    return Tensor({hidden_size}, dtype, device)[Broadcast()];
}
