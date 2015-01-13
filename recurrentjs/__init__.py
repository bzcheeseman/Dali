"""
Python module for automatic differentiation following the steps of Andrej Kaparthy
in his 2014 [recurrentjs library](https://github.com/karpathy/recurrentjs)

@translator: Jonathan Raiman
@date: Janurary 13th 2015

TODO: just as in Cython LSTM, goal is to have simple, intuitive, and monstrously fast library (Torch fast)

1. Switch entire module to C++ (fully, not an extension),
2. get rid of lambda expressions,
3. build clean constructors and wrappers for LSTM, Layers, RNNs, etc. so API stays simple.

"""

import numpy as np
try:
    from scipy.special import expit as sigmoid
except ImportError:
    # in case people don't have scipy
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

class Mat(object):
    """
    Matrix wrapper class that holds weights
    and elementwise derivatives for each element
    in the matrix.

    Inputs:
    -------

    n int : number of rows
    d int : number of columns

    """
    __slots__ = ["n", "d", "w", "dw"]

    def __init__(self, n, d, mat=None):
        self.w = np.zeros([n, d]) if mat is None else mat
        self.dw = np.zeros([n, d])
        self.n = n
        self.d = d
        
def RandMat(n, d, std):
    """
    Constructor for matrices initialized with random weights.


    Inputs:
    -------

    n int : number of rows
    d int : number of columns
    std float : standard deviation for gaussian from which
                weights are sampled


    Outputs:
    --------

    mat Mat : the matrix with random weights.

    See `Mat`
    """
    return Mat(n, d, np.random.standard_normal([n, d]) * std)

class Graph(object):
    """
    Graph holding the computation backprop steps.
    Knows about the operator derivatives for
    several common computations (sigmoid, tanh, slices,
    matrix multiplies, etc..).
    Not as rich as most other Tensor and A.D. graph libraries
    but easily extensible.

    Inputs:
    -------
    needs_backprop boolean (optional) : whether ops should keep their
                                        derivative steps for backprop.

    """
    __slots__ = ["backprop", "needs_backprop"]
    
    def __init__(self, needs_backprop=True):
        self.needs_backprop = needs_backprop
        self.backprop = []
        
    def backward(self):
        """
        Travel graph backwards, applying the backpropagation
        steps.

        """
        for step in reversed(self.backprop):
            step()
            
    def row_pluck(self, matrix, ix):
        """
        Row slice of a matrix

            > y = matrix[ index , : ]

        """
        out = Mat(matrix.d, 1)
        out.w[:,0] = matrix.w[ix]
        
        if self.needs_backprop:
            def backward():
                matrix.dw[ix,:] += out.dw[:,0]
                
            self.backprop.append(backward)
        
        return out
    
    def tanh(self, matrix):
        """
        Hyperbolic tangent activation element wise on a matrix:

            > y = tanh( x )

        """
        out = Mat(matrix.n, matrix.d, np.tanh(matrix.w))
        
        if self.needs_backprop:
            def backward():
                matrix.dw += (1. - (out.w ** 2)) * out.dw
            self.backprop.append(backward)
            
        return out
    
    def sigmoid(self, matrix):
        """
        Sigmoid activation (or logistic) element wise on a matrix:

            > y = 1. / ( 1. + exp(-x) )

        """
        out = Mat(matrix.n, matrix.d, sigmoid(matrix.w))
        
        if self.needs_backprop:
            def backward():
                matrix.dw += out.w * (1. - out.w) * out.dw
            self.backprop.append(backward)
            
        return out
    
    def relu(self, matrix):
        """
        Rectified linear activation element wise on a matrix

            > y = max(0, x)

        """
        out = Mat(matrix.n, matrix.d, np.fmax(0, matrix.w))
        if self.needs_backprop:
            def backward():
                matrix.dw += np.sign(out.w) * out.dw 
            self.backprop.append(backward)
            
        return out
    
    def mul(self, matrix1, matrix2):
        """
        Matrix dot product
        """
        assert(matrix1.d == matrix2.n), "matmul dimensions misaligned"
        out = Mat(matrix1.n, matrix2.d, matrix1.w.dot(matrix2.w))
        
        if self.needs_backprop:
            def backward():
                matrix1.dw += out.dw.dot(matrix2.w.T)
                matrix2.dw += matrix1.w.T.dot(out.dw)
            self.backprop.append(backward)
        return out
    
    def add(self, matrix1, matrix2):
        """
        Element add two matrices
        """
        assert(matrix1.n == matrix2.n and matrix1.d == matrix2.d)
        out = Mat(matrix1.n, matrix1.d, matrix1.w + matrix2.w)
        
        if self.needs_backprop:
            def backward():
                matrix1.dw += out.dw
                matrix2.dw += out.dw
            self.backprop.append(backward)
        return out
    
    def eltmul(self, matrix1, matrix2):
        """
        Element multply two matrices
        """
        assert(matrix1.n == matrix2.n and matrix1.d == matrix2.d)
        out = Mat(matrix1.n, matrix1.d, matrix1.w * matrix2.w)
        if self.needs_backprop:
            def backward():
                matrix1.dw += matrix2.dw * out.dw
                matrix2.dw += matrix1.dw * out.dw
            self.backprop.append(backward)
        return out
    
def softmax(matrix):
    layer_max = matrix.w.max()
    exped_distributions = np.exp(matrix.w - layer_max)
    total_distribution = exped_distributions.sum()
    
    out = Mat(matrix.n, matrix.d, exped_distributions / total_distribution)
    
    return out
      
class Solver(object):
    """
    Optimizer for Graphs applying gradient descent with RMS prop.

    Inputs:
    -------

    decay_rate float (optional): RMS prop decay rate (adaptive learning
                                 rate hyperparameter, controls the leaky integration
                                 rate for squared gradients)

    smooth_eps float (optional): epsilon value used for numerical stability
                                 (so division by zero doesn't occur)

    """


    def __init__(self, decay_rate = 0.999, smooth_eps = 1e-8):
        self.decay_rate = decay_rate
        self.smooth_eps = smooth_eps
        self.step_cache = {}
    
    def step(self, model, step_size, regc, clipval):
        """
        Perform a step of Gradient descent step using RMS prop on
        the parameters provided by `model`.


        Inputs:
        -------

        model      dict : dictionary with model parameters to update
        step_size float : learning rate for gradient descent
        regc      float : L2 regularization parameter
        clipval   float : cutoff for elementwise gradients (prevents
                          [some] explosions from happening during
                          learning )

        Outputs:
        --------

        None

        """
        for param in model.values():
            if param not in self.step_cache:
                self.step_cache[param] = Mat(param.n, param.d)
                
            s = self.step_cache[param]
            
            grad = param.dw
            s.w[:] = s.w[:] * self.decay_rate + (1. - self.decay_rate) * grad**2
            
            np.clip(grad, -clipval, clipval, out=grad)
            
            param.w += - (step_size * (grad / np.sqrt(s.w + self.smooth_eps))) - regc * param.w
            param.dw.fill(0)
            
def initLSTM(input_size, hidden_sizes, output_size):
    """
    Initialize LSTM parameters with a decoder (à la Andrej Kaparthy)

    Inputs:
    -------

    input_size    int : the size of input to the LSTM (bottom layer)
    hidden_sizes list : hidden sizes for the different stacked LSTM layers
                        (each hidden LSTM gets fed up as input to the next).
    output_size   int : final layer size for decoding the hidden activation of
                        topmost LSTM


    Outputs:
    --------

    model dict : Python dictionary with model parameter matrices (`Mat`)

    See `Mat`.

    """
    model = {}
    for d in range(len(hidden_sizes)):
        prev_size = input_size if d == 0 else hidden_sizes[d-1]
        hidden_size = hidden_sizes[d]
        
        model["Wix%d" % d] = RandMat(hidden_size, prev_size, 0.08)
        model["Wih%d" % d] = RandMat(hidden_size, hidden_size, 0.08)
        
        model["bi%d"  % d] = Mat(hidden_size, 1)
        
        model["Wfx%d" % d] = RandMat(hidden_size, prev_size, 0.08)
        model["Wfh%d" % d] = RandMat(hidden_size, hidden_size, 0.08)
        
        model["bf%d"  % d] = Mat(hidden_size, 1)
        
        model['Wox%d' % d] = RandMat(hidden_size, prev_size , 0.08)
        model['Woh%d' % d] = RandMat(hidden_size, hidden_size , 0.08)
        
        model['bo%d'  % d] = Mat(hidden_size, 1)
        
        # cell write params
        model['Wcx%d' % d] = RandMat(hidden_size, prev_size , 0.08)
        model['Wch%d' % d] = RandMat(hidden_size, hidden_size , 0.08)
        model['bc%d'  % d] = Mat(hidden_size, 1)
    
    # decoder params
    model['Whd'] = RandMat(output_size, hidden_size, 0.08)
    model['bd'] = Mat(output_size, 1)
    
    return model

class LSTMact:
    """
    Utility class for storing outputs of LSTM at every step
    through time
    """
    __slots__ = ["h", "c", "output"]
    def __init__(self, h, c, output):
        self.h = h
        self.c = c
        self.output = output

def forwardLSTM(G, model, hidden_sizes, x, prev):
    """
    forward prop for a single tick of LSTM
    
    Inputs:
    -------
    
    G      Graph : G is graph to append ops to,
    model   dict : contains LSTM parameters
    x    ndarray : 1D column vector with observation
    prev LSTMact : struct containing hidden and cell activations
                   from previous iteration
    
    Outputs:
    --------
    
    out LSTMact : new hidden, cell, and output activations for this step
    
    """
    if prev is None:
        hidden_prevs = []
        cell_prevs = []
        for d in range(len(hidden_sizes)):
            hidden_prevs.append( Mat(hidden_sizes[d], 1) )
            cell_prevs.append( Mat(hidden_sizes[d], 1) )
    else:
        hidden_prevs = prev.h
        cell_prevs = prev.c
        
    hidden = []
    cell = []
    for d in range(len(hidden_sizes)):
        input_vector = x if d == 0 else hidden[d-1]
        hidden_prev = hidden_prevs[d]
        cell_prev = cell_prevs[d]
        
        # input gate
        h0 = G.mul(model["Wix%d" % d], input_vector)
        h1 = G.mul(model["Wih%d" % d], hidden_prev)
        input_gate = G.sigmoid(G.add(G.add(h0, h1), model["bi%d" % d]))
        
        # forget gate
        h2 = G.mul(model['Wfx%d' % d], input_vector)
        h3 = G.mul(model['Wfh%d' % d], hidden_prev)
        forget_gate = G.sigmoid(G.add(G.add(h2, h3),model['bf%d' % d]))

        # output gate
        h4 = G.mul(model['Wox%d' % d], input_vector)
        h5 = G.mul(model['Woh%d' % d], hidden_prev)
        output_gate = G.sigmoid(G.add(G.add(h4, h5),model['bo%d' % d]))

        # write operation on cells
        h6 = G.mul(model['Wcx%d' % d], input_vector)
        h7 = G.mul(model['Wch%d' % d], hidden_prev)
        cell_write = G.tanh(G.add(G.add(h6, h7),model['bc%d' % d]));

        # compute new cell activation
        retain_cell = G.eltmul(forget_gate, cell_prev) # what do we keep from cell
        write_cell = G.eltmul(input_gate, cell_write) # what do we write to cell
        cell_d = G.add(retain_cell, write_cell) # new cell contents

        # compute hidden state as gated, saturated cell activations
        hidden_d = G.eltmul(output_gate, G.tanh(cell_d))
    
        hidden.append(hidden_d)
        cell.append(cell_d)
    
    # one decoder to outputs at end
    output = G.add(G.mul(model['Whd'], hidden[-1]), model['bd'])

    # return cell memory, hidden representation and output
    return LSTMact(hidden, cell, output)

def initRNN(input_size, hidden_sizes, output_size):
    """
    Initialize RNN parameters with a decoder (à la Andrej Kaparthy)

    Inputs:
    -------

    input_size    int : the size of input to the RNN (bottom layer)
    hidden_sizes list : hidden sizes for the different stacked RNN layers
                        (each hidden RNN gets fed up as input to the next).
    output_size   int : final layer size for decoding the hidden activation of
                        topmost RNN


    Outputs:
    --------

    model dict : Python dictionary with model parameter matrices (`Mat`)

    See `Mat`.
    
    """
    model = {}
    
    for d in range(len(hidden_sizes)):
        prev_size = input_size if d == 0 else hidden_sizes[d-1]
        hidden_size = hidden_sizes[d]
        model["Wxh%d" % d] = RandMat(hidden_size, prev_size , 0.08)
        model['Whh%d' % d] = RandMat(hidden_size, hidden_size, 0.08)
        model['bhh%d' % d] = Mat(hidden_size, 1)
        
    # decoder params
    model["Whd"] = RandMat(output_size, hidden_size, 0.08)
    model['bd']  = Mat(output_size, 1)
    return model

def forwardRNN(G, model, hidden_sizes, x, prev):
    """
    forward prop for a single tick of RNN
    
    Inputs:
    -------
    
    G      Graph : G is graph to append ops to,
    model   dict : contains RNN parameters
    x    ndarray : 1D column vector with observation
    prev LSTMact : struct containing hidden and output activations
                   from previous iteration
    
    Outputs:
    --------
    
    out LSTMact : new hidden, cell, and output activations for this step
    
    """
    
    if prev is None:
        hidden_prevs = []
        for d in range(len(hidden_sizes)):
            hidden_prevs.append( Mat(hidden_sizes[d], 1 ) )
            
    else:
        hidden_prevs = prev.h
        
    hidden = []
    for d in range(len(hidden_sizes)):
        input_vector = x if d == 0 else hidden[d-1]
        hidden_prev = hidden_prevs[d]
        
        h0 = G.mul( model["Wxh%d" % d], input_vector)
        h1 = G.mul( model["Whh%d" % d], hidden_prev )
        
        hidden_d = G.relu( G.add( G.add(h0, h1), model["bhh%d" % d]) )
        
        hidden.append(hidden_d)
        
    # one decoder to outputs at end
    output = G.add(G.mul(model['Whd'], hidden[-1]),model['bd'])

    # return cell memory, hidden representation and output
    return LSTMact(hidden, None, output)

__all__ = [
    "Mat",
    "RandMat",
    "Graph",
    "softmax",
    "initLSTM",
    "forwardLSTM",
    "LSTMact",
    "initRNN",
    "forwardRNN",
    ]