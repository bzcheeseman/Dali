import numpy as np
import subprocess
import io

def rnn_param_shapes(input_size, hidden_size):
    return [
        (hidden_size, input_size), # Wx
        (hidden_size, hidden_size), # Wh
        (hidden_size, 1) # b
    ]
def lstm_param_shapes(input_size, hidden_size):
    sizes = []
    # input_layer
    sizes.extend(rnn_param_shapes(input_size, hidden_size))
    # forget_layer
    sizes.extend(rnn_param_shapes(input_size, hidden_size))
    # output_layer
    sizes.extend(rnn_param_shapes(input_size, hidden_size))
    # cell_layer
    sizes.extend(rnn_param_shapes(input_size, hidden_size))
    return sizes

def classifier_shapes(input_size, output_size):
    return [(output_size, input_size), (output_size, 1)]

def embedding_shape(vocab_size, input_size):
    return [(vocab_size, input_size)]

def receive_params(process_output, num_shapes):
    bytes_io = io.BytesIO(process_output)
    return [np.load(bytes_io) for i in range(num_shapes)]   

def get_com(p):
    process_output,_ = p.communicate("")
    return process_output

def send_param(p, params):
    temp = io.BytesIO()
    for param in params:
        np.save(temp, param)
    temp.seek(0)
    process_input = temp.read()
    process_output,_ = p.communicate(process_input)
    return process_output


def run_optimization(program, params):
    # create loadable params
    p = subprocess.Popen(c_program,
          stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)

    process_output = send_param(p, params)

    cost, matrices = process_output.split(b'\n', 1)
    new_weights = receive_params(matrices, len(params))
    return new_weights, cost

def create_matrices(shapes, dtype='float32'):
    return [np.random.uniform(-0.2, 0.2, size=shape).astype(dtype) for shape in shapes]

if __name__ == "__main__":
    c_program    = 'examples/character_prediction.o'

    input_size = 5
    vocab_size = 300
    hidden_sizes = [20, 20]

    # create shapes:
    shapes = []
    shapes.extend( embedding_shape(vocab_size, input_size) )
    shapes.extend( classifier_shapes(hidden_sizes[-1], vocab_size) )
    shapes.extend( lstm_param_shapes(input_size, hidden_sizes[0]) )
    shapes.extend( lstm_param_shapes(hidden_sizes[0], hidden_sizes[1]) )

    new_mats, cost = run_optimization(c_program, create_matrices(shapes))
    print("epoch (101) = %.3f" % float(cost))

    new_mats, cost = run_optimization(c_program, new_mats)
    print("epoch (202) = %.3f" % float(cost))