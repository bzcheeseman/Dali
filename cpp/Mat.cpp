#include "utils.hpp"
#include "Mat.hpp"

template<typename T>
class LSTM {
    /*
    Initialize LSTM Layer à la Andrej Kaparthy.

    Inputs:
    -------

    input_size    int& : the size of input to the LSTM
    hidden_size   int& : hidden size for cell, and hidden
                         output of this layer.

    Outputs:
    --------

    LSTM class

    See `Mat`.
    */
    typedef Mat<T>                      mat;
    typedef std::shared_ptr<mat> shared_mat;

    // cell input modulation:
    shared_mat Wix;
    shared_mat Wih;
    shared_mat bi;

    // cell forget gate:
    shared_mat Wfx;
    shared_mat Wfh;
    shared_mat bf;

    // cell output modulation
    shared_mat Wox;
    shared_mat Woh;
    shared_mat bo;

    // cell write params
    shared_mat Wcx;
    shared_mat Wch;
    shared_mat bc;

    void create_variables() {
        using std::make_shared;
        T std = 0.08;
        // initialize the parameters:
        Wix = make_shared<mat>(hidden_size, input_size, std);
        Wih = make_shared<mat>(hidden_size, hidden_size, std);
        bi  = make_shared<mat>(hidden_size, 1);

        Wfx = make_shared<mat>(hidden_size, input_size, std);
        Wfh = make_shared<mat>(hidden_size, hidden_size, std);
        bf  = make_shared<mat>(hidden_size, 1);

        Wox = make_shared<mat>(hidden_size, input_size, std);
        Woh = make_shared<mat>(hidden_size, hidden_size, std);
        bo  = make_shared<mat>(hidden_size, 1);

        Wcx = make_shared<mat>(hidden_size, input_size, std);
        Wch = make_shared<mat>(hidden_size, hidden_size, std);
        bc  = make_shared<mat>(hidden_size, 1);
    }

    public:
        const int hidden_size;
        const int input_size;

        LSTM (int _input_size, int _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
            create_variables();
        }
        LSTM (int& _input_size, int& _hidden_size) : hidden_size(_hidden_size), input_size(_input_size) {
            create_variables();
        }
        std::pair<shared_mat, shared_mat> forward (
            Graph<T>& G,
            shared_mat input_vector,
            shared_mat cell_prev,
            shared_mat hidden_prev) {

            // input gate:
            auto h0 = G.mul(Wix, input_vector);
            auto h1 = G.mul(Wih, hidden_prev);
            auto input_gate = G.sigmoid( G.add_broadcast( G.add(h0, h1), bi));

            // forget gate
            auto h2 = G.mul(Wfx, input_vector);
            auto h3 = G.mul(Wfh, hidden_prev);
            auto forget_gate = G.sigmoid( G.add_broadcast( G.add(h2, h3), bf));

            // output gate
            auto h4 = G.mul(Wox, input_vector);
            auto h5 = G.mul(Woh, hidden_prev);
            auto output_gate = G.sigmoid( G.add_broadcast( G.add(h4, h5), bo));

            // write operation on cells
            auto h6 = G.mul(Wcx, input_vector);
            auto h7 = G.mul(Wch, hidden_prev);
            auto cell_write = G.tanh( G.add_broadcast( G.add(h6, h7), bc));

            // compute new cell activation
            auto retain_cell = G.eltmul(forget_gate, cell_prev); // what do we keep from cell
            auto write_cell = G.eltmul(input_gate, cell_write); // what do we write to cell
            auto cell_d = G.add(retain_cell, write_cell); // new cell contents

            // compute hidden state as gated, saturated cell activations
            auto hidden_d = G.eltmul(output_gate, G.tanh(cell_d));

            return std::pair<shared_mat,shared_mat>(cell_d, hidden_d);
        }
};

int main() {
    using std::make_shared;
    using std::cout;
    using std::endl;
	typedef float REAL_t;
	typedef Mat<REAL_t> mat;

	// build blank matrix of double type:
    auto A = make_shared<mat>(3, 5);
    A->w = (A->w.array() + ((REAL_t) 1.2)).matrix();
    // build random matrix of double type with standard deviation 2:
    auto B = make_shared<mat>(A->n, A->d, (REAL_t) 2.0);
    auto C = make_shared<mat>(A->d, 4,    (REAL_t) 2.0);

    cout << "A                = " << endl;
    A->print();
    cout << "B                = " << endl;
    B->print();

    Graph<REAL_t>       graph;
	auto A_times_B    = graph.eltmul(A, B);
	auto A_plus_B_sig = graph.sigmoid(graph.add(A, B));
    auto A_dot_C_tanh = graph.tanh( graph.mul(A, C) );
    auto A_plucked    = graph.row_pluck(A, 2);
    
    cout << "A * B            =" << endl;
    A_times_B   ->print();
    cout << "sigmoid( A + B ) =" << endl;
    A_plus_B_sig->print();
    cout << "tanh( A dot C )  =" << endl;
    A_dot_C_tanh->print();
    cout << "A[2,:]           =" << endl;
    A_plucked   ->print();

    auto prod  = graph.mul(A, C);
	auto activ = graph.tanh(prod);

    // add some random singularity and use exponential
    // normalization:
    A_plucked->w(2,0) += (REAL_t) 3.0;
    auto A_plucked_normed = softmax(A_plucked);
    cout << "softmax(A[2,:]) + eps =" << endl;
    A_plucked_normed->print();

    REAL_t regc = 0.000001; //L2 regularization strength
    REAL_t learning_rate = 0.01;// learning rate
    REAL_t clipval = 5.0;//

    Solver<REAL_t> solver((REAL_t) 0.95, (REAL_t) 1e-6, clipval);
    auto model = std::vector<std::shared_ptr<mat>>();

    model.push_back(A);
    model.push_back(B);
    solver.step(model, learning_rate, regc);
    // ======== END BASICS =========

    // ======== ENTER LSTM =========
    LSTM<REAL_t> lstm(20, 30);
    int batchsize = 25;


    // words we care about (their indices:)
    std::vector<int> indices = {0, 1, 5, 2, 1, 2, 3};

    // some embedding matrix (hopefully one day)
    auto input_embed  = make_shared<mat>(1000, lstm.input_size, (REAL_t) 2.0);

    // what we've extracted so far:
    auto plucked = graph.rows_pluck(input_embed, indices);

    // pass this into the meatgrinder:
    auto cell_prev    = make_shared<mat>(lstm.hidden_size, 1);
    auto hidden_prev  = make_shared<mat>(lstm.hidden_size, 1);
    // while initially the cell_prev and hidden_prev are vectors
    // they get broadcasted down the pipe into the right sizes
    auto cell_hidden  = lstm.forward(graph, plucked, cell_prev, hidden_prev);

    cout << "LSTM of plucked rows =>" << endl;
    cell_hidden.first->print();

    // backpropagate
    graph.backward();

    // ========  END LSTM  =========

    return 0;
}