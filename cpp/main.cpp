#include "Layers.h"

// Test file for LSTM
int main () {

	typedef double REAL_t;
	typedef Mat<REAL_t> mat;
	using std::make_shared;

	LSTM<REAL_t> lstm(30, 50);
	Graph<REAL_t> G;

	auto embedding = make_shared<mat>(1000, 30, 2.0);

	auto prev_cell = make_shared<mat>(50, 1);
	auto prev_hidden = make_shared<mat>(50, 1);

	std::vector<int> indices = {0, 1, 10, 2, 1, 3};

	auto out = lstm.activate(G, G.rows_pluck(embedding, indices), prev_cell, prev_hidden);

	out.first->print();

	return 0;
}