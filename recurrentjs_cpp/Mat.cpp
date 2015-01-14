#include <iostream>
#include <iomanip>
#include <random>


typedef double REAL_t;

// Utilities:

void fill_random(int n, REAL_t std, REAL_t * ptr) {
	REAL_t * end = ptr + n;
	std::default_random_engine generator;
	std::normal_distribution<REAL_t> distribution(0.0,std);
	while (ptr != end) {
		*ptr = distribution(generator);
		ptr++;
	}
}

void print_matrix(int n, int d, REAL_t * ptr) {
	REAL_t * end = ptr + n * d;
	int i = 0;
	std::cout << "[";
	while (ptr != end ) {
		std::cout << std::fixed
				  << std::setw( 7 )
				  << std::setprecision( 3 )
				  << std::setfill( ' ' )
				  << *ptr;
		i++;
		if (i == n * d) {
			std::cout << "]" << std::endl;
		} else {
			std::cout << " ";
		}
		if (i % d == 0 && i < n * d) {
			std::cout << "\n ";
		}
		ptr++;
	}
}

// Matrix class (with random initializer):

class Mat {
	int n;
	int d;
	REAL_t * w;
	REAL_t * dw;

	public:

	Mat (int n, int d) {
		this->n = n;
		this->d = d;
		this->w = (REAL_t *) calloc(n * d, sizeof(REAL_t));
		this->dw = (REAL_t *) calloc(n * d, sizeof(REAL_t));
	}

	Mat (int n, int d, REAL_t std) {
		this->n = n;
		this->d = d;
		this->w = (REAL_t *) malloc(n * d * sizeof(REAL_t));
		fill_random(n * d, std, this->w);
		this->dw = (REAL_t *) calloc(n * d, sizeof(REAL_t));
	}

	void print () {
		print_matrix(this->n, this->d, this->w);
	}

	~Mat() {
		free(this->w);
		free(this->dw);
	}

	static Mat RandMat(int n, int d, REAL_t std) {
		return Mat(n, d, std);
	}
};

int main()
{

    Mat mymatrix(3,5);

    Mat mymatrix2 = Mat::RandMat(3, 4, 2.0);

    mymatrix2.print();

    return 0;
}