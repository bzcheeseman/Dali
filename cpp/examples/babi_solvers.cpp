#include <iostream>
#include <vector>

#include "core/babi.h"

using std::vector;
using std::string;

class DumbModel: public babi::Model {
    public:

        void train(vector<babi::Story> data) {

        }

        void new_story() {

        }
        void fact(const vector<string>& fact) {

        }
        vector<string> question(const vector<string> quesiton) {
            return {};
        }
};

int main() {
    babi::benchmark<DumbModel>();
}
