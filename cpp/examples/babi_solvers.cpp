#include <iostream>
#include <set>
#include <vector>

#include "core/babi.h"

using std::vector;
using std::string;
using std::set;

class DumbModel: public babi::Model {
    set<string> tokens;
    public:
        void train(vector<babi::Story> data) {
        }

        void new_story() {
            tokens.clear();
        }
        void fact(const vector<string>& fact) {
            for(const string& token: fact) {
                if (token.compare(".") == 0 || token.compare("?") == 0)
                    continue;
                tokens.insert(token);
            }
        }
        vector<string> question(const vector<string> quesiton) {
            string ans;
            int ans_idx = rand()%tokens.size();
            int current_idx = 0;
            for(auto& el: tokens) {
                if (current_idx == ans_idx)
                    ans = el;
                ++current_idx;
            }
            return {ans};
        }
};

int main() {
    babi::benchmark<DumbModel>();
}
