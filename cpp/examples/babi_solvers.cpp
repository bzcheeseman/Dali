#include <iostream>

#include "core/babi.h"


int main() {
    for(auto& example: babi::Parser::tasks()) {
        std::cout << example << std::endl;
    }
    auto task1 = babi::Parser::tasks()[17];
    for(auto& story : babi::Parser::training_data(task1, 10)) {
        for (auto& item: story) {
            std::cout << *item << std::endl;
        }
    }
}
