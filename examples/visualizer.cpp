#include <iostream>
#include <redox.hpp>
#include <chrono>
#include <thread>

#include "dali/visualizer/visualizer.h"


int main(int argc, char* argv[]) {
    Visualizer v("babi");

    std::this_thread::sleep_for(std::chrono::seconds(3));
}
