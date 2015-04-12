#include <iostream>
#include <redox.hpp>
#include <chrono>
#include <thread>

#include "dali/visualizer/EventQueue.h"

using redox::Redox;
using namespace std::chrono;
/**
Simple hello world with Redis Server and Hi-Redis.

Here's the gist: we'e gonna set the key "hello" to "world"
in a redis server, and then retrieve it.

To run this example run a process with redis server:

```bash
redis-server
```
**/

int main(int argc, char* argv[]) {
    redox::Redox rdx;
    if(!rdx.connect("localhost", 6379)) return 1;

    rdx.set("hello", "world!");
    std::cout << "Hello, " << rdx.get("hello") << std::endl;

    rdx.disconnect();

    EventQueue eq;
    std::cout << "experiment 1" << std::endl;

    {
        auto handle = eq.run_every([]() {
            std::cout << "siema" << std::endl;
        }, milliseconds(1000));

        std::this_thread::sleep_for(seconds(3));
    }
    std::this_thread::sleep_for(seconds(3));
    std::cout << "experiment 1: done!" << std::endl;
    std::cout << "experiment 2" << std::endl;


    auto handle2 = eq.run_every([]() {
        std::cout << "printed every 5s" << std::endl;
    }, milliseconds(5000));

    std::this_thread::sleep_for(seconds(6));
    eq.push([] {
        std::cout << "printed 1s after previous guy" << std::endl;
    });

    std::this_thread::sleep_for(seconds(5));
    std::cout << "experiment 2: done!" << std::endl;
    return 0;
}
