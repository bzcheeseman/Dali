/*
Python style generators for C++ 11
@author Daniel Speyer
@source https://github.com/dspeyer/generators

An oasis of Python in lava sea of C++
*/

#ifndef DALI_UTILS_GENERATOR_H
#define DALI_UTILS_GENERATOR_H

#include <initializer_list>
#include <exception>
#include <functional>
#include <thread>
#include <mutex>

namespace utils {

    // subclass for handling forloop state internally.
    template<typename T>
    class ForLooping {
        private:
            T* gen;
        public:
            ForLooping(T* _gen) : gen(_gen) {};
            ForLooping<T>& operator++() {
                ++(*gen);
                return *this;
            }
            typename T::Output operator*(){
                return **gen;
            }
            operator bool() const{
                return gen && *gen;
            }
            operator T&() {
                return *gen;
            }
            bool operator!=(const ForLooping<T>& oth) {
                return static_cast<bool>(*this) != static_cast<bool>(oth);
            }
    };

    template<typename Heart>
    class Gen;

    template<typename OutputT>
    class GeneratorHeart {
        private:
            OutputT value;
            bool hasOutputted;
            std::shared_ptr<std::mutex> mutex;
            bool abort;
            class AbortException : public std::exception {};
        public:
            typedef OutputT Output;

            template<typename T>
            friend class Gen;

            GeneratorHeart() : hasOutputted(false), abort(false) {}

            void yield(OutputT v) {
                assert2((bool)mutex, "Mutex was not present during yield.");
                value = v;
                hasOutputted = true;
                while (hasOutputted) {
                    mutex->unlock();
                    std::this_thread::yield();
                    mutex->lock();
                }
                if (abort) {
                    throw AbortException();
                }
            }
    };

    template<typename Heart>
    class Gen {
        private:
            std::function<void()> callback;
            Heart heart;
            std::shared_ptr<std::mutex> mutex;
            std::shared_ptr<std::thread> thread_ptr;
            bool done;
        public:
            typedef typename Heart::Output Output;
            template<typename... ARGS>
            Gen(ARGS... args) :
                    done(false),
                    mutex(std::make_shared<std::mutex>()) {
                // Use bind instead of capture because capture does not get along with parameter packs
                // callback=std::bind([&](Gen<Heart>* gen, ARGS... args){ gen->threadmain(args...); }, this, args...);
                mutex->lock();
                heart.mutex = mutex;
                // pthread_create (&this->thread, NULL, generator_utils::invoke_callback, static_cast<void*>(&callback));
                thread_ptr = std::make_shared<std::thread>(
                    [this, args...]() {
                        threadmain(args...);
                    }
                );
                ++(*this);
            }
            ~Gen() {
                if (!done) {
                    heart.abort = true;
                    ++(*this);
                }
                thread_ptr->join();
            }
            template<typename... ARGS>
            void threadmain(ARGS... args) {
                mutex->lock();
                try {
                    heart.run(args...);
                } catch (typename Heart::AbortException ex) {}
                done = true;
                mutex->unlock();
            }
            operator bool() {
                return !done;
            }
            Gen<Heart>& operator++() {
                heart.hasOutputted = false;
                while (!heart.hasOutputted && !done) {
                    mutex->unlock();
                    std::this_thread::yield();
                    mutex->lock();
                }
                return *this;
            }
            Output operator*() {
                return heart.value;
            }
            typedef utils::ForLooping<Gen<Heart>> ForLooping;
            ForLooping begin() { return ForLooping(this); }
            ForLooping end() { return ForLooping(NULL); }
    };

    // Generator heart specifically tailored for lambdas.
    template<typename T>
    class LambdaGeneratorHeart : public GeneratorHeart<T> {
        public:
            typedef std::function<void(std::function<void(T)>)> generator_t;

            void run(generator_t generator) {
                generator(std::bind(&GeneratorHeart<T>::yield, this, std::placeholders::_1));
            }
    };

    template<typename T>
    using yield_t = std::function<void(T)>;

    /*
    Generator<T>
    ------------

    A wrapper around Gen<LambdaGeneratorHeart<T>>, a generator built from
    a lambda. This wrapper allows easy copying, moving, and resetting
    of a generator.

    */
    template<typename T>
    class Generator {
        public:
            typedef typename LambdaGeneratorHeart<T>::generator_t generator_t;
            typedef Gen<LambdaGeneratorHeart<T>> heart_t;
            typedef typename heart_t::ForLooping ForLooping;
            std::shared_ptr< heart_t > genheart;
            generator_t gen;

            Generator(generator_t _gen) : gen(_gen), genheart(NULL) {};
            Generator(const Generator<Gen<LambdaGeneratorHeart<T>>>& other) : gen(other.gen), genheart(NULL) {}

            void reset() {genheart = NULL;}

            ForLooping begin() {
                if (!genheart)
                    genheart = std::make_shared<heart_t>(gen);
                return genheart->begin();
            };

            ForLooping end() {
                if (!genheart)
                    genheart = std::make_shared<Gen<LambdaGeneratorHeart<T>>>(gen);
                return genheart->end();
            }
    };

    template<typename T>
    Generator<T> make_generator(typename LambdaGeneratorHeart<T>::generator_t generator) {
        return Generator<T>(generator);
    }
}

#endif
