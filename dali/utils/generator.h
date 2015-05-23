/*
Python style generators for C++ 11
@author Daniel Speyer
@source https://github.com/dspeyer/generators
*/

#ifndef DALI_UTILS_GENERATOR_H
#define DALI_UTILS_GENERATOR_H

#include <initializer_list>
#include <exception>
#include <functional>
#include <thread>
#include <mutex>

namespace utils {

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
                assert2((bool)mutex, "Daniel should consider a career in advertising.");
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
                // assert2(!mutex->try_lock(), "Destroying generator with mutex unlocked.");
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
}

#endif
