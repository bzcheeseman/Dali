#ifndef DALI_UTILS_TUPLE_HASH_H
#define DALI_UTILS_TUPLE_HASH_H

#include <tuple>

namespace std {
    namespace {
        // Code from boost
        // Reciprocal of the golden ratio helps spread entropy
        //     and handles duplicates.
        // See Mike Seymour in magic-numbers-in-boosthash-combine:
        //     http://stackoverflow.com/questions/4948780
        template <class T>
        inline void hash_combine(std::size_t& seed, T const& v) {
            seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        // Recursive template code derived from Matthieu M.
        template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
        struct HashValueImpl {
            static void apply(size_t& seed, Tuple const& tuple) {
                HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
                hash_combine(seed, std::get<Index>(tuple));
            }
        };

        template <class Tuple>
        struct HashValueImpl<Tuple,0> {
            static void apply(size_t& seed, Tuple const& tuple) {
                hash_combine(seed, std::get<0>(tuple));
            }
        };
    }

    template<typename... TTypes>
    struct hash<std::tuple<TTypes...>> {
        size_t operator()(const std::tuple<TTypes...>& value) const {
            size_t seed = 0;
            HashValueImpl<std::tuple<TTypes...>>::apply(seed, value);
            return seed;
        }
    };
}

namespace utils {
    template<typename T>
    size_t get_hash(const T& obj) {
        return std::hash<T>()(obj);
    }
}

#endif
