#ifndef DALI_UTILS_TUPLE_HASH_H
#define DALI_UTILS_TUPLE_HASH_H

#include <tuple>

namespace std {
    template<typename... TTypes>
    class hash<std::tuple<TTypes...>> {
        private:
            typedef std::tuple<TTypes...> Tuple;

            template<int N>
            size_t operator()(Tuple value) const {
                return 0;
            }

            template<int N, typename THead, typename... TTail>
            size_t operator()(Tuple value) const {
                constexpr int Index = N - sizeof...(TTail) - 1;
                return hash<THead>()(std::get<Index>(value)) ^ operator()<N, TTail...>(value);
            }

        public:
            size_t operator()(Tuple value) const {
                return operator()<sizeof...(TTypes), TTypes...>(value);
            }
    };
}

namespace utils {
    template<typename T>
    size_t get_hash(const T& obj) {
        return std::hash<T>()(obj);
    }
}

#endif
