#ifndef DALI_UTILS_HASH_UTILS_H
#define DALI_UTILS_HASH_UTILS_H

#include <functional>

typedef uint64_t hash_t;

namespace utils {

    class Hasher {
        hash_t value_;
        public:
            Hasher(hash_t seed=0);

            template<typename T>
            Hasher& add(const T& element) {
                value_ ^= std::hash<T>()(element) + 0x9e3779b9 + (value_<<6) + (value_>>2);
                return *this;
            }

            hash_t value() const;
    };

    template<typename T>
    size_t get_hash(const T& obj)  {
        return std::hash<T>()(obj);
    }
}  // namespace utils

#endif  // DALI_UTILS_HASH_UTILS_H
