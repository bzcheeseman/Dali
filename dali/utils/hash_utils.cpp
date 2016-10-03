#include "hash_utils.h"

namespace utils {
    Hasher::Hasher(hash_t seed) : value_(seed) {}


    hash_t Hasher::value() const {
        return value_;
    }
}  // namespace utils
