#ifndef DALI_UTILS_PRINT_UTILS_H
#define DALI_UTILS_PRINT_UTILS_H

#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// MACRO DEFINITIONS
#define ELOG(EXP) std::cout << #EXP "\t=\t" << (EXP) << std::endl
#define SELOG(STR,EXP) std::cout << #STR "\t=\t" << (EXP) << std::endl


template<typename T>
std::ostream& operator<<(std::ostream&, const std::vector<T>&);
std::ostream& operator<<(std::ostream&, const std::vector<std::string>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, uint>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, float>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, double>&);
std::ostream& operator<<(std::ostream&, const std::map<std::string, std::string>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, uint>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, float>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, double>&);
std::ostream& operator<<(std::ostream&, const std::unordered_map<std::string, std::string>&);

namespace utils {
    class MS {
        public:
            std::stringstream stream;
            operator std::string() const { return stream.str(); }

            template<class T>
            MS& operator<<(T const& VAR) { stream << VAR; return *this; }
    };

    template<typename T>
    std::string iter_to_str(T begin, T end) {
        std::stringstream ss;
        bool first = true;
        for (; begin != end; begin++) {
            if (!first)
                ss << ", ";
            ss << *begin;
            first = false;
        }
        return ss.str();
    }

    extern std::string green;
    extern std::string red;
    extern std::string blue;
    extern std::string yellow;
    extern std::string cyan;
    extern std::string black;
    extern std::string reset_color;
    extern std::string bold;
}


#endif
