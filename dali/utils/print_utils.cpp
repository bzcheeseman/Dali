#include "print_utils.h"

#include <iomanip>
#include <iterator>

using std::vector;
using std::string;
using std::stringstream;

namespace utils {
    // color codes: http://www.codebuilder.me/2014/01/color-terminal-text-in-c/
    std::string green       = "\033[32m";
    std::string red         = "\033[31m";
    std::string blue        = "\033[34m";
    std::string yellow      = "\033[33m";
    std::string cyan        = "\033[36m";
    std::string black       = "\033[30m";
    std::string reset_color = "\033[0m";
    std::string bold        = "\033[1m";
}

std::ostream &operator <<(std::ostream &os, const vector<string> &v) {
   if (v.size() == 0) return os << "[]";
   os << "[\"";
   std::copy(v.begin(), v.end() - 1, std::ostream_iterator<string>(os, "\", \""));
   return os << v.back() << "\"]";
}

std::ostream &operator <<(std::ostream &os, const std::map<string, uint> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}

std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, uint> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, float> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, double> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::map<string, float> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::map<string, double> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => " << kv.second << ",\n";
   }
   return os << "}";
}

std::ostream &operator <<(std::ostream &os, const std::map<string, string> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => \"" << kv.second << "\",\n";
   }
   return os << "}";
}
std::ostream &operator <<(std::ostream &os, const std::unordered_map<string, string> &v) {
   if (v.size() == 0) return os << "{}";
   os << "{\n";
   for (auto& kv : v) {
       os << "\"" << kv.first << "\" => \"" << kv.second << "\",\n";
   }
   return os << "}";
}

int needed_digits(int value) {
  if (value > 0) {
    return 1 + needed_digits(value / 10);
  } else {
    return 0;
  }
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& v) {
        if (v.size() == 0) return os << "[]";
        os << "[";
        size_t i = 0;
        for (auto& f : v) {
                os << std::fixed
                   << std::setw( 7 ) // keep 7 digits
                   << std::setprecision( 3 ) // use 3 decimals
                   << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
                   << f;
          if (i++ + 1 < v.size()) {
            os << ", ";
          }
        }
        return os << "]";
}

template<>
std::ostream& operator<<(std::ostream& os, const vector<int>& v) {
    if (v.size() == 0) return os << "[]";
    auto max_el = *std::max_element(v.begin(), v.end());
    auto min_el = *std::max_element(v.begin(), v.end());
    int digits = needed_digits(std::max(std::abs(max_el), std::abs(min_el)));
    os << "[";
    size_t i = 0;
    for (auto& f : v) {
            os << std::fixed
               << std::setw( digits ) // keep 7 digits
               << std::setfill( ' ' ) // pad values with blanks this->w(i,j)
               << f;
      if (i++ + 1 < v.size()) {
        os << ", ";
      }
    }
    return os << "]";
}

template std::ostream& operator<< <double>(std::ostream& strm, const vector<double>& a);
template std::ostream& operator<< <float>(std::ostream& strm, const vector<float>& a);
template std::ostream& operator<< <uint>(std::ostream& strm, const vector<uint>& a);
template std::ostream& operator<< <int>(std::ostream& strm, const vector<int>& a);
template std::ostream& operator<< <size_t>(std::ostream& strm, const vector<size_t>& a);
