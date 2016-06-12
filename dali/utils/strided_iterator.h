#ifndef DALI_UTILS_STRIDED_ITERATOR_H
#define DALI_UTILS_STRIDED_ITERATOR_H

/*
Strided Iterator
================

Iterator that can traverse memory in non-contiguous fashion.

## Example Usage

	std::vector<int> elements = {1, 2, 3, 4, 5};
	auto begin = strided_iterator<int>(elements.data(), 2);
	auto end = begin + 3;
	// reverse
	std::sort(begin, end, [](const int& l, const int& r) {return l > r;});

	for (int i = 0; i < elements.size(); i++) {
		std::cout << i << " = " << elements[i]  << std::endl;
	}

This should now print 5, 2, 3, 4, 1, reversing only the numbers 1,
3, and 5.

*/

namespace utils {
	template<typename T>
    class strided_iterator {
        public:
            typedef strided_iterator self_type;
            typedef T value_type;
            typedef T& reference;
            typedef T* pointer;
            typedef std::forward_iterator_tag iterator_category;
            typedef int difference_type;
            strided_iterator(pointer ptr_, int stride_) : ptr(ptr_), stride(stride_) { }
            self_type operator++(int junk) { self_type i = *this; ptr = ptr + stride; return i; }
            self_type operator--(int junk) { self_type i = *this; ptr = ptr - stride; return i; }
            self_type& operator++() { ptr = ptr + stride; return *this; }
            self_type& operator--() { ptr = ptr - stride; return *this; }
            self_type& operator+=(const int& value) {ptr = ptr + stride * value; return *this;}
            self_type& operator-=(const int& value) {ptr = ptr - stride * value; return *this;}
            reference operator*() { return *ptr; }
            pointer operator->() { return ptr; }
            bool operator==(const self_type& rhs) { return ptr == rhs.ptr; }
            bool operator!=(const self_type& rhs) { return ptr != rhs.ptr; }

            pointer ptr;
            int stride;
        private:

    };

    template<typename T>
    class const_strided_iterator {
        public:
            typedef const_strided_iterator self_type;
            typedef T value_type;
            typedef T& reference;
            typedef T* pointer;
            typedef int difference_type;
            typedef std::forward_iterator_tag iterator_category;
            const_strided_iterator(pointer ptr_, int stride_) : ptr(ptr_), stride(stride_) { }
            self_type operator++(int junk) { self_type i = *this; ptr = ptr + stride; return i; }
            self_type operator--(int junk) { self_type i = *this; ptr = ptr - stride; return i; }
            self_type& operator+=(const int& value) {ptr = ptr + stride * value; return *this;}
            self_type& operator-=(const int& value) {ptr = ptr - stride * value; return *this;}
            self_type& operator++() { ptr = ptr + stride; return *this; }
            self_type& operator--() { ptr = ptr - stride; return *this; }
            reference operator*() { return *ptr; }
            const pointer operator->() { return ptr; }
            bool operator==(const self_type& rhs) { return ptr == rhs.ptr; }
            bool operator!=(const self_type& rhs) { return ptr != rhs.ptr; }

            pointer ptr;
            int stride;
        private:

    };

    template<typename T>
    int operator-(const const_strided_iterator<T>& left, const const_strided_iterator<T>& right) {
        return (left.ptr - right.ptr);
    }

    template<typename T>
    int operator-(const strided_iterator<T>& left, const strided_iterator<T>& right) {
        return (left.ptr - right.ptr);
    }

    template<typename T>
    bool operator>=(const strided_iterator<T>& left, const strided_iterator<T>& right) {
        if (left.stride < 0) {
            return !(left.ptr >= right.ptr);
        } else {
            return left.ptr >= right.ptr;
        }
    }

    template<typename T>
    bool operator>=(const const_strided_iterator<T>& left, const const_strided_iterator<T>& right) {
        if (left.stride < 0) {
            return !(left.ptr >= right.ptr);
        } else {
            return left.ptr >= right.ptr;
        }
    }

    template<typename T>
    bool operator<(const strided_iterator<T>& left, const strided_iterator<T>& right) {
        if (left.stride < 0) {
            return !(left.ptr < right.ptr);
        } else {
            return left.ptr < right.ptr;
        }
    }

    template<typename T>
    bool operator<(const const_strided_iterator<T>& left, const const_strided_iterator<T>& right) {
        if (left.stride < 0) {
            return !(left.ptr < right.ptr);
        } else {
            return left.ptr < right.ptr;
        }
    }

    template<typename T>
    bool operator>(const strided_iterator<T>& left, const strided_iterator<T>& right) {
        if (left.stride < 0) {
            return !(left.ptr > right.ptr);
        } else {
            return left.ptr > right.ptr;
        }
    }

    template<typename T>
    bool operator>(const const_strided_iterator<T>& left, const const_strided_iterator<T>& right) {
        if (left.stride < 0) {
            return !(left.ptr > right.ptr);
        } else {
            return left.ptr > right.ptr;
        }
    }

    template<typename T>
    const_strided_iterator<T> operator+(const const_strided_iterator<T>& iter, const int& value) {
        auto ret = iter;
        ret += value;
        return ret;
    }

    template<typename T>
    strided_iterator<T> operator+(const strided_iterator<T>& iter, const int& value) {
        auto ret = iter;
        ret += value;
        return ret;
    }

    template<typename T>
    const_strided_iterator<T> operator-(const const_strided_iterator<T>& iter, const int& value) {
        auto ret = iter;
        ret -= value;
        return ret;
    }

    template<typename T>
    strided_iterator<T> operator-(const strided_iterator<T>& iter, const int& value) {
        auto ret = iter;
        ret -= value;
        return ret;
    }
}  // namespace utils

#endif
