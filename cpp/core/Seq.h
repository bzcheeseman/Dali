#ifndef CORE_SEQ_H
#define CORE_SEQ_H

#include <vector>

#include "core/Mat.h"


template<typename T>
class Seq {
    private:
        std::vector<T> seq;
    public:
        typedef typename std::vector<T>::iterator SeqIter;
        typedef typename std::vector<T>::const_iterator SeqConstIter;
        typedef typename std::vector<T>::reverse_iterator RSeqIter;
        typedef typename std::vector<T>::const_reverse_iterator RSeqConstIter;



        void push_back(T el);

        std::size_t size() const;

        virtual T& operator[](std::size_t idx);

        virtual T operator[](std::size_t idx) const;

        SeqIter begin();
        SeqIter end();
        SeqConstIter begin() const;
        SeqConstIter end() const;

        RSeqIter rbegin();
        RSeqIter rend();
        RSeqConstIter rbegin() const;
        RSeqConstIter rend() const;

        SeqConstIter cbegin() const;
        SeqConstIter cend() const;
        void insert(SeqIter where, SeqConstIter begin, SeqConstIter end);
        void insert(SeqIter where, RSeqConstIter begin, RSeqConstIter end);

};



#endif
