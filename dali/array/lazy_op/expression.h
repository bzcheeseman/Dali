#ifndef DALI_ARRAY_LAZY_OP_EXPRESSION
#define DALI_ARRAY_LAZY_OP_EXPRESSION

// inspired by tqchen's mshadow

template<typename SubType>
struct Exp {
    inline const SubType& self() const {
        return *static_cast<const SubType*>(this);
    }
};

#endif
