#ifndef DALI_ARRAY_LAZY_EXPRESSION
#define DALI_ARRAY_LAZY_EXPRESSION

// inspired by tqchen's mshadow

class AssignableArray;

template<typename SubType>
struct Exp {
    inline const SubType& self() const {
        return *static_cast<const SubType*>(this);
    }
};

template<typename SubType>
struct RValueExp : Exp<SubType>{
    virtual AssignableArray as_assignable() const = 0;
};

#endif
