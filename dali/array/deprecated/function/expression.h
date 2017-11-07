// #ifndef DALI_ARRAY_LAZY_EXPRESSION
// #define DALI_ARRAY_LAZY_EXPRESSION

// // inspired by tqchen's mshadow
// template<typename OutType>
// struct Assignable;

// template<typename SubType>
// struct Exp {
//     inline const SubType& self() const {
//         return *static_cast<const SubType*>(this);
//     }
// };

// // this is used to quickly check that a class is a lazy expression
// // e.g. std::is_base_of<LazyExpType, Derived>::value
// // it is necessary, if we tried to use LazyExp directly, we would not
// // know the template argument for it.
// struct LazyExpType {};

// template<typename SubType>
// struct LazyExp : Exp<SubType>, LazyExpType {
// };

// #endif
