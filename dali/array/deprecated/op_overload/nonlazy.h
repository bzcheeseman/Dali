#ifndef DALI_ARRAY_OP_OVERLOAD_NONLAZY_H

#define DALI_DECLARE_ARRAY_INTERACTION(SYMBOL)\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const Array& right);\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const expression::ExpressionGraph& right);\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const Array& right);\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const expression::ExpressionGraph& right);

#define DALI_DECLARE_SCALAR_INTERACTION(SYMBOL)\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const double& right);\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const float& right);\
    expression::ExpressionGraph operator SYMBOL (const Array& left, const int& right);\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const double& right);\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const float& right);\
    expression::ExpressionGraph operator SYMBOL (const expression::ExpressionGraph& left, const int& right);\
    expression::ExpressionGraph operator SYMBOL (const double& left, const Array& right);\
    expression::ExpressionGraph operator SYMBOL (const float& left, const Array& right);\
    expression::ExpressionGraph operator SYMBOL (const int& left, const Array& right);\
    expression::ExpressionGraph operator SYMBOL (const double& left, const expression::ExpressionGraph& right);\
    expression::ExpressionGraph operator SYMBOL (const float& left, const expression::ExpressionGraph& right);\
    expression::ExpressionGraph operator SYMBOL (const int& left, const expression::ExpressionGraph& right);

DALI_DECLARE_ARRAY_INTERACTION(+);
DALI_DECLARE_ARRAY_INTERACTION(-);
DALI_DECLARE_ARRAY_INTERACTION(*);
DALI_DECLARE_ARRAY_INTERACTION(/);

DALI_DECLARE_SCALAR_INTERACTION(-);
DALI_DECLARE_SCALAR_INTERACTION(+);
DALI_DECLARE_SCALAR_INTERACTION(*);
DALI_DECLARE_SCALAR_INTERACTION(/);

expression::ExpressionGraph operator-(const Array& in);
expression::ExpressionGraph operator-(const expression::ExpressionGraph& in);

#endif
