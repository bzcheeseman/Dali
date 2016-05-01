#ifndef DALI_ARRAY_FUNCTION_OPERATOR_H
#define DALI_ARRAY_FUNCTION_OPERATOR_H
// define different ways assignment between two expressions
// can happen:
enum OPERATOR_T {
    OPERATOR_T_EQL = 0,/* =  */
    OPERATOR_T_ADD = 1,/* += */
    OPERATOR_T_SUB = 2,/* -= */
    OPERATOR_T_MUL = 3,/* *= */
    OPERATOR_T_DIV = 4/* /= */
};

#endif
