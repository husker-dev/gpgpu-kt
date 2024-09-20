package com.huskerdev.gpkt.ast.types


const val FLAG_EQUAL_TYPES = 1
const val FLAG_NUMERIC_TYPES = 2
const val FLAG_LOGICAL_TYPES = 4
const val FLAG_FIELD_TYPE = 8
const val FLAG_INT_TYPE = 16

const val FLAG_RETURNS_BOOLEAN = 16


enum class Operator(
    val priority: Int,
    val usage: Usage,
    val token: String = "",
    val flags: Int = 0
) {
    // Assignment (Math)
    ASSIGN(14, Usage.AxB, "=", FLAG_EQUAL_TYPES),
    PLUS_ASSIGN(14, Usage.AxB, "+=", FLAG_NUMERIC_TYPES),
    MINUS_ASSIGN(14, Usage.AxB, "-=", FLAG_NUMERIC_TYPES),
    MULTIPLY_ASSIGN(14, Usage.AxB, "*=", FLAG_NUMERIC_TYPES),
    DIVIDE_ASSIGN(14, Usage.AxB, "/=", FLAG_NUMERIC_TYPES),
    MOD_ASSIGN(14, Usage.AxB, "%=", FLAG_NUMERIC_TYPES),

    // Assignment (Bitwise)
    BITWISE_AND_ASSIGN(14, Usage.AxB, "&=", FLAG_LOGICAL_TYPES),
    BITWISE_OR_ASSIGN(14, Usage.AxB, "|=", FLAG_LOGICAL_TYPES),
    BITWISE_XOR_ASSIGN(14, Usage.AxB, "^=", FLAG_LOGICAL_TYPES),
    BITWISE_SHIFT_RIGHT_ASSIGN(14, Usage.AxB, ">>=", FLAG_INT_TYPE),
    BITWISE_SHIFT_LEFT_ASSIGN(14, Usage.AxB, "<<=", FLAG_INT_TYPE),

    // Increment/decrement
    INCREASE(2, Usage.Ax, "++", FLAG_FIELD_TYPE),
    DECREASE(2, Usage.Ax, "--", FLAG_FIELD_TYPE),

    // Math
    POSITIVE(2, Usage.xB, "+", FLAG_NUMERIC_TYPES),
    NEGATIVE(2, Usage.xB, "-", FLAG_NUMERIC_TYPES),
    PLUS(4, Usage.AxB, "+", FLAG_NUMERIC_TYPES),
    MINUS(4, Usage.AxB, "-", FLAG_NUMERIC_TYPES),
    MULTIPLY(3, Usage.AxB, "*", FLAG_NUMERIC_TYPES),
    DIVIDE(3, Usage.AxB, "/", FLAG_NUMERIC_TYPES),
    MOD(3, Usage.AxB, "%", FLAG_NUMERIC_TYPES),

    // Bitwise
    BITWISE_AND(8, Usage.AxB, "&", FLAG_INT_TYPE),
    BITWISE_OR(10, Usage.AxB, "|", FLAG_INT_TYPE),
    BITWISE_XOR(9, Usage.AxB, "^", FLAG_INT_TYPE),
    BITWISE_NOT(2, Usage.xB, "~", FLAG_INT_TYPE),
    BITWISE_SHIFT_RIGHT(5, Usage.AxB, ">>", FLAG_INT_TYPE),
    BITWISE_SHIFT_LEFT(5, Usage.AxB, "<<", FLAG_INT_TYPE),

    // Logical
    LOGICAL_NOT(2, Usage.xB, "!", FLAG_LOGICAL_TYPES),
    LOGICAL_AND(11, Usage.AxB, "&&", FLAG_LOGICAL_TYPES),
    LOGICAL_OR(12, Usage.AxB, "||", FLAG_LOGICAL_TYPES),

    // Comparison
    EQUAL(7, Usage.AxB, "==", FLAG_EQUAL_TYPES or FLAG_RETURNS_BOOLEAN),
    NOT_EQUAL(7, Usage.AxB, "!=", FLAG_EQUAL_TYPES or FLAG_RETURNS_BOOLEAN),
    LESS(6, Usage.AxB, "<", FLAG_NUMERIC_TYPES or FLAG_RETURNS_BOOLEAN),
    GREATER(6, Usage.AxB, ">", FLAG_NUMERIC_TYPES or FLAG_RETURNS_BOOLEAN),
    LESS_OR_EQUAL(6, Usage.AxB, "<=", FLAG_NUMERIC_TYPES or FLAG_RETURNS_BOOLEAN),
    GREATER_OR_EQUAL(6, Usage.AxB, ">=", FLAG_NUMERIC_TYPES or FLAG_RETURNS_BOOLEAN),

    // Special cases
    ARRAY_ACCESS(1, Usage.ARRAY_ACCESS),
    FUNCTION(1, Usage.FUNCTION),
    CONDITION(13, Usage.CONDITION),
    CAST(2, Usage.CAST),
    ;

    companion object {
        val sorted = entries.sortedWith(compareBy(Operator::priority, Operator::name))
        val sortedReverse = sorted.reversed()
    }

    enum class Usage {
        AxB,
        Ax,
        xB,
        FUNCTION,
        ARRAY_ACCESS,
        CAST,
        CONDITION
    }
}