package com.huskerdev.gpkt.ast.types

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme


enum class Operator(
    val priority: Int,
    val usage: Usage,
    val token: String = ""
) {
    // Assignment (Math)
    ASSIGN(14, Usage.AxB, "="),
    PLUS_ASSIGN(14, Usage.AxB, "+="),
    MINUS_ASSIGN(14, Usage.AxB, "-="),
    MULTIPLY_ASSIGN(14, Usage.AxB, "*="),
    DIVIDE_ASSIGN(14, Usage.AxB, "/="),
    MOD_ASSIGN(14, Usage.AxB, "%="),

    // Assignment (Bitwise)
    BITWISE_AND_ASSIGN(14, Usage.AxB, "&="),
    BITWISE_OR_ASSIGN(14, Usage.AxB, "|="),
    BITWISE_XOR_ASSIGN(14, Usage.AxB, "^="),
    BITWISE_SHIFT_RIGHT_ASSIGN(14, Usage.AxB, ">>="),
    BITWISE_SHIFT_LEFT_ASSIGN(14, Usage.AxB, "<<="),

    // Increment/decrement
    INCREASE(2, Usage.Ax, "++"),
    DECREASE(2, Usage.Ax, "--"),

    // Math
    POSITIVE(2, Usage.xB, "+"),
    NEGATIVE(2, Usage.xB, "-"),
    PLUS(4, Usage.AxB, "+"),
    MINUS(4, Usage.AxB, "-"),
    MULTIPLY(3, Usage.AxB, "*"),
    DIVIDE(3, Usage.AxB, "/"),
    MOD(3, Usage.AxB, "%"),

    // Bitwise
    BITWISE_AND(8, Usage.AxB, "&"),
    BITWISE_OR(10, Usage.AxB, "|"),
    BITWISE_XOR(9, Usage.AxB, "^"),
    BITWISE_NOT(2, Usage.xB, "~"),
    BITWISE_SHIFT_RIGHT(5, Usage.AxB, ">>"),
    BITWISE_SHIFT_LEFT(5, Usage.AxB, "<<"),

    // Logical
    LOGICAL_NOT(2, Usage.xB, "!"),
    LOGICAL_AND(11, Usage.AxB, "&&"),
    LOGICAL_OR(12, Usage.AxB, "||"),

    // Comparison
    EQUAL(7, Usage.AxB, "=="),
    NOT_EQUAL(7, Usage.AxB, "!="),
    LESS(6, Usage.AxB, "<"),
    GREATER(6, Usage.AxB, ">"),
    LESS_OR_EQUAL(6, Usage.AxB, "<="),
    GREATER_OR_EQUAL(6, Usage.AxB, ">="),

    // Special cases
    ARRAY_ACCESS(1, Usage.ARRAY_ACCESS),
    FUNCTION(1, Usage.FUNCTION),
    FIELD(0, Usage.FIELD),  // Correct priority is 1, but it is used often, so set 0 for optimization
    CONDITION(13, Usage.CONDITION),
    CAST(2, Usage.CAST),
    NEW(-1, Usage.NEW),
    ;

    companion object {
        val sortedReverse = entries.sortedWith(compareBy(Operator::priority, Operator::name)).reversed()
    }

    enum class Usage {
        AxB,
        Ax,
        xB,
        FUNCTION,
        FIELD,
        ARRAY_ACCESS,
        CAST,
        CONDITION,
        NEW,
    }

    /* =================== *\
         Type conditions
    \* =================== */

    fun checkOpAxB(
        left: Expression,
        right: Expression,
        operatorLexeme: Lexeme,
        rightLexeme: Lexeme,
        codeBlock: String
    ) {
        val leftType = left.type
        val rightType = right.type
        when(this){
            PLUS, MINUS, MULTIPLY, DIVIDE, MOD
            -> if(!leftType.isNumber || !rightType.isNumber)
                throw operatorUsageException(this, "numeric types", operatorLexeme, codeBlock)

            BITWISE_AND, BITWISE_OR, BITWISE_XOR, BITWISE_NOT, BITWISE_SHIFT_RIGHT, BITWISE_SHIFT_LEFT
            -> if(!leftType.isInteger || !rightType.isInteger)
                throw operatorUsageException(this, "integer types", operatorLexeme, codeBlock)

            LOGICAL_AND, LOGICAL_OR
            -> if(!leftType.isLogical || !rightType.isLogical)
                throw operatorUsageException(this, "logical types", operatorLexeme, codeBlock)

            PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN, MOD_ASSIGN
            -> {
                if(!leftType.isNumber || !rightType.isNumber)
                    throw operatorUsageException(this, "numeric types", operatorLexeme, codeBlock)
                if(!left.canAssign())
                    throw constAssignException(operatorLexeme, codeBlock)
            }

            BITWISE_AND_ASSIGN, BITWISE_OR_ASSIGN, BITWISE_XOR_ASSIGN, BITWISE_SHIFT_RIGHT_ASSIGN, BITWISE_SHIFT_LEFT_ASSIGN
            -> {
                if(!leftType.isInteger || !rightType.isInteger)
                    throw operatorUsageException(this, "integer types", operatorLexeme, codeBlock)
                if(!left.canAssign())
                    throw constAssignException(operatorLexeme, codeBlock)
            }

            ASSIGN -> {
                if (leftType != rightType && !PrimitiveType.canAssignNumbers(leftType, rightType))
                    throw expectedTypeException(leftType, rightType, rightLexeme, codeBlock)
                if(!left.canAssign())
                    throw constAssignException(operatorLexeme, codeBlock)
            }

            EQUAL, NOT_EQUAL, LESS, GREATER, LESS_OR_EQUAL, GREATER_OR_EQUAL
            -> if(leftType != rightType && !leftType.isNumber && !rightType.isNumber)
                throw operatorUsageException(this, "equal or numeric types", operatorLexeme, codeBlock)
            else -> throw UnsupportedOperationException()
        }
    }

    fun checkOpAx(
        left: Expression,
        operatorLexeme: Lexeme,
        codeBlock: String
    ) {
        val leftType = left.type
        when (this) {
            INCREASE, DECREASE
            -> {
                if (left !is FieldExpression || !leftType.isNumber)
                    throw operatorUsageException(this, "variables", operatorLexeme, codeBlock)
                if(!left.canAssign())
                    throw constAssignException(operatorLexeme, codeBlock)
            }
            else -> throw UnsupportedOperationException()
        }
    }

    fun checkOpXB(
        right: Expression,
        operatorLexeme: Lexeme,
        codeBlock: String
    ) {
        val rightType = right.type
        when(this){
            POSITIVE, NEGATIVE
            -> if(!rightType.isNumber)
                throw operatorUsageException(this, "numeric types", operatorLexeme, codeBlock)
            BITWISE_NOT
            -> if(!rightType.isInteger)
                throw operatorUsageException(this, "integer types", operatorLexeme, codeBlock)
            LOGICAL_NOT
            -> if(!rightType.isLogical)
                throw operatorUsageException(this, "logical types", operatorLexeme, codeBlock)
            else -> throw UnsupportedOperationException()
        }
    }

    /* ==================== *\
         Type transitions
    \* ==================== */

    fun operateTypeAxB(left: PrimitiveType, right: PrimitiveType) = when(this){
        PLUS, MINUS, MULTIPLY, DIVIDE, MOD,
        BITWISE_AND, BITWISE_OR, BITWISE_XOR, BITWISE_NOT,
        -> PrimitiveType.mergeNumberTypes(left as SinglePrimitiveType<*>, right as SinglePrimitiveType<*>)

        PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN, MOD_ASSIGN,
        BITWISE_AND_ASSIGN, BITWISE_OR_ASSIGN, BITWISE_XOR_ASSIGN, BITWISE_SHIFT_RIGHT_ASSIGN, BITWISE_SHIFT_LEFT_ASSIGN,
        BITWISE_SHIFT_RIGHT, BITWISE_SHIFT_LEFT
        -> left

        EQUAL, NOT_EQUAL, LESS, GREATER, LESS_OR_EQUAL, GREATER_OR_EQUAL,
        LOGICAL_AND, LOGICAL_OR
        -> BOOLEAN

        ASSIGN -> if(left == right) left
            else PrimitiveType.mergeNumberTypes(left as SinglePrimitiveType<*>, right as SinglePrimitiveType<*>)
        else -> throw UnsupportedOperationException()
    }

    fun operateTypeAx(left: PrimitiveType) = when(this){
        INCREASE, DECREASE -> left
        else -> throw UnsupportedOperationException()
    }

    fun operateTypeXB(right: PrimitiveType) = when(this){
        POSITIVE, NEGATIVE, BITWISE_NOT, LOGICAL_NOT -> right
        else -> throw UnsupportedOperationException()
    }


    fun assignOpToSimple() = when(this){
        MOD_ASSIGN -> MOD
        BITWISE_AND_ASSIGN -> BITWISE_AND
        BITWISE_OR_ASSIGN -> BITWISE_OR
        BITWISE_XOR_ASSIGN -> BITWISE_XOR
        BITWISE_SHIFT_RIGHT_ASSIGN -> BITWISE_SHIFT_RIGHT
        BITWISE_SHIFT_LEFT_ASSIGN -> BITWISE_SHIFT_LEFT
        DIVIDE_ASSIGN -> DIVIDE
        MULTIPLY_ASSIGN -> MULTIPLY
        MINUS_ASSIGN -> MINUS
        PLUS_ASSIGN -> PLUS
        INCREASE -> PLUS
        DECREASE -> MINUS
        else -> throw UnsupportedOperationException()
    }
}