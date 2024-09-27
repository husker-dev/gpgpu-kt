package com.huskerdev.gpkt.ast.types

import com.huskerdev.gpkt.ast.Expression
import com.huskerdev.gpkt.ast.FieldExpression
import com.huskerdev.gpkt.ast.compilationError
import com.huskerdev.gpkt.ast.expectedTypeException
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
    CONDITION(13, Usage.CONDITION),
    CAST(2, Usage.CAST),
    ;

    companion object {
        val sortedReverse = entries.sortedWith(compareBy(Operator::priority, Operator::name)).reversed()
    }

    enum class Usage {
        AxB,
        Ax,
        xB,
        FUNCTION,
        ARRAY_ACCESS,
        CAST,
        CONDITION;
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
            PLUS, MINUS, MULTIPLY, DIVIDE, MOD,
            PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN, MOD_ASSIGN
            -> if(!leftType.isNumber || !rightType.isNumber)
                throw compilationError("Operator '$token' can be only used with numeric types", operatorLexeme, codeBlock)

            BITWISE_AND_ASSIGN, BITWISE_OR_ASSIGN, BITWISE_XOR_ASSIGN, BITWISE_SHIFT_RIGHT_ASSIGN, BITWISE_SHIFT_LEFT_ASSIGN,
            BITWISE_AND, BITWISE_OR, BITWISE_XOR, BITWISE_NOT, BITWISE_SHIFT_RIGHT, BITWISE_SHIFT_LEFT
            -> if(!leftType.isInteger || !rightType.isInteger)
                throw compilationError("Operator '$token' can be only used with integer types", operatorLexeme, codeBlock)

            LOGICAL_AND, LOGICAL_OR
            -> if(!leftType.isLogical || !rightType.isLogical)
                throw compilationError("Operator '$token' can be only used with logical types", operatorLexeme, codeBlock)

            ASSIGN
            -> if(leftType != rightType && !Type.canAssignNumbers(leftType, rightType))
                throw expectedTypeException(leftType, rightType, rightLexeme, codeBlock)

            EQUAL, NOT_EQUAL, LESS, GREATER, LESS_OR_EQUAL, GREATER_OR_EQUAL
            -> if(leftType != rightType && !leftType.isNumber && !rightType.isNumber)
                throw compilationError("Operator '$token' can be only used with equal or numeric types", operatorLexeme, codeBlock)
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
            -> if (left !is FieldExpression || !leftType.isNumber)
                throw compilationError("Operator '$token' can be only used with variables", operatorLexeme, codeBlock)
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
                throw compilationError("Operator '$token' can be only used with numeric types", operatorLexeme, codeBlock)
            BITWISE_NOT
            -> if(!rightType.isInteger)
                throw compilationError("Operator '$token' can be only used with integer types", operatorLexeme, codeBlock)
            LOGICAL_NOT
            -> if(!rightType.isLogical)
                throw compilationError("Operator '$token' can be only used with logical types", operatorLexeme, codeBlock)
            else -> throw UnsupportedOperationException()
        }
    }

    /* ==================== *\
         Type transitions
    \* ==================== */

    fun operateTypeAxB(left: Type, right: Type) = when(this){
        PLUS, MINUS, MULTIPLY, DIVIDE, MOD,
        BITWISE_AND, BITWISE_OR, BITWISE_XOR, BITWISE_NOT,
        -> Type.mergeNumberTypes(left, right)

        PLUS_ASSIGN, MINUS_ASSIGN, MULTIPLY_ASSIGN, DIVIDE_ASSIGN, MOD_ASSIGN,
        BITWISE_AND_ASSIGN, BITWISE_OR_ASSIGN, BITWISE_XOR_ASSIGN, BITWISE_SHIFT_RIGHT_ASSIGN, BITWISE_SHIFT_LEFT_ASSIGN,
        BITWISE_SHIFT_RIGHT, BITWISE_SHIFT_LEFT
        -> left

        EQUAL, NOT_EQUAL, LESS, GREATER, LESS_OR_EQUAL, GREATER_OR_EQUAL,
        LOGICAL_AND, LOGICAL_OR
        -> Type.BOOLEAN

        ASSIGN
        -> if(left == right) left else Type.mergeNumberTypes(left, right)
        else -> throw UnsupportedOperationException()
    }

    fun operateTypeAx(left: Type) = when(this){
        INCREASE, DECREASE -> left
        else -> throw UnsupportedOperationException()
    }

    fun operateTypeXB(right: Type) = when(this){
        POSITIVE, NEGATIVE, BITWISE_NOT, LOGICAL_NOT -> right
        else -> throw UnsupportedOperationException()
    }
}