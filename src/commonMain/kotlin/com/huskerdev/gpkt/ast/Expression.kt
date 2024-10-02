package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.Type


abstract class Expression {
    abstract val type: Type
    abstract val lexemeIndex: Int
    abstract val lexemeLength: Int

    fun canAssign() = when {
        this is FieldExpression && field.isConstant -> false
        this is ArrayAccessExpression && array.isReadonly -> false
        else -> true
    }
}


// Wrapped by brackets
class BracketExpression(
    val wrapped: Expression,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Expression() {
    override val type = wrapped.type
}

// ======================
//  Operator expressions
// ======================

abstract class OperatorExpression(val operator: Operator): Expression()

// AxB
class AxBExpression(
    operator: Operator,
    override val type: Type,
    val left: Expression,
    val right: Expression,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): OperatorExpression(operator)

// Ax
class AxExpression(
    operator: Operator,
    override val type: Type,
    val left: Expression,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): OperatorExpression(operator)

// Ax
class XBExpression(
    operator: Operator,
    override val type: Type,
    val right: Expression,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): OperatorExpression(operator)

// A[]
class ArrayAccessExpression(
    val array: Field,
    val index: Expression,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): OperatorExpression(Operator.ARRAY_ACCESS) {
    override val type = Type.toSingleType(array.type)
}

// A()
class FunctionCallExpression(
    operator: Operator,
    val function: Function,
    val arguments: List<Expression>,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): OperatorExpression(operator) {
    override val type = function.returnType
}

// A
class FieldExpression(
    val field: Field,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Expression() {
    override val type = field.type
}

// (type)A
class CastExpression(
    override val type: Type,
    val right: Expression,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Expression()

// Const
data class ConstExpression(
    val lexeme: Lexeme,
    override val type: Type,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Expression()