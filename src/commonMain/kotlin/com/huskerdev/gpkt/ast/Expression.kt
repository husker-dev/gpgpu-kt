package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPClass
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.types.ArrayPrimitiveType
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.SinglePrimitiveType


abstract class Expression {
    abstract val type: PrimitiveType
    abstract val lexemeIndex: Int
    abstract val lexemeLength: Int

    open fun canAssign() = false
}


// Wrapped by brackets
class BracketExpression(
    val wrapped: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression() {
    override val type = wrapped.type
}

// Array definition
class ArrayDefinitionExpression(
    val elements: Array<Expression>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression() {
    override val type = (elements[0].type as SinglePrimitiveType<*>).toArray(elements.size)
}

// Class creation
class ClassCreationExpression(
    val classObj: GPClass,
    val arguments: List<Expression>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression() {
    override val type = classObj.type
}

// ======================
//  Operator expressions
// ======================

abstract class OperatorExpression(val operator: Operator): Expression()

// AxB
class AxBExpression(
    operator: Operator,
    override val type: PrimitiveType,
    val left: Expression,
    val right: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(operator)

// Ax
class AxExpression(
    operator: Operator,
    override val type: PrimitiveType,
    val left: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(operator)

// Ax
class XBExpression(
    operator: Operator,
    override val type: PrimitiveType,
    val right: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(operator)

// A[]
class ArrayAccessExpression(
    val array: Expression,
    val index: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(Operator.ARRAY_ACCESS) {
    override val type = (array.type as ArrayPrimitiveType<*>).single
    override fun canAssign() = array is FieldExpression && !array.field.isReadonly
}

// A()
class FunctionCallExpression(
    val obj: Expression?,
    val function: GPFunction,
    val arguments: List<Expression>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(Operator.FUNCTION) {
    override val type = function.returnType
}

// A
class FieldExpression(
    val obj: Expression?,
    val field: GPField,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression() {
    override val type = field.type
    override fun canAssign() = !field.isConstant
}

// (type)A
class CastExpression(
    override val type: PrimitiveType,
    val right: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression()

// Const
data class ConstExpression(
    val lexeme: Lexeme,
    override val type: PrimitiveType,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression()