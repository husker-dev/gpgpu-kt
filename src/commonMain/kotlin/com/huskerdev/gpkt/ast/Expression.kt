package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPClass
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.ArrayPrimitiveType
import com.huskerdev.gpkt.ast.types.Operator
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.SinglePrimitiveType


interface Expression {
    val type: PrimitiveType
    val lexemeIndex: Int
    val lexemeLength: Int

    fun canAssign() = false
    fun clone(scope: GPScope): Expression
}


// Wrapped by brackets
class BracketExpression(
    val wrapped: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression {
    override val type = wrapped.type
    override fun clone(scope: GPScope) =
        BracketExpression(wrapped.clone(scope), lexemeIndex, lexemeLength)
}

// Array definition
class ArrayDefinitionExpression(
    val elements: Array<Expression>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression {
    override val type = (elements[0].type as SinglePrimitiveType<*>).toArray(elements.size)
    override fun clone(scope: GPScope) =
        ArrayDefinitionExpression(elements.map { it.clone(scope) }.toTypedArray(), lexemeIndex, lexemeLength)
}

// Class creation
class ClassCreationExpression(
    val classObj: GPClass,
    val arguments: List<Expression>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression {
    override val type = classObj.type
    override fun clone(scope: GPScope) =
        ClassCreationExpression(scope.findClass(classObj.name)!!, arguments.map { it.clone(scope) }, lexemeIndex, lexemeLength)
}

// ======================
//  Operator expressions
// ======================

abstract class OperatorExpression(val operator: Operator): Expression

// AxB
class AxBExpression(
    operator: Operator,
    override val type: PrimitiveType,
    val left: Expression,
    val right: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(operator){
    override fun clone(scope: GPScope) =
        AxBExpression(operator, type, left.clone(scope), right.clone(scope), lexemeIndex, lexemeLength)
}

// Ax
class AxExpression(
    operator: Operator,
    override val type: PrimitiveType,
    val left: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(operator) {
    override fun clone(scope: GPScope) =
        AxExpression(operator, type, left.clone(scope), lexemeIndex, lexemeLength)
}

// Ax
class XBExpression(
    operator: Operator,
    override val type: PrimitiveType,
    val right: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(operator) {
    override fun clone(scope: GPScope) =
        XBExpression(operator, type, right.clone(scope), lexemeIndex, lexemeLength)
}

// A[]
class ArrayAccessExpression(
    val array: Expression,
    val index: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): OperatorExpression(Operator.ARRAY_ACCESS) {
    override val type = (array.type as ArrayPrimitiveType<*>).single
    override fun canAssign() =
        array is FieldExpression && !array.field.isReadonly
    override fun clone(scope: GPScope) =
        ArrayAccessExpression(array.clone(scope), index.clone(scope), lexemeIndex, lexemeLength)
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
    override fun clone(scope: GPScope) =
        FunctionCallExpression(obj?.clone(scope), scope.findFunction(function.name)!!, arguments.map { it.clone(scope) })
}

// A
class FieldExpression(
    val obj: Expression?,
    val field: GPField,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression {
    override val type = field.type
    override fun canAssign() = !field.isConstant
    override fun clone(scope: GPScope) =
        FieldExpression(obj?.clone(scope), scope.findField(field.name)!!, lexemeIndex, lexemeLength)
}

// (type)A
class CastExpression(
    override val type: PrimitiveType,
    val right: Expression,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression {
    override fun clone(scope: GPScope) =
        CastExpression(type, right.clone(scope), lexemeIndex, lexemeLength)
}

// Const
data class ConstExpression(
    val lexeme: Lexeme,
    override val type: PrimitiveType,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Expression {
    override fun clone(scope: GPScope) =
        ConstExpression(lexeme, type, lexemeIndex, lexemeLength)
}