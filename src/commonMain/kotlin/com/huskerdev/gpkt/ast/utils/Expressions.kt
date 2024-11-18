@file:Suppress("unused")
package com.huskerdev.gpkt.ast.utils

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.*

fun constIntExpr(value: String) =
    ConstExpression(Lexeme(value, Lexeme.Type.INT), INT)

fun constFloatExpr(value: String) =
    ConstExpression(Lexeme(value, Lexeme.Type.FLOAT), FLOAT)

fun constBooleanExpr(value: String) =
    ConstExpression(Lexeme(value, Lexeme.Type.LOGICAL), BOOLEAN)

fun Expression.toStatement(scope: GPScope) =
    ExpressionStatement(scope, this)

private fun axbExpr(operator: Operator, left: Expression, right: Expression) = AxBExpression(
    operator = operator,
    type = PrimitiveType.mergeNumberTypes(
        left.type as SinglePrimitiveType<*>,
        right.type as SinglePrimitiveType<*>
    ),
    left = left,
    right = right
)


operator fun Expression.div(other: Expression) =
    axbExpr(Operator.DIVIDE, this, other)

infix fun Expression.divAssign(other: Expression) =
    axbExpr(Operator.DIVIDE_ASSIGN, this, other)

operator fun Expression.times(other: Expression) =
    axbExpr(Operator.MULTIPLY, this, other)

operator fun Expression.plus(other: Expression) =
    axbExpr(Operator.PLUS, this, other)

operator fun Expression.minus(other: Expression) =
    axbExpr(Operator.MINUS, this, other)

operator fun Expression.rem(other: Expression) =
    axbExpr(Operator.MOD, this, other)

infix fun Expression.equal(other: Expression) =
    axbExpr(Operator.EQUAL, this, other)

infix fun Expression.notEqual(other: Expression) =
    axbExpr(Operator.NOT_EQUAL, this, other)

infix fun Expression.assign(other: Expression) =
    axbExpr(Operator.ASSIGN, this, other)

fun Expression.brackets() = BracketExpression(this)

fun funcCall(func: GPFunction, vararg args: Expression) = FunctionCallExpression(
    obj = null,
    function = func,
    arguments = args.toList()
)