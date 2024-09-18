package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope


interface Statement{
    val lexemeIndex: Int
    val lexemeLength: Int
}

class ExpressionStatement(
    val expression: Expression
): Statement {
    override val lexemeIndex = expression.lexemeIndex
    override val lexemeLength = expression.lexemeLength + 1
}

class FunctionStatement(
    val function: Function,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement

class EmptyStatement(
    override val lexemeIndex: Int,
    override val lexemeLength: Int
) : Statement

class FieldStatement(
    val fields: List<Field>,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement

class ReturnStatement(
    val expression: Expression?,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement

class IfStatement(
    val condition: Expression,
    val body: Scope,
    val elseBody: Scope?,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement

class WhileStatement(
    val condition: Expression,
    val body: Scope,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement

class ForStatement(
    val initialization: Statement,
    val condition: Statement,
    val iteration: Statement,
    val body: Scope,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement