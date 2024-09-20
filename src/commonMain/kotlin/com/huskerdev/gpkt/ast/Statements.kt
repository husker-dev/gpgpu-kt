package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope


interface Statement{
    val lexemeIndex: Int
    val lexemeLength: Int
    val returns: Boolean
}

class ExpressionStatement(
    val expression: Expression
): Statement {
    override val lexemeIndex = expression.lexemeIndex
    override val lexemeLength = expression.lexemeLength + 1
    override val returns = false
}

class FunctionStatement(
    val function: Function,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = false
}

class EmptyStatement(
    override val lexemeIndex: Int,
    override val lexemeLength: Int
) : Statement {
    override val returns = false
}

class FieldStatement(
    val fields: List<Field>,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = false
}

class ReturnStatement(
    val expression: Expression?,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = true
}

class BreakStatement(
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = false
}

class ContinueStatement(
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = false
}

class IfStatement(
    val condition: Expression,
    val body: Scope,
    val elseBody: Scope?,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = body.returns && (elseBody == null || elseBody.returns)
}

class WhileStatement(
    val condition: Expression,
    val body: Scope,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = false
}

class ForStatement(
    val initialization: Statement,
    val condition: Statement,
    val iteration: Statement,
    val body: Scope,
    override val lexemeIndex: Int,
    override val lexemeLength: Int
): Statement {
    override val returns = false
}