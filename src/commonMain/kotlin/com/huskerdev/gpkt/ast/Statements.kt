package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Import
import com.huskerdev.gpkt.ast.objects.Scope


interface Statement{
    val scope: Scope
    val lexemeIndex: Int
    val lexemeLength: Int
    val returns: Boolean
}

class ScopeStatement(
    override val scope: Scope,
    val statements: MutableList<Statement> = mutableListOf(),
    override val returns: Boolean,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement

class ExpressionStatement(
    override val scope: Scope,
    val expression: Expression
): Statement {
    override val lexemeIndex = expression.lexemeIndex
    override val lexemeLength = expression.lexemeLength + 1
    override val returns = false
}

class ImportStatement(
    override val scope: Scope,
    val import: Import,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class FunctionStatement(
    override val scope: Scope,
    val function: Function,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class EmptyStatement(
    override val scope: Scope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
) : Statement {
    override val returns = false
}

class FieldStatement(
    override val scope: Scope,
    val fields: List<Field>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ReturnStatement(
    override val scope: Scope,
    val expression: Expression?,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = true
}

class BreakStatement(
    override val scope: Scope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ContinueStatement(
    override val scope: Scope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class IfStatement(
    override val scope: Scope,
    val condition: Expression,
    val body: Statement,
    val elseBody: Statement?,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = body.returns && (elseBody == null || elseBody.returns)
}

class WhileStatement(
    override val scope: Scope,
    val condition: Expression,
    val body: Statement,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ForStatement(
    override val scope: Scope,
    val initialization: Statement,
    val condition: Expression?,
    val iteration: Expression?,
    val body: Statement,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}