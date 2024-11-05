package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.objects.*


interface Statement{
    val scope: GPScope
    val lexemeIndex: Int
    val lexemeLength: Int
    val returns: Boolean
}

class ScopeStatement(
    override val scope: GPScope,
    val statements: MutableList<Statement> = mutableListOf(),
    override val returns: Boolean = false,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement

class ExpressionStatement(
    override val scope: GPScope,
    val expression: Expression
): Statement {
    override val lexemeIndex = expression.lexemeIndex
    override val lexemeLength = expression.lexemeLength + 1
    override val returns = false
}

class ImportStatement(
    override val scope: GPScope,
    val import: Import,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

open class FunctionStatement(
    override val scope: GPScope,
    val function: GPFunction,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class FunctionDefinitionStatement(
    scope: GPScope,
    function: GPFunction,
    lexemeIndex: Int = 0,
    lexemeLength: Int = 0
): FunctionStatement(scope, function, lexemeIndex, lexemeLength)

class EmptyStatement(
    override val scope: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
) : Statement {
    override val returns = false
}

class FieldStatement(
    override val scope: GPScope,
    val fields: List<GPField>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ReturnStatement(
    override val scope: GPScope,
    val expression: Expression?,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = true
}

class BreakStatement(
    override val scope: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ContinueStatement(
    override val scope: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class IfStatement(
    override val scope: GPScope,
    val condition: Expression,
    val body: Statement,
    val elseBody: Statement?,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = body.returns && (elseBody == null || elseBody.returns)
}

class WhileStatement(
    override val scope: GPScope,
    val condition: Expression,
    val body: Statement,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ForStatement(
    override val scope: GPScope,
    val initialization: Statement,
    val condition: Expression?,
    val iteration: Expression?,
    val body: Statement,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}

class ClassStatement(
    override val scope: GPScope,
    val classObj: GPClass,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
}