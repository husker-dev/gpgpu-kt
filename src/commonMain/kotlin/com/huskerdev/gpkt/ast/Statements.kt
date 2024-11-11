package com.huskerdev.gpkt.ast

import com.huskerdev.gpkt.ast.objects.*


interface Statement{
    val scope: GPScope?
    val lexemeIndex: Int
    val lexemeLength: Int
    val returns: Boolean

    fun clone(scope: GPScope): Statement
}

class ScopeStatement(
    override val scope: GPScope?,
    val scopeObj: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns: Boolean = scopeObj.returns
    override fun clone(scope: GPScope) =
        ScopeStatement(this.scope, scopeObj.clone(scope), lexemeIndex, lexemeLength)
}

class ExpressionStatement(
    override val scope: GPScope,
    val expression: Expression
): Statement {
    override val lexemeIndex = expression.lexemeIndex
    override val lexemeLength = expression.lexemeLength + 1
    override val returns = false
    override fun clone(scope: GPScope) =
        ExpressionStatement(scope, expression.clone(scope))
}

class ImportStatement(
    override val scope: GPScope,
    val import: Import,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        ImportStatement(scope, import, lexemeIndex, lexemeLength)
}

open class FunctionStatement(
    override val scope: GPScope,
    val function: GPFunction,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        FunctionStatement(scope, function.clone(scope), lexemeIndex, lexemeLength)
}

class FunctionDefinitionStatement(
    scope: GPScope,
    function: GPFunction,
    lexemeIndex: Int = 0,
    lexemeLength: Int = 0
): FunctionStatement(scope, function, lexemeIndex, lexemeLength){
    override fun clone(scope: GPScope) =
        FunctionDefinitionStatement(scope, function.clone(scope), lexemeIndex, lexemeLength)
}

class EmptyStatement(
    override val scope: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
) : Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        EmptyStatement(scope, lexemeIndex, lexemeLength)
}

class FieldStatement(
    override val scope: GPScope,
    val fields: List<GPField>,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        FieldStatement(scope, fields.map { it.clone(scope) }, lexemeIndex, lexemeLength)
}

class ReturnStatement(
    override val scope: GPScope,
    val expression: Expression?,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = true
    override fun clone(scope: GPScope) =
        ReturnStatement(scope, expression, lexemeIndex, lexemeLength)
}

class BreakStatement(
    override val scope: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        BreakStatement(scope, lexemeIndex, lexemeLength)
}

class ContinueStatement(
    override val scope: GPScope,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        ContinueStatement(scope, lexemeIndex, lexemeLength)
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
    override fun clone(scope: GPScope) =
        IfStatement(scope, condition, body, elseBody, lexemeIndex, lexemeLength)
}

class WhileStatement(
    override val scope: GPScope,
    val condition: Expression,
    val body: Statement,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        WhileStatement(scope, condition, body, lexemeIndex, lexemeLength)
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
    override fun clone(scope: GPScope) =
        ForStatement(scope, initialization, condition, iteration, body, lexemeIndex, lexemeLength)
}

class ClassStatement(
    override val scope: GPScope,
    val classObj: GPClass,
    override val lexemeIndex: Int = 0,
    override val lexemeLength: Int = 0
): Statement {
    override val returns = false
    override fun clone(scope: GPScope) =
        ClassStatement(scope, classObj, lexemeIndex, lexemeLength)
}