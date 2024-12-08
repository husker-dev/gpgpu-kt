package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.BOOLEAN
import com.huskerdev.gpkt.utils.Dictionary


fun parseForStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): ForStatement{
    var i = from + 1
    if(lexemes[i].text != "(")
        throw compilationError("Expected for body", lexemes[i], codeBlock)

    // Find closing bracket
    var r = i
    var brackets = 1
    while(brackets != 0 && r < to){
        val text = lexemes[++r].text
        if(text == "(" || text == "{" || text == "[") brackets++
        if(text == ")" || text == "}" || text == "]") brackets--
    }
    if(brackets != 0)
        throw compilationError("Expected ')'", lexemes.last(), codeBlock)

    // Head scope
    val headScopeStatement = parseScopeStatement(scope, lexemes, codeBlock, i+1, r, dictionary)
    val headScope = headScopeStatement.scopeObj
    i += headScopeStatement.lexemeLength + 2

    if(headScope.statements.size < 2)
        throw compilationError("Expected at least two statements", lexemes[i], codeBlock)

    // block 1
    val initialization = headScope.statements[0]
    val fields = if(initialization is FieldStatement)
        linkedMapOf(*initialization.fields.map { it.name to it }.toTypedArray())
    else linkedMapOf()

    // block 2
    val condition = headScope.statements[1]
    val conditionExpression = when {
        condition is ExpressionStatement && condition.expression.type != BOOLEAN ->
            throw expectedTypeException(BOOLEAN, condition.expression.type, lexemes[condition.lexemeIndex], codeBlock)
        condition is EmptyStatement -> null
        condition is ExpressionStatement -> condition.expression
        else -> throw compilationError("Condition must be 'boolean' or empty", lexemes[condition.lexemeIndex], codeBlock)
    }

    // block 3
    val iteration = headScope.statements.getOrNull(2)
    val iterationExpression = when (iteration) {
        null -> null
        !is ExpressionStatement ->
            throw compilationError("Iteration must be expression or empty", lexemes[condition.lexemeIndex], codeBlock)
        else -> iteration.expression
    }

    // body
    val iterableScope = GPScope(scope.context, scope, dictionary = scope.dictionary, iterable = true, fields = fields)
    val body = parseStatement(iterableScope, lexemes, codeBlock, i, to, dictionary)
    i += body.lexemeLength

    return ForStatement(scope, initialization, conditionExpression, iterationExpression, body, from, i - from)
}