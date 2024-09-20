package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseForStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
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
    val headScope = Scope(scope, Type.VOID)
    i = parseScope(headScope, lexemes, codeBlock, i+1, r) + 1

    if(headScope.statements.size < 2)
        throw compilationError("Expected at least two statements", lexemes[i], codeBlock)

    // block 1
    val initialization = headScope.statements[0]

    // block 2
    val condition = headScope.statements[1]
    if(condition !is ExpressionStatement && condition !is EmptyStatement)
        throw compilationError("Condition must be 'boolean' expression or empty", lexemes[condition.lexemeIndex], codeBlock)
    if(condition is ExpressionStatement && condition.expression.type != Type.BOOLEAN)
        throw expectedTypeException(Type.BOOLEAN, condition.expression.type, lexemes[condition.lexemeIndex], codeBlock)

    // block 3
    val iteration = headScope.statements.getOrElse(2) {
        EmptyStatement(scope, condition.lexemeIndex + condition.lexemeLength, 0)
    }

    // body
    val body = Scope(scope, iterable = true).apply {
        if(initialization is FieldStatement)
            initialization.fields.forEach { addField(it, lexemes[i], codeBlock) }
        i = if(lexemes[i].text == "{")
            parseScope(this, lexemes, codeBlock, i + 1, to)
        else parseScope(this, lexemes, codeBlock, i, findExpressionEnd(i, lexemes, codeBlock))
    }

    return ForStatement(scope, initialization, condition, iteration, body, from, i - from)
}