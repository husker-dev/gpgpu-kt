package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseWhileStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): WhileStatement {
    var i = from + 1
    if(lexemes[i].text != "(")
        throw compilationError("Expected condition", lexemes[i], codeBlock)

    val condition = parseExpression(scope, lexemes, codeBlock, i+1)!!
    if(!condition.type.isLogical)
        throw expectedTypeException(Type.BOOLEAN, condition.type, lexemes[i+1], codeBlock)
    i += condition.lexemeLength + 2

    val body = Scope(scope, iterable = true).apply {
        i = if(lexemes[i].text == "{")
            parseScope(this, lexemes, codeBlock, i+1, lexemes.size)
        else parseScope(this, lexemes, codeBlock, i, findExpressionEnd(lexemes, i)) + 1
    }
    return WhileStatement(condition, body, from, i - from)
}