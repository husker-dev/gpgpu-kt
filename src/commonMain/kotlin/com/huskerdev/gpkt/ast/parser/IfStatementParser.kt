package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseIfStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): IfStatement {
    var i = from + 1
    if(lexemes[i].text != "(")
        throw compilationError("Expected condition", lexemes[i], codeBlock)

    val condition = parseExpression(scope, lexemes, codeBlock, i+1)!!
    if(!condition.type.isLogical)
        throw expectedTypeException(Type.BOOLEAN, condition.type, lexemes[i+1], codeBlock)
    i += condition.lexemeLength + 2

    fun parseBlock() = Scope(scope).apply {
        i = if(lexemes[i].text == "{")
            parseScope(this, lexemes, codeBlock, i+1, lexemes.size)
        else parseScope(this, lexemes, codeBlock, i, findExpressionEnd(i, lexemes, codeBlock))
    }
    val body = parseBlock()
    val elseBody: Scope? = if(lexemes[i].text == "else") {
        i++
        parseBlock()
    } else null

    return IfStatement(condition, body, elseBody, from, i - from)
}