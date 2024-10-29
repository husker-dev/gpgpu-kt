package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.BOOLEAN
import com.huskerdev.gpkt.ast.types.SinglePrimitiveType
import com.huskerdev.gpkt.utils.Dictionary


fun parseIfStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): IfStatement {
    var i = from + 1
    if(lexemes[i].text != "(")
        throw compilationError("Expected condition", lexemes[i], codeBlock)

    val condition = parseExpression(scope, lexemes, codeBlock, i+1)!!
    val type = condition.type
    if(type !is SinglePrimitiveType<*> || !type.isLogical)
        throw expectedTypeException(BOOLEAN, condition.type, lexemes[i+1], codeBlock)
    i += condition.lexemeLength + 2

    val body = parseStatement(scope, lexemes, codeBlock, i, to, dictionary)
    i += body.lexemeLength

    var elseBody: Statement? = null
    if(lexemes[i].text == "else") {
        i++
        elseBody = parseStatement(scope, lexemes, codeBlock, i, to, dictionary)
        i += elseBody.lexemeLength
    }

    return IfStatement(scope, condition, body, elseBody, from, i - from)
}