package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseReturnStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): ReturnStatement {
    var i = from + 1

    val next = lexemes[i]
    val returnType = scope.findReturnType()
    if(next.text == ";" && returnType != Type.VOID)
        throw compilationError("Expected return value", next, codeBlock)

    val expression = if(returnType != Type.VOID) {
        val expression = if(
            next.type == Lexeme.Type.INT ||
            next.type == Lexeme.Type.FLOAT ||
            next.type == Lexeme.Type.LOGICAL
        ) createConstExpression(i, next, codeBlock)
        else parseExpression(scope, lexemes, codeBlock, i)!!

        if (expression.type != returnType && !Type.canAssignNumbers(expression.type, returnType))
            throw expectedTypeException(returnType, expression.type, next, codeBlock)
        expression
    }else null

    i += (expression?.lexemeLength ?: 0) + 1

    return ReturnStatement(scope, expression, from, i - from)
}