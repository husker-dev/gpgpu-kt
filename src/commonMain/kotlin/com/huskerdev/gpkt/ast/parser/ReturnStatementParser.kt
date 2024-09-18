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
    if(next.text == ";" && scope.returnType != Type.VOID)
        throw compilationError("Expected return value", next, codeBlock)

    val expression = if(scope.returnType != Type.VOID) {
        val expression = if(
            next.type == Lexeme.Type.NUMBER ||
            next.type == Lexeme.Type.NUMBER_FLOATING_POINT ||
            next.type == Lexeme.Type.LOGICAL
        ) createConstExpression(i, next, codeBlock)
        else parseExpression(scope, lexemes, codeBlock, i)!!

        if (expression.type != scope.returnType)
            throw expectedTypeException(scope.returnType, expression.type, next, codeBlock)
        expression
    }else null

    i += (scope.returnStatement?.expression?.lexemeLength ?: 0)

    return ReturnStatement(expression, from, i - from)
}