package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.VOID


fun parseReturnStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): ReturnStatement {
    var i = from + 1

    val next = lexemes[i]
    val returnType = scope.findReturnType()
    if(next.text == ";" && returnType != VOID)
        throw compilationError("Expected return value", next, codeBlock)

    val expression = if(returnType != VOID) {
        val expression = if(
            next.type == Lexeme.Type.INT ||
            next.type == Lexeme.Type.FLOAT ||
            next.type == Lexeme.Type.LOGICAL
        ) createConstExpression(i, next, codeBlock)
        else parseExpression(scope, lexemes, codeBlock, i)!!

        if (expression.type != returnType && !PrimitiveType.canAssignNumbers(returnType, expression.type))
            throw expectedTypeException(returnType, expression.type, next, codeBlock)
        expression
    }else null

    i += (expression?.lexemeLength ?: 0) + 1

    return ReturnStatement(scope, expression, from, i - from)
}