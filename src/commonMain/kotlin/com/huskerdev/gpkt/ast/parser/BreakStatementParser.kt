package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope


fun parseBreakStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): BreakStatement {
    val i = from + 1

    val next = lexemes[i]
    if(next.text != ";")
        throw compilationError("Expected ;", next, codeBlock)
    if(!scope.isInIterableScope())
        throw compilationError("'break' can only be used inside iterator", next, codeBlock)

    return BreakStatement(scope, from, i - from)
}