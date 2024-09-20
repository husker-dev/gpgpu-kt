package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Scope


fun parseContinueStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
): ContinueStatement {
    val i = from + 1

    val next = lexemes[i]
    if(next.text != ";")
        throw compilationError("Expected ;", next, codeBlock)
    if(!scope.isInIterableScope())
        throw compilationError("'continue' can only be used inside iterator", next, codeBlock)

    return ContinueStatement(from, i - from)
}