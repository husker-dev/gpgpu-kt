package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPScope


fun parseBreakStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
): BreakStatement {
    var i = from + 1

    if(i + 1 < to) {
        val next = lexemes[i]
        if (next.text != ";")
            throw compilationError("Expected ;", next, codeBlock)
        i++
    }
    if(!scope.isInIterableScope())
        throw compilationError("'break' can only be used inside iterator", lexemes[from], codeBlock)

    return BreakStatement(scope, from, i - from)
}