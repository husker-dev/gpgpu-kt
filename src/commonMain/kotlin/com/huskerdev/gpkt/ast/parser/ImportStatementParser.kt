package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Import
import com.huskerdev.gpkt.ast.objects.Scope


fun parseImportStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
): ImportStatement {
    var i = from + 1
    if(lexemes[i].type != Lexeme.Type.NAME)
        throw compilationError("Expected module name", lexemes[i], codeBlock)

    val names = arrayListOf<String>()
    while(i < to){
        names += lexemes[i].text

        i++
        val next = lexemes[i]
        if(next.text == ";")
            break
        else if(next.text == ",")
            i++
        else
            throw compilationError("Unrecognized symbol '${next.text}' in import declaration", lexemes[i], codeBlock)
    }

    return ImportStatement(scope, Import(names, lexemes[from+1]), from, i - from + 1)
}