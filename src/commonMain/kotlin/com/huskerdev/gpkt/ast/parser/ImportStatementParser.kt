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

    val path = StringBuilder()
    while(lexemes[i].text != ";" && i < to){
        path.append(lexemes[i++].text)
    }
    if(path.endsWith('.'))
        throw compilationError("Module path cannot ends with '.'", lexemes[i], codeBlock)
    return ImportStatement(scope, Import(path.toString(), lexemes[from+1]), from, i - from + 1)
}