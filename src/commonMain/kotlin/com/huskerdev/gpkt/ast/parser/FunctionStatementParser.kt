package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.objects.*
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type


fun parseFunctionStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
): FunctionStatement {
    var i = from

    // Getting modifiers
    val mods = mutableListOf<Modifiers>()
    while(i < lexemes.size && lexemes[i].text in modifiers){
        mods += Modifiers.map[lexemes[i].text]!!
        i++
    }

    // Getting type
    var type = Type.map[lexemes[i].text]!!
    if(lexemes[i+1].text == "[" && lexemes[i+2].text == "]"){
        type = Type.toArrayType(type)
        i += 2
    }
    i++

    val nameLexeme = lexemes[i]
    val function = Function(nameLexeme, scope, nameLexeme.text, mods, type)
    i += 2

    // Getting parameters
    while(lexemes[i].text != ")"){
        if(i >= to)
            throw compilationError("Expected ')'", lexemes.last(), codeBlock)

        val fieldDeclaration = parseFieldDeclaration(
            scope,
            lexemes,
            codeBlock,
            i,
            to,
            allowMultipleDeclaration = false,
            allowDefaultValue = false,
            endsWithSemicolon = false
        )
        function.addArgument(fieldDeclaration.fields[0], lexemes[fieldDeclaration.lexemeIndex], codeBlock)

        i += fieldDeclaration.lexemeLength
        if(lexemes[i].text == ",")
            i++
    }
    if(lexemes[i+1].text != "{")
        throw compilationError("Expected function block", lexemes[i+1], codeBlock)

    i = parseScope(function, lexemes, codeBlock, i+2, lexemes.size)

    return FunctionStatement(scope, function, from, i - from)
}