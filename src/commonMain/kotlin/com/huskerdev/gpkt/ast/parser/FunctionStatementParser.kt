package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.lexer.primitives
import com.huskerdev.gpkt.ast.objects.*
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type


fun parseFunctionStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int
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
        if(lexemes[i].text !in primitives)
            throw compilationError("Expected argument type", lexemes[i], codeBlock)
        var argType = Type.map[lexemes[i].text]!!

        if(lexemes[i+1].text == "[" && lexemes[i+2].text == "]"){
            argType = Type.toArrayType(argType)
            i += 2
        }

        if(lexemes[i+1].type != Lexeme.Type.NAME)
            throw compilationError("Expected argument name", lexemes[i+1], codeBlock)
        function.addArgument(Field(lexemes[i+1], lexemes[i+1].text, emptyList(), argType), lexemes[i], codeBlock)
        i += if(lexemes[i+2].text == ",") 3 else 2
    }
    if(lexemes[i+1].text != "{")
        throw compilationError("Expected function block", lexemes[i+1], codeBlock)

    scope.addFunction(function, nameLexeme, codeBlock)
    i = parseScope(function, lexemes, codeBlock, i+2, lexemes.size)

    return FunctionStatement(function, from, i)
}