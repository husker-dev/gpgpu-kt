package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.objects.*
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.types.Modifiers


fun parseFunctionStatement(
    scope: GPScope,
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
    val typeDeclaration = parseTypeDeclaration(i, lexemes, codeBlock)
    val type = typeDeclaration.first
    i += typeDeclaration.second

    val nameLexeme = lexemes[i]
    var function = GPFunction(scope, nameLexeme.text, mods, type)
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
        function.addArgument(fieldDeclaration.fields[0])

        i += fieldDeclaration.lexemeLength
        if(lexemes[i].text == ",")
            i++
    }
    if(lexemes[i+1].text != "{" && lexemes[i+1].text != ";")
        throw compilationError("Expected function block or ';'", lexemes[i+1], codeBlock)

    if(lexemes[i+1].text == ";")
        return FunctionDefinitionStatement(scope, function, from, i - from + 1)
    else {
        // Check if function was previously pre-defined
        // If yes -> get its object and continue
        scope.findDefinedFunction(function.name)?.let { def ->
            if(def.returnType != function.returnType)
                throw wrongFunctionDefinitionType(def, function.returnType, lexemes[from], codeBlock)
            if(!def.canAcceptArguments(function.argumentsTypes))
                throw wrongFunctionDefinitionParameters(def, function.argumentsTypes, lexemes[from], codeBlock)
            function = def
        }
    }

    val functionScope = parseScopeStatement(
        parentScope = scope,
        lexemes = lexemes,
        codeBlock = codeBlock,
        from = i+1,
        to = to,
        returnType = type,
        fields = linkedMapOf(*function.arguments.map { it.name to it }.toTypedArray())
    )
    i += functionScope.lexemeLength

    function.body = functionScope
    return FunctionStatement(scope, function, from, i - from + 1)
}