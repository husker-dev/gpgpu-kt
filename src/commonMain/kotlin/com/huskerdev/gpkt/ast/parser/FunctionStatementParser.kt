package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.*
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.utils.Dictionary

@Suppress("unused")
fun parseFunctionStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): FunctionStatement {
    // Modifiers
    val (mods, modsEnd) = parseModifiers(from, to, lexemes)
    var r = modsEnd

    // Type
    val (type, typeEnd) = parseTypeDeclaration(scope, r, lexemes, codeBlock)
    r = typeEnd

    return parseFunctionStatement(scope, mods, type, r, lexemes, codeBlock, from, to, dictionary)
}

fun parseFunctionStatement(
    scope: GPScope,
    mods: List<Modifiers>,
    type: PrimitiveType?,
    nameIndex: Int,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): FunctionStatement {
    var i = nameIndex

    if(type == null)
        throw compilationError("Cannot use var with functions", lexemes[nameIndex-1], codeBlock)

    val name = lexemes[i++].text
    var function = GPFunction(scope, name, dictionary.nextWord(name), mods, type)

    if(lexemes[i].text != "(")
        throw expectedException("arguments block", lexemes[i], codeBlock)
    i++

    // Getting parameters
    while(lexemes[i].text != ")"){
        if(i >= to)
            throw compilationError("Expected ')'", lexemes.last(), codeBlock)

        val fieldDeclaration = parseFieldDeclaration(
            scope, lexemes, codeBlock, i, to,
            dictionary,
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
        // If function was previously pre-defined
        scope.findFunction(function.name)?.let { def ->
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
        dictionary,
        returnType = type,
        fields = linkedMapOf(*function.arguments.map { it.name to it }.toTypedArray())
    )
    i += functionScope.lexemeLength

    function.body = functionScope
    return FunctionStatement(scope, function, from, i - from + 1)
}