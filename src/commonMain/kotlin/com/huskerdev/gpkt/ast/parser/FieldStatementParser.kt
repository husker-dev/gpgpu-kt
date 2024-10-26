package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.SinglePrimitiveType
import com.huskerdev.gpkt.ast.types.primitivesMap


fun parseFieldStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
) = parseFieldDeclaration(
    scope,
    lexemes,
    codeBlock,
    from,
    to,
    allowMultipleDeclaration = true,
    allowDefaultValue = true,
    endsWithSemicolon = true
)


fun parseFieldDeclaration(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    allowMultipleDeclaration: Boolean,
    allowDefaultValue: Boolean,
    endsWithSemicolon: Boolean
): FieldStatement{
    val fields = arrayListOf<Field>()
    var i = from

    // Getting modifiers
    val mods = mutableListOf<Modifiers>()
    while(i < to && lexemes[i].text in modifiers){
        mods += Modifiers.map[lexemes[i].text]!!
        i++
    }

    // Getting type
    val typeDeclaration = parseTypeDeclaration(i, lexemes, codeBlock)
    val type = typeDeclaration.first
    i += typeDeclaration.second

    // Iterate over field declarations
    while(i < to){
        if(lexemes[i].type != Lexeme.Type.NAME)
            throw expectedException("variable name", lexemes[i], codeBlock)
        val nameLexeme = lexemes[i]

        var initialExpression: Expression? = null
        if(lexemes[i+1].text == "="){
            if(!allowDefaultValue)
                throw compilationError("Default initialization is not allowed here", lexemes[i+1], codeBlock)

            initialExpression = parseExpression(scope, lexemes, codeBlock, i+2) ?:
                throw compilationError("expected initial value", lexemes[i+2], codeBlock)

            if(type != initialExpression.type && !PrimitiveType.canAssignNumbers(type, initialExpression.type))
                throw expectedTypeException(type, initialExpression.type, lexemes[i+2], codeBlock)

            i += initialExpression.lexemeLength + 1
        }
        fields += Field(nameLexeme.text, mods, type, initialExpression)

        if(i >= to)
            return FieldStatement(scope, fields, from, i - from)

        if(endsWithSemicolon && lexemes[i+1].text == ";")
            return FieldStatement(scope, fields, from, i - from + 2)
        else if(lexemes[i+1].text == "," && lexemes[i+2].type == Lexeme.Type.NAME) {
            if(!allowMultipleDeclaration)
                throw compilationError("Multiple field declaration is not allowed here", lexemes[i + 1], codeBlock)
            i += 2
        }else
            return FieldStatement(scope, fields, from, i - from + 1)

    }
    throw compilationError("Can not read field declaration", lexemes[from], codeBlock)
}

fun parseTypeDeclaration(i: Int, lexemes: List<Lexeme>, codeBlock: String): Pair<PrimitiveType, Int>{
    val type: PrimitiveType = primitivesMap[lexemes[i].text] ?:
        throw expectedException("type", lexemes[i].text, lexemes[i], codeBlock)

    if(lexemes[i+1].text == "["){
        if(lexemes[i+2].text == "]"){
            return (type as SinglePrimitiveType<*>).toDynamicArray() to 3

        }else if(lexemes[i+3].text == "]"){
            if(lexemes[i+2].type != Lexeme.Type.INT)
                throw compilationError("Array size should be constant int", lexemes[i], codeBlock)
            return (type as SinglePrimitiveType<*>).toArray(lexemes[i+2].text.toInt()) to 4

        }else throw compilationError("Failed to get array size", lexemes[i], codeBlock)
    }
    return type to 1
}