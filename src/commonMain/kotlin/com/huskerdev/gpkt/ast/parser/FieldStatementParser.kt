package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseFieldStatement(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
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
    var type = Type.map[lexemes[i].text]!!
    if(lexemes[i+1].text == "[" && lexemes[i+2].text == "]"){
        type = Type.toArrayType(type)
        i += 2
    }
    i++

    // Iterate over field declarations
    var field: Field? = null
    while(i < to){
        if(field != null){
            val separator = lexemes[i].text
            if(separator == "," || separator == ";") {
                fields += field
                field = null
            } else throw compilationError("Expected ';' or ','", lexemes[i], codeBlock)
            i++
            if(separator == ";")
                break
        }

        if(lexemes[i].type != Lexeme.Type.NAME)
            throw compilationError("Expected variable name", lexemes[i], codeBlock)
        val nameLexeme = lexemes[i]

        var initialExpression: Expression? = null
        if(lexemes[i+1].text == "="){
            initialExpression = parseExpression(scope, lexemes, codeBlock, i+2) ?:
                    throw compilationError("expected initial value", lexemes[i+2], codeBlock)
            if(type != initialExpression.type)
                throw expectedTypeException(type, initialExpression.type, lexemes[i+2], codeBlock)

            i += initialExpression.lexemeLength + 1
        }

        field = Field(nameLexeme, nameLexeme.text, mods, type, initialExpression)
        i++
    }
    // If declaration wasn't ended with ';', then flush
    if(field != null)
        fields += field
    return FieldStatement(scope, fields, from, i - from)
}