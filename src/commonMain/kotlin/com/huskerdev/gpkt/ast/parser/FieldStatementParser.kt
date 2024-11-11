package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.*
import com.huskerdev.gpkt.utils.Dictionary


fun parseFieldStatement(
    scope: GPScope,
    mods: MutableList<Modifiers>,
    type: PrimitiveType?,
    nameIndex: Int,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
) = parseFieldDeclaration(
    scope,
    mods, type, nameIndex,
    lexemes, codeBlock,
    from, to, dictionary,
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
    dictionary: Dictionary,
    allowMultipleDeclaration: Boolean,
    allowDefaultValue: Boolean,
    endsWithSemicolon: Boolean
): FieldStatement{
    // Modifiers
    val (mods, modsEnd) = parseModifiers(from, to, lexemes)
    var r = modsEnd

    // Type
    val (type, typeEnd) = parseTypeDeclaration(scope, r, lexemes, codeBlock)
    r = typeEnd

    return parseFieldDeclaration(
        scope,
        mods, type, r,
        lexemes, codeBlock,
        from, to, dictionary,
        allowMultipleDeclaration, allowDefaultValue, endsWithSemicolon
    )
}

fun parseFieldDeclaration(
    scope: GPScope,
    mods: MutableList<Modifiers>,
    initialType: PrimitiveType?,
    nameIndex: Int,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary,
    allowMultipleDeclaration: Boolean,
    allowDefaultValue: Boolean,
    endsWithSemicolon: Boolean
): FieldStatement{
    var type = initialType
    val fields = arrayListOf<GPField>()
    var i = nameIndex

    // Iterate over field declarations
    while(i < to){
        if(lexemes[i].type != Lexeme.Type.NAME)
            throw expectedException("variable name", lexemes[i], codeBlock)
        val nameLexeme = lexemes[i]
        val name = nameLexeme.text

        var initialExpression: Expression? = null
        if(lexemes[i+1].text == "="){
            if(!allowDefaultValue)
                throw compilationError("Default initialization is not allowed here", lexemes[i+1], codeBlock)

            initialExpression = parseExpression(scope, lexemes, codeBlock, i+2) ?:
                throw compilationError("expected initial value", lexemes[i+2], codeBlock)

            // If var
            if(type == null)
                type = initialExpression.type

            if(type != initialExpression.type && !PrimitiveType.canAssignNumbers(type, initialExpression.type))
                throw expectedTypeException(type, initialExpression.type, lexemes[i+2], codeBlock)

            // If class, then unpack
            if(type != initialExpression.type && initialExpression.type is ClassType){
                val clazz = scope.findClass((initialExpression.type as ClassType).className)!!
                initialExpression = FunctionCallExpression(initialExpression, getPrimitiveClassGetter(clazz), emptyList(),
                    initialExpression.lexemeIndex, initialExpression.lexemeLength)
            }

            i += initialExpression.lexemeLength + 1
        }
        if(type == null)
            throw compilationError("Unable to determine type of '$name'", lexemes[nameIndex-1], codeBlock)

        fields += GPField(name, dictionary.nextWord(name), mods, type, initialExpression)

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

fun parseModifiers(from: Int, to: Int, lexemes: List<Lexeme>): Pair<MutableList<Modifiers>, Int>{
    var r = from
    val mods = mutableListOf<Modifiers>()
    while(r < to && lexemes[r].text in modifiers){
        mods += Modifiers.map[lexemes[r].text]!!
        r++
    }
    return mods to r
}

fun parseTypeDeclaration(scope: GPScope, i: Int, lexemes: List<Lexeme>, codeBlock: String): Pair<PrimitiveType?, Int>{
    val lexeme = lexemes[i]
    if(lexeme.text == "var")
        return null to i+1

    val type: SinglePrimitiveType<*> =
        primitivesMap[lexeme.text]
        ?: scope.findClass(lexeme.text)?.type
        ?: throw compilationError("Type '${lexeme.text}' is not defined", lexeme, codeBlock)

    if(lexemes[i+1].text == "["){
        if(lexemes[i+2].text == "]"){
            return type.toDynamicArray() to i+3

        }else if(lexemes[i+3].text == "]"){
            if(lexemes[i+2].type != Lexeme.Type.INT &&
                lexemes[i+2].type != Lexeme.Type.BYTE
            ) throw compilationError("Array size should be constant int", lexeme, codeBlock)
            return type.toArray(lexemes[i+2].text.toInt()) to i+4

        }else throw compilationError("Failed to get array size", lexeme, codeBlock)
    }
    return type to i+1
}