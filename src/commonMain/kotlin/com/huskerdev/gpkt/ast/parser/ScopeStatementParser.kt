package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.VOID
import com.huskerdev.gpkt.utils.Dictionary


fun parseScopeStatement(
    parentScope: GPScope?,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary,
    context: GPContext?         = parentScope?.context,
    returnType: PrimitiveType?  = null,
    iterable: Boolean           = false,
    fields: LinkedHashMap<String, GPField>       = linkedMapOf(),
    functions: LinkedHashMap<String, GPFunction> = linkedMapOf()
): ScopeStatement {
    val scope = GPScope(
        context = context,
        parentScope = parentScope,
        dictionary = dictionary,
        returnType = returnType,
        iterable = iterable,
        fields = fields,
        functions = functions
    )

    var i = from
    if(lexemes[i].text == "{")
        i++

    try {
        while (i < to) {
            val lexeme = lexemes[i]
            val text = lexeme.text

            if(text == "}"){
                if(returnType != null && returnType != VOID && !scope.returns)
                    throw compilationError("Expected return statement", lexeme, codeBlock)
                return ScopeStatement(parentScope, scope, from, i - from + 1)
            }
            val statement = parseStatement(scope, lexemes, codeBlock, i, to, dictionary)
            scope.addStatement(statement, lexeme, codeBlock)

            i += statement.lexemeLength
        }
    }catch (e: IndexOutOfBoundsException){
        e.printStackTrace()
        throw unexpectedEofException(lexemes.last(), codeBlock)
    }
    return ScopeStatement(parentScope, scope, from, to - from)
}