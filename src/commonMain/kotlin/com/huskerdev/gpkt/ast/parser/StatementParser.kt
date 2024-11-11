package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.lexer.primitives
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.Dictionary


fun parseStatement(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): Statement{
    val lexeme = lexemes[from]
    val text = lexeme.text

    return if(lexeme.type == Lexeme.Type.SPECIAL){
        when {
            text == "{" -> parseScopeStatement(scope, lexemes, codeBlock, from, to, dictionary)
            text == ";" -> EmptyStatement(scope, from, 1)
            text == "return" -> parseReturnStatement(scope, lexemes, codeBlock, from)
            text == "if" -> parseIfStatement(scope, lexemes, codeBlock, from, to, dictionary)
            text == "while" -> parseWhileStatement(scope, lexemes, codeBlock, from, to, dictionary)
            text == "for" -> parseForStatement(scope, lexemes, codeBlock, from, to, dictionary)
            text == "break" -> parseBreakStatement(scope, lexemes, codeBlock, from, to)
            text == "continue" -> parseContinueStatement(scope, lexemes, codeBlock, from, to)
            text == "import" -> parseImportStatement(scope, lexemes, codeBlock, from, to)
            text == "class" -> parseClassStatement(scope, lexemes, codeBlock, from, to, dictionary)
            (text == "var" || text in primitives || text in modifiers) ->
                parseFunctionOrField(scope, lexemes, codeBlock, from, to, dictionary)
            else -> throw compilationError("Unexpected symbol: '${text}'", lexeme, codeBlock)
        }
    } else if(scope.findClass(text) != null || (lexeme.type == Lexeme.Type.NAME && lexemes[from+1].type == Lexeme.Type.NAME))
        parseFunctionOrField(scope, lexemes, codeBlock, from, to, dictionary)
    else
        ExpressionStatement(scope, parseExpression(scope, lexemes, codeBlock, from)!!)
}

private fun parseFunctionOrField(
    scope: GPScope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    dictionary: Dictionary
): Statement {
    // Modifiers
    val (mods, modsEnd) = parseModifiers(from, to, lexemes)
    var r = modsEnd

    // Type
    val (type, typeEnd) = parseTypeDeclaration(scope, r, lexemes, codeBlock)
    r = typeEnd

    return if(lexemes[r+1].text == "(")
        parseFunctionStatement(scope, mods, type, r, lexemes, codeBlock, from, to, dictionary)
    else parseFieldStatement(scope, mods, type, r, lexemes, codeBlock, from, to, dictionary)
}