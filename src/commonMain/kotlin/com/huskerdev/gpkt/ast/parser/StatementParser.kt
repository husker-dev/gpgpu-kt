package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.EmptyStatement
import com.huskerdev.gpkt.ast.ExpressionStatement
import com.huskerdev.gpkt.ast.Statement
import com.huskerdev.gpkt.ast.compilationError
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
            (text in primitives || text in modifiers) -> {
                var r = from
                while(lexemes[r].type != Lexeme.Type.NAME && r < to)
                    r++

                if(lexemes[r+1].text == "(") parseFunctionStatement(scope, lexemes, codeBlock, from, to, dictionary)
                else parseFieldStatement(scope, lexemes, codeBlock, from, to, dictionary)
            }
            else -> throw compilationError("Unexpected symbol: '${text}'", lexeme, codeBlock)
        }
    } else
        ExpressionStatement(scope, parseExpression(scope, lexemes, codeBlock, from)!!)
}