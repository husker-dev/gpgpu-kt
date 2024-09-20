package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.lexer.primitives
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseScope(
    scope: Scope,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int
): Int {
    //println("==== SCOPE: ${from}-${to} ====")

    var i = from
    try {
        while (i < to) {
            val lexeme = lexemes[i]
            val text = lexeme.text
            //println("current: ${lexemes.subList(i, kotlin.math.min(to, i+3)).joinToString(" ") { it.text }}")

            if(text == "}"){
                if(scope.returnType != null && scope.returnType != Type.VOID && !scope.statements.any { it.returns })
                    throw compilationError("Expected return statement", lexeme, codeBlock)
                return i + 1
            }

            val statement: Statement = if(lexeme.type == Lexeme.Type.SPECIAL){
                when {
                    text == ";" -> EmptyStatement(scope, i, 1)
                    text == "return" -> parseReturnStatement(scope, lexemes, codeBlock, i)
                    text == "if" -> parseIfStatement(scope, lexemes, codeBlock, i)
                    text == "while" -> parseWhileStatement(scope, lexemes, codeBlock, i)
                    text == "for" -> parseForStatement(scope, lexemes, codeBlock, i, to)
                    text == "break" -> parseBreakStatement(scope, lexemes, codeBlock, i)
                    text == "continue" -> parseContinueStatement(scope, lexemes, codeBlock, i)
                    (text in primitives || text in modifiers) -> {
                        var r = i
                        while(lexemes[r].type != Lexeme.Type.NAME && r < to)
                            r++

                        if(lexemes[r+1].text == "(") parseFunctionStatement(scope, lexemes, codeBlock, i, to)
                        else parseFieldStatement(scope, lexemes, codeBlock, i, to)
                    }
                    else -> throw compilationError("Unexpected symbol: '${text}'", lexeme, codeBlock)
                }
            } else
                ExpressionStatement(scope, parseExpression(scope, lexemes, codeBlock, i)!!)

            scope.addStatement(statement, codeBlock)
            i += statement.lexemeLength
        }

    }catch (e: IndexOutOfBoundsException){
        e.printStackTrace()
        throw unexpectedEofException(lexemes.last(), codeBlock)
    }
    return to
}