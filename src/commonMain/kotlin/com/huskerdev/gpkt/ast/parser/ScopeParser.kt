package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.lexer.modifiers
import com.huskerdev.gpkt.ast.lexer.primitives
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type


fun parseScope(scope: Scope, lexemes: List<Lexeme>, codeBlock: String, from: Int, to: Int): Int{
    //println("==== SCOPE: ${from}-${to} ====")

    var i = from
    try {
        while (i < to) {
            val lexeme = lexemes[i]
            //println("current: ${lexemes.subList(i, min(to, i+3)).joinToString(" ") { it.text }}")

            if(lexeme.text == "}"){
                if(scope.returnType != null && scope.returnType != Type.VOID && scope.returnStatement == null)
                    throw compilationError("Expected return statement", lexeme, codeBlock)
                return i + 1
            }

            val statement: Statement = if(lexeme.type == Lexeme.Type.SPECIAL){
                when {
                    lexeme.text == ";" -> EmptyStatement(i, 1)
                    lexeme.text == "return" -> parseReturnStatement(scope, lexemes, codeBlock, i)
                    lexeme.text == "if" -> parseIfStatement(scope, lexemes, codeBlock, i)
                    lexeme.text == "while" -> parseWhileStatement(scope, lexemes, codeBlock, i)
                    lexeme.text == "for" -> parseForStatement(scope, lexemes, codeBlock, i)
                    (lexeme.text in primitives || lexeme.text in modifiers) -> {
                        var r = i
                        while(lexemes[r].type != Lexeme.Type.NAME)
                            r++

                        if(lexemes[r+1].text == "(") parseFunctionStatement(scope, lexemes, codeBlock, i)
                        else parseFieldStatement(scope, lexemes, codeBlock, i, to)
                    }
                    else -> throw compilationError("Not implemented", lexeme, codeBlock)
                }
            } else
                ExpressionStatement(parseExpression(scope, lexemes, codeBlock, i)!!)

            scope.addStatement(statement, codeBlock)
            i += statement.lexemeLength
        }

    }catch (e: IndexOutOfBoundsException){
        e.printStackTrace()
        throw unexpectedEofException(lexemes.last(), codeBlock)
    }
    return i
}