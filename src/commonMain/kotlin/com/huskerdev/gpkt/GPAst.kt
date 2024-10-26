package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.processLexemes
import com.huskerdev.gpkt.ast.parser.parseScopeStatement


class GPAst {
    companion object {
        fun parse(text: String, device: GPContext? = null, executableProgram: Boolean = true): ScopeStatement {
            val lexemes = processLexemes(text)

            val scope =  parseScopeStatement(
                parentScope = null,
                lexemes = lexemes,
                codeBlock = text,
                from = 0,
                to = lexemes.size,
                device = device,
            )
            if(executableProgram){

                // Expand modules
                val statements = arrayListOf<Statement>()
                scope.scope.modules.forEach { importScope ->
                    importScope.statements.forEach {
                        when(it){
                            is FieldStatement -> {
                                scope.scope.fields += it.fields
                                statements += it
                            }
                            is FunctionStatement -> {
                                if(!scope.scope.functions.any { f -> f.name == it.function.name })
                                    scope.scope.functions += it.function
                                statements += it
                            }
                        }
                    }
                }
                scope.statements.addAll(0, statements)

                // Check if all functions have body
                scope.scope.functions.forEach { func ->
                    if(func.body == null) {
                        // Getting statement
                        val statement = scope.statements.find { it is FunctionStatement && it.function == func }!!
                        throw compilationError("Function doesn't have implementation", lexemes[statement.lexemeIndex], text)
                    }
                }
            }
            return scope
        }
    }
}

