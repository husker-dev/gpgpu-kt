package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.FieldStatement
import com.huskerdev.gpkt.ast.FunctionStatement
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.Statement
import com.huskerdev.gpkt.ast.lexer.processLexemes
import com.huskerdev.gpkt.ast.parser.parseScopeStatement


class GPAst {
    companion object {
        fun parse(text: String, device: GPContext? = null, expandImports: Boolean = true): ScopeStatement {
            val lexemes = processLexemes(text)

            val scope =  parseScopeStatement(
                parentScope = null,
                lexemes = lexemes,
                codeBlock = text,
                from = 0,
                to = lexemes.size,
                device = device,
            )
            if(expandImports){
                val statements = arrayListOf<Statement>()
                scope.scope.modules.forEach { importScope ->
                    importScope.statements.forEach {
                        when(it){
                            is FieldStatement -> {
                                scope.scope.fields += it.fields
                                statements += it
                            }
                            is FunctionStatement -> {
                                scope.scope.functions += it.function
                                statements += it
                            }
                        }
                    }
                }
                scope.statements.addAll(0, statements)
            }
            return scope
        }
    }
}

