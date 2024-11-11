package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.processLexemes
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.parser.parseScopeStatement
import com.huskerdev.gpkt.ast.preproc.processPreprocessor
import com.huskerdev.gpkt.utils.Dictionary


class GPAst {
    companion object {
        fun parse(text: String, context: GPContext? = null): GPScope {
            val preprocessed = processPreprocessor(text, context)
            val lexemes = processLexemes(preprocessed)

            val scope = parseScopeStatement(
                parentScope = null,
                lexemes = lexemes,
                codeBlock = preprocessed,
                from = 0,
                to = lexemes.size,
                dictionary = Dictionary(),
                context = context,
            )

            scope.scopeObj.statements.forEach {
                if(it is FunctionStatement && it.function.body == null)
                    throw functionNotImplemented(it.function, lexemes[it.lexemeIndex], preprocessed)
            }
            return scope.scopeObj
        }
    }
}

