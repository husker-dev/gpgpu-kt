package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.processLexemes
import com.huskerdev.gpkt.ast.parser.parseScopeStatement
import com.huskerdev.gpkt.utils.Dictionary


class GPAst {
    companion object {
        fun parse(text: String, context: GPContext? = null, executableProgram: Boolean = true): ScopeStatement {
            val lexemes = processLexemes(text)

            val currentDictionary = context?.modules?.dictionary ?: Dictionary()
            val dictionary = if(executableProgram) currentDictionary.copy() else currentDictionary

            val scope = parseScopeStatement(
                parentScope = null,
                lexemes = lexemes,
                codeBlock = text,
                from = 0,
                to = lexemes.size,
                dictionary = dictionary,
                context = context,
            )
            if(executableProgram){

                // Expand modules
                val statements = arrayListOf<Statement>()
                scope.scope.modules.forEach { importScope ->
                    importScope.statements.forEach {
                        when(it){
                            is FieldStatement -> {
                                it.fields.forEach { field ->
                                    if(field.name !in scope.scope.fields)
                                        scope.scope.fields[field.name] = field
                                }
                                statements += it
                            }
                            is FunctionStatement -> {
                                if(it !is FunctionDefinitionStatement && it.function.name !in scope.scope.functions)
                                    scope.scope.functions[it.function.name] = it.function
                                statements += it
                            }
                            is ClassStatement -> {
                                if(it.classObj.name !in scope.scope.classes)
                                    scope.scope.classes[it.classObj.name] = it.classObj
                                statements += it
                            }
                        }
                    }
                }
                scope.statements.addAll(0, statements)

                // Check if all functions have body
                scope.scope.functions.forEach { entry ->
                    val func = entry.value
                    if(func.body == null)
                        throw functionNotImplemented(func, lexemes[0], text)
                }
            }
            return scope
        }
    }
}

