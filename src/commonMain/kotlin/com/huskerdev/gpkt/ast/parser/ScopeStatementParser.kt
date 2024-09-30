package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type


fun parseScopeStatement(
    parentScope: Scope?,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    device: GPDevice?   = parentScope?.device,
    returnType: Type?   = null,
    iterable: Boolean   = false,
    fields: MutableList<Field> = mutableListOf(),
    functions: MutableList<Function> = mutableListOf()
): ScopeStatement {
    //println("==== SCOPE: ${from}-${to} ====")
    val scope = Scope(
        device = device,
        parentScope = parentScope,
        returnType = returnType,
        iterable = iterable,
        fields = fields,
        functions = functions
    )
    val statements = mutableListOf<Statement>()
    var returns = false

    var i = from
    if(lexemes[i].text == "{")
        i++

    try {
        while (i < to) {
            val lexeme = lexemes[i]
            val text = lexeme.text
            //println("current: ${lexemes.subList(i, kotlin.math.min(to, i+3)).joinToString(" ") { it.text }}")

            if(text == "}"){
                if(returnType != null && returnType != Type.VOID && !statements.any { it.returns })
                    throw compilationError("Expected return statement", lexeme, codeBlock)
                return ScopeStatement(scope, statements, returns, from, i - from + 1)
            }
            val statement = parseStatement(scope, lexemes, codeBlock, i, to)
            statements += statement

            when (statement) {
                is ReturnStatement -> {
                    returns = true
                }
                is FieldStatement -> {
                    statement.fields.forEach { field ->
                        scope.addField(field, field.lexeme, codeBlock)
                        if(parentScope == null && field.modifiers.isEmpty())
                            field.modifiers += Modifiers.CONST
                    }
                }
                is FunctionStatement -> {
                    val function = statement.function
                    scope.addFunction(function, function.lexeme, codeBlock)
                }
                is ImportStatement -> {
                    val import = statement.import
                    if(device == null || !device.modules.ast.containsKey(import.path))
                        throw compilationError("Module '${import.path}' not found", import.lexeme, codeBlock)

                    val module = device.modules.ast[import.path]!!
                    module.statements.forEach {
                        when(it){
                            is FieldStatement -> {
                                statements += it
                                scope.fields += it.fields
                            }
                            is FunctionStatement -> {
                                statements += it
                                scope.functions += it.function
                            }
                        }
                    }
                }
            }

            i += statement.lexemeLength
        }
    }catch (e: IndexOutOfBoundsException){
        e.printStackTrace()
        throw unexpectedEofException(lexemes.last(), codeBlock)
    }
    return ScopeStatement(scope, statements, returns, from, to - from)
}