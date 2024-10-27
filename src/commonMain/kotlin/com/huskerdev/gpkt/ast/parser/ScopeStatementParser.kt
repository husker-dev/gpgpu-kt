package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.VOID


fun parseScopeStatement(
    parentScope: GPScope?,
    lexemes: List<Lexeme>,
    codeBlock: String,
    from: Int,
    to: Int,
    device: GPContext?          = parentScope?.context,
    returnType: PrimitiveType?  = null,
    iterable: Boolean           = false,
    modules: LinkedHashSet<ScopeStatement>       = linkedSetOf(),
    fields: LinkedHashMap<String, GPField>       = linkedMapOf(),
    functions: LinkedHashMap<String, GPFunction> = linkedMapOf()
): ScopeStatement {
    val scope = GPScope(
        context = device,
        parentScope = parentScope,
        returnType = returnType,
        iterable = iterable,
        modules = modules,
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

            if(text == "}"){
                if(returnType != null && returnType != VOID && !statements.any { it.returns })
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
                        scope.addField(field, lexeme, codeBlock)
                        if(parentScope == null && field.modifiers.isEmpty())
                            field.modifiers += Modifiers.THREADLOCAL
                    }
                }
                is FunctionStatement -> {
                    val function = statement.function
                    scope.addFunction(function, lexeme, codeBlock)
                }
                is ImportStatement -> {
                    val import = statement.import
                    import.paths.forEach { path ->
                        if(device == null || !device.modules.ast.containsKey(path))
                            throw compilationError("Module '${path}' not found", import.lexeme, codeBlock)

                        val module = device.modules.ast[path]!!
                        scope.modules += module.scope.modules   // Get dependent modules,
                        scope.modules += module                 // and itself
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