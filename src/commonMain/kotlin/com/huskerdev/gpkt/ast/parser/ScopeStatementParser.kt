package com.huskerdev.gpkt.ast.parser

import com.huskerdev.gpkt.GPContext
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
    device: GPContext?      = parentScope?.context,
    returnType: Type?       = null,
    iterable: Boolean       = false,
    modules: HashSet<ScopeStatement> = hashSetOf(),
    fields: MutableList<Field> = mutableListOf(),
    functions: MutableList<Function> = mutableListOf()
): ScopeStatement {
    val scope = Scope(
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
                        scope.addField(field, lexeme, codeBlock)
                        if(parentScope == null && field.modifiers.isEmpty())
                            field.modifiers += Modifiers.CONST
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
                        scope.modules += module                 // Get module itself
                        scope.modules += module.scope.modules   // and its dependencies
                    }
                }
            }

            i += statement.lexemeLength
        }
    }catch (e: IndexOutOfBoundsException){
        e.printStackTrace()
        throw unexpectedEofException(lexemes.last(), codeBlock)
    }

    if(parentScope == null){
        // If this scope is main, then add module statements
        statements.addAll(0, scope.modules.flatMap { module ->
            module.statements.filter { it is FieldStatement || it is FunctionStatement }
        })
    }
    return ScopeStatement(scope, statements, returns, from, to - from)
}