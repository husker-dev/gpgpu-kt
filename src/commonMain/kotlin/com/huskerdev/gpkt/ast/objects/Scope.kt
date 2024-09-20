package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Type

open class Scope(
    val parentScope: Scope?,
    open val returnType: Type? = null,
    val iterable: Boolean = false
) {
    val statements = mutableListOf<Statement>()

    val fields = mutableListOf<Field>()
    val functions = mutableListOf<Function>()

    var returns = false

    fun findDefinedField(name: String): Field? =
        fields.firstOrNull { it.name == name } ?: parentScope?.findDefinedField(name)

    fun findDefinedFunction(name: String, arguments: List<Type> = emptyList()): Function? =
        functions.firstOrNull {
            it.name == name && it.argumentsTypes == arguments
        } ?: parentScope?.findDefinedFunction(name, arguments)

    fun findReturnType(): Type =
        returnType ?: parentScope?.findReturnType() ?: Type.VOID

    fun isInIterableScope(): Boolean = if(iterable)
        true
    else parentScope?.isInIterableScope() ?: false

    private fun checkAvailableName(name: String, lexeme: Lexeme, codeBlock: String){
        if(findDefinedFunction(name) != null)
            throw functionAlreadyDefinedException(name, lexeme, codeBlock)
        if(findDefinedField(name) != null)
            throw variableAlreadyDefinedException(name, lexeme, codeBlock)
    }

    fun addField(field: Field, lexeme: Lexeme, codeBlock: String){
        checkAvailableName(field.name, lexeme, codeBlock)
        fields += field
    }

    fun addFunction(function: Function, lexeme: Lexeme, codeBlock: String){
        checkAvailableName(function.name, lexeme, codeBlock)
        functions += function
    }

    fun addStatement(statement: Statement, codeBlock: String){
        statements += statement
        if(statement is ReturnStatement)
            returns = true

        if(statement is FieldStatement){
            statement.fields.forEach { field ->
                addField(field, field.lexeme, codeBlock)
            }
        }
        if(statement is FunctionStatement){
            val function = statement.function
            addFunction(function, function.lexeme, codeBlock)
        }
    }
}