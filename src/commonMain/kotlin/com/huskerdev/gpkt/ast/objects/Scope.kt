package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Type

open class Scope(
    val parentScope: Scope?,
    val returnType: Type
) {
    val statements = mutableListOf<Statement>()

    val fields = mutableListOf<Field>()
    val functions = mutableListOf<Function>()

    var returnStatement: ReturnStatement? = null

    fun findDefinedField(name: String): Field? =
        fields.firstOrNull { it.name == name } ?: parentScope?.findDefinedField(name)

    fun findDefinedFunction(name: String, arguments: List<Type> = emptyList()): Function? =
        functions.firstOrNull {
            it.name == name && it.argumentsTypes == arguments
        } ?: parentScope?.findDefinedFunction(name, arguments)

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