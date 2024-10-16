package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Type

open class Scope(
    val context: GPContext?,
    val parentScope: Scope?,
    val returnType: Type? = null,
    val iterable: Boolean = false,
    val modules: LinkedHashSet<ScopeStatement> = linkedSetOf(),
    val fields: MutableList<Field> = mutableListOf(),
    val functions: MutableList<Function> = mutableListOf(),
) {

    fun findDefinedField(name: String): Field? =
        fields.firstOrNull {
            it.name == name
        } ?: modules.firstNotNullOfOrNull {
            it.scope.findDefinedField(name)
        } ?: if(parentScope == null)
            allPredefinedFields[name]
        else parentScope.findDefinedField(name)

    fun findDefinedFunction(name: String, arguments: List<Type> = emptyList()): Function? =
        functions.firstOrNull {
            it.name == name && it.canAcceptArguments(arguments)
        } ?: modules.firstNotNullOfOrNull {
            it.scope.findDefinedFunction(name, arguments)
        } ?: if(parentScope == null){
            val func = allPredefinedFunctions[name]
            if(func != null && func.canAcceptArguments(arguments))
                func
            else null
        }else parentScope.findDefinedFunction(name, arguments)

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
}