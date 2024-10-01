package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Type

open class Scope(
    val device: GPDevice?,
    private val parentScope: Scope?,
    val returnType: Type? = null,
    val iterable: Boolean = false,
    val fields: MutableList<Field> = mutableListOf(),
    val functions: MutableList<Function> = mutableListOf(),
) {

    fun findDefinedField(name: String): Field? =
        fields.firstOrNull {
            it.name == name
        } ?: if(parentScope == null)
            predefinedFields[name]
        else parentScope.findDefinedField(name)

    fun findDefinedFunction(name: String, arguments: List<Type> = emptyList()): Function? =
        functions.firstOrNull {
            it.name == name && it.canAcceptArguments(arguments)
        } ?: if(parentScope == null){
            val func = predefinedFunctions[name]
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