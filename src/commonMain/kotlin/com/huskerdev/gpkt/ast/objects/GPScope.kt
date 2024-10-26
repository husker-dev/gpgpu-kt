package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.VOID

open class GPScope(
    val context: GPContext?,
    val parentScope: GPScope?,
    val returnType: PrimitiveType? = null,
    val iterable: Boolean = false,
    val modules: LinkedHashSet<ScopeStatement> = linkedSetOf(),
    val fields: MutableList<Field> = mutableListOf(),
    val functions: MutableList<GPFunction> = mutableListOf(),
) {

    fun findDefinedField(name: String): Field? =
        fields.firstOrNull {
            it.name == name
        } ?: modules.firstNotNullOfOrNull {
            it.scope.findDefinedField(name)
        } ?: if(parentScope == null)
            allPredefinedFields[name]
        else parentScope.findDefinedField(name)

    fun findDefinedFunction(name: String, arguments: List<PrimitiveType> = emptyList()): GPFunction? =
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

    fun findReturnType(): PrimitiveType =
        returnType ?: parentScope?.findReturnType() ?: VOID

    fun isInIterableScope(): Boolean = if(iterable)
        true
    else parentScope?.isInIterableScope() ?: false

    private fun checkAvailableName(name: String, lexeme: Lexeme, codeBlock: String){
        if(findDefinedFunction(name)?.body != null)
            throw functionAlreadyDefinedException(name, lexeme, codeBlock)
        if(findDefinedField(name) != null)
            throw variableAlreadyDefinedException(name, lexeme, codeBlock)
    }

    fun addField(field: Field, lexeme: Lexeme, codeBlock: String){
        checkAvailableName(field.name, lexeme, codeBlock)
        fields += field
    }

    fun addFunction(function: GPFunction, lexeme: Lexeme, codeBlock: String){
        checkAvailableName(function.name, lexeme, codeBlock)
        val definedFunction = functions.find { it.name == function.name && it.body == null }
        if(definedFunction != null){
            if(definedFunction.returnType != function.returnType)
                throw compilationError("Function type does not match with previously defined", lexeme, codeBlock)
            if(definedFunction.argumentsTypes != function.argumentsTypes)
                throw compilationError("Function arguments do not match with previously defined", lexeme, codeBlock)
            functions[functions.indexOf(definedFunction)] = function
        }else
            functions += function
    }
}