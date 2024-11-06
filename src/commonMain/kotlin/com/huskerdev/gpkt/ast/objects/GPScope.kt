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
    val fields: LinkedHashMap<String, GPField> = linkedMapOf(),
    val functions: LinkedHashMap<String, GPFunction> = linkedMapOf(),
    val classes: LinkedHashMap<String, GPClass> = linkedMapOf(),
) {

    fun findDefinedField(name: String): GPField? =
        fields[name]
        ?: parentScope?.findDefinedField(name)
        ?: allPredefinedFields[name]
        ?: modules.firstNotNullOfOrNull { it.scope.findDefinedField(name) }

    fun findDefinedFunction(name: String): GPFunction? =
        functions[name]
        ?: parentScope?.findDefinedFunction(name)
        ?: allPredefinedFunctions[name]
        ?: modules.firstNotNullOfOrNull { it.scope.findDefinedFunction(name) }

    fun findDefinedClass(name: String): GPClass? =
        classes[name]
        ?: parentScope?.findDefinedClass(name)
        ?: modules.firstNotNullOfOrNull { it.scope.findDefinedClass(name) }

    fun findReturnType(): PrimitiveType =
        returnType
        ?: parentScope?.findReturnType()
        ?: VOID

    fun isInIterableScope(): Boolean =
        iterable || parentScope?.isInIterableScope() ?: false

    fun addField(field: GPField, lexeme: Lexeme, codeBlock: String){
        if(findDefinedField(field.name) != null)
            throw nameAlreadyDefinedException(field.name, lexeme, codeBlock)
        fields[field.name] = field
    }

    fun addFunction(function: GPFunction, lexeme: Lexeme, codeBlock: String){
        val defined = findDefinedFunction(function.name)
        if(defined == function) return
        if(defined?.body != null) throw nameAlreadyDefinedException(function.name, lexeme, codeBlock)

        functions[function.name] = function
    }

    fun addClass(classObj: GPClass, lexeme: Lexeme, codeBlock: String){
        val defined = findDefinedClass(classObj.name)
        if(defined == classObj) return
        if(defined != null)
            throw nameAlreadyDefinedException(classObj.name, lexeme, codeBlock)
        classes[classObj.name] = classObj
    }
}