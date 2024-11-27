package com.huskerdev.gpkt.ast.objects

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.lexer.Lexeme
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.PrimitiveType
import com.huskerdev.gpkt.ast.types.VOID
import com.huskerdev.gpkt.utils.Dictionary

open class GPScope(
    val context: GPContext?,
    val parentScope: GPScope?,
    val dictionary: Dictionary,
    val returnType: PrimitiveType? = null,
    val iterable: Boolean = false,
    val fields: LinkedHashMap<String, GPField> = linkedMapOf(),
    val functions: LinkedHashMap<String, GPFunction> = linkedMapOf(),
    val classes: LinkedHashMap<String, GPClass> = linkedMapOf(),
) {
    val statements: MutableList<Statement> = mutableListOf()

    var returns: Boolean = false
        private set

    fun findField(name: String): GPField? =
        fields[name]
        ?: parentScope?.findField(name)
        ?: allPredefinedFields[name]

    fun findFunction(name: String): GPFunction? =
        functions[name]
        ?: parentScope?.findFunction(name)
        ?: allPredefinedFunctions[name]

    fun findClass(name: String): GPClass? =
        classes[name]
        ?: parentScope?.findClass(name)

    fun findReturnType(): PrimitiveType =
        returnType
        ?: parentScope?.findReturnType()
        ?: VOID

    fun isInIterableScope(): Boolean =
        iterable || parentScope?.isInIterableScope() == true

    fun addField(field: GPField, lexeme: Lexeme?, codeBlock: String?){
        if(lexeme != null && codeBlock != null && findField(field.name) != null)
            throw nameAlreadyDefinedException(field.name, lexeme, codeBlock)
        fields[field.name] = field
    }

    fun addFunction(function: GPFunction, lexeme: Lexeme?, codeBlock: String?){
        val defined = findFunction(function.name)
        if(defined == function) return
        if(lexeme != null && codeBlock != null && defined?.body != null)
            throw nameAlreadyDefinedException(function.name, lexeme, codeBlock)

        functions[function.name] = function
    }

    fun addClass(classObj: GPClass, lexeme: Lexeme?, codeBlock: String?){
        val defined = findClass(classObj.name)
        if(defined == classObj) return
        if(lexeme != null && codeBlock != null && defined != null)
            throw nameAlreadyDefinedException(classObj.name, lexeme, codeBlock)
        classes[classObj.name] = classObj
    }

    fun addStatement(statement: Statement, lexeme: Lexeme? = null, codeBlock: String? = null) =
        addStatement(-1, statement, lexeme, codeBlock)

    fun addStatement(index: Int, statement: Statement, lexeme: Lexeme? = null, codeBlock: String? = null){
        if(statement.returns)
            returns = true
        when (statement) {
            is FieldStatement -> {
                statement.fields.forEach { field ->
                    addField(field, lexeme, codeBlock)
                    if(parentScope == null && field.modifiers.isEmpty())
                        field.modifiers += Modifiers.THREADLOCAL
                }
            }
            is FunctionStatement ->
                addFunction(statement.function, lexeme, codeBlock)
            is ClassStatement ->
                addClass(statement.classObj, lexeme, codeBlock)
        }
        if(index == -1)
            statements += statement
        else
            statements.add(index, statement)
    }

    fun clone(scope: GPScope?) =
        GPScope(context, scope, scope?.dictionary ?: Dictionary(), returnType, iterable).apply {
            statements.forEach { addStatement(it.clone(this), null, null) }
        }
}