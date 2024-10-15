package com.huskerdev.gpkt.apis.js

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.predefinedMathFields
import com.huskerdev.gpkt.ast.objects.predefinedMathFunctions
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.apis.interpreter.CPUMemoryPointer

class JSProgram(ast: ScopeStatement): SimpleCProgram(ast) {

    private var source: String

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)

        println(buffer.toString())
        source = buffer.toString()
    }

    override fun executeRange(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val scope = js("{}")
        scope["__o"] = indexOffset
        scope["__c"] = instances

        buffers.forEach { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            scope[field.name] = when(value){
                is CPUMemoryPointer<*> -> value.array
                is Float, is Double, is Long, is Int, is Byte -> value
                else -> throw UnsupportedOperationException()
            }
        }
        js("Function")("\"use strict\";$source").bind(scope)()
    }

    override fun dealloc() = Unit

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder) {
        val function = statement.function
        val name = function.name

        if(name == "main") {
            buffer.append("for(let i=this.__o;i<this.__o+this.__c;i++){")
            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        }else {
            buffer.append("function ").append(name).append("(")
            function.arguments.forEachIndexed { i, field ->
                buffer.append(field.name)
                if(i < function.arguments.lastIndex)
                    buffer.append(",")
            }
            buffer.append(")")
            stringifyScopeStatement(buffer, function.body, true)
        }
    }

    override fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder) {
        val modifiers = fieldStatement.fields[0].modifiers
        if(Modifiers.EXTERNAL in modifiers)
            return

        buffer.append("let ")
        fieldStatement.fields.forEachIndexed { i, field ->
            buffer.append(field.name)
            if(field.initialExpression != null){
                buffer.append("=")
                stringifyExpression(buffer, field.initialExpression!!)
            }
            if(i < fieldStatement.fields.lastIndex)
                buffer.append(",")
        }
        buffer.append(";")
    }

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        if(expression.field in buffers)
            buffer.append("this.")
        if(expression.field.name in predefinedMathFields)
            buffer.append("Math.")
        super.stringifyFieldExpression(buffer, expression)
    }

    override fun stringifyFunctionCallExpression(buffer: StringBuilder, expression: FunctionCallExpression) {
        if(expression.function.name in predefinedMathFunctions)
            buffer.append("Math.")
        super.stringifyFunctionCallExpression(buffer, expression)
    }

    override fun stringifyArrayAccessExpression(buffer: StringBuilder, expression: ArrayAccessExpression) {
        if(expression.array in buffers)
            buffer.append("this.")
        super.stringifyArrayAccessExpression(buffer, expression)
    }

    override fun stringifyConstExpression(buffer: StringBuilder, expression: ConstExpression) {
        val type = expression.type

        buffer.append(expression.lexeme.text)
        if(type == Type.FLOAT && "." !in expression.lexeme.text)
            buffer.append(".0")
    }
}