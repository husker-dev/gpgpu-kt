package com.huskerdev.gpkt.apis.js

import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.predefinedMathFields
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.apis.interpreter.CPUMemoryPointer
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.types.FLOAT
import com.huskerdev.gpkt.utils.SimpleCProgram

class JSProgram(ast: ScopeStatement): SimpleCProgram(ast, false) {

    private var source: String

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)

        println(buffer.toString())
        source = buffer.toString()
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val scope = js("{}")
        scope["__o"] = indexOffset
        scope["__c"] = instances

        buffers.forEach { field ->
            scope[field.name] = when(val value = map[field.name]!!){
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
            stringifyScopeStatement(buffer, function.body!!, false)
            buffer.append("}")
        }else if(function.body != null) {
            buffer.append("function ").append(name).append("(")
            function.arguments.forEachIndexed { i, field ->
                buffer.append(field.name)
                if (i < function.arguments.lastIndex)
                    buffer.append(",")
            }
            buffer.append(")")
            stringifyScopeStatement(buffer, function.body!!, true)
        }
    }

    override fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: GPFunction) {
        buffer.append("for(let i=this.__o;i<this.__o+this.__c;i++)")
    }

    override fun stringifyMainFunctionBody(buffer: StringBuilder, function: GPFunction) = Unit

    override fun stringifyModifiersInStruct(field: Field) = ""
    override fun stringifyModifiersInGlobal(obj: Any) = ""
    override fun stringifyModifiersInLocal(field: Field) = ""
    override fun stringifyModifiersInArg(field: Field) = ""

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

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) =
        "Math.${functionExpression.function.name}"

    override fun stringifyArrayAccessExpression(buffer: StringBuilder, expression: ArrayAccessExpression) {
        if(expression.array.type.isDynamicArray)
            buffer.append("this.")
        super.stringifyArrayAccessExpression(buffer, expression)
    }

    override fun stringifyConstExpression(buffer: StringBuilder, expression: ConstExpression) {
        val type = expression.type

        buffer.append(expression.lexeme.text)
        if(type == FLOAT && "." !in expression.lexeme.text)
            buffer.append(".0")
    }
}