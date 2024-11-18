package com.huskerdev.gpkt.apis.js

import com.huskerdev.gpkt.GPContext
import com.huskerdev.gpkt.GPProgram
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.predefinedMathFields
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.apis.interpreter.CPUMemoryPointer
import com.huskerdev.gpkt.ast.objects.GPField
import com.huskerdev.gpkt.ast.objects.GPFunction
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.ast.types.FLOAT
import com.huskerdev.gpkt.utils.CProgramPrinter

class JSProgram(
    override val context: GPContext,
    ast: GPScope
): GPProgram(ast) {
    override var released = false
    private var source = JSProgramPrinter(ast, buffers, locals).stringify()

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

    override fun release() {
        if(released) return
        context.releaseProgram(this)
        released = true
    }
}

class JSProgramPrinter(
    ast: GPScope,
    buffers: List<GPField>,
    locals: List<GPField>
): CProgramPrinter(ast, buffers, locals,
    useLocalStruct = false
){
    override fun stringifyFunctionStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        statement: FunctionStatement
    ) {
        val function = statement.function
        val name = function.name

        if(name == "main") {
            buffer.append("for(let i=this.__o;i<this.__o+this.__c;i++){")
            stringifyScopeStatement(header, buffer, function.body!!, false)
            buffer.append("}")
        }else if(function.body != null) {
            buffer.append("function ").append(name).append("(")
            function.arguments.forEachIndexed { i, field ->
                buffer.append(field.name)
                if (i < function.arguments.lastIndex)
                    buffer.append(",")
            }
            buffer.append(")")
            stringifyScopeStatement(header, buffer, function.body!!, true)
        }
    }

    override fun stringifyMainFunctionDefinition(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) {
        buffer.append("for(let i=this.__o;i<this.__o+this.__c;i++)")
    }

    override fun stringifyMainFunctionBody(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        function: GPFunction
    ) = Unit

    override fun stringifyModifiersInStruct(field: GPField) = ""
    override fun stringifyModifiersInGlobal(obj: Any) = ""
    override fun stringifyModifiersInLocal(field: GPField) = ""
    override fun stringifyModifiersInArg(field: GPField) = ""

    override fun stringifyFieldStatement(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        fieldStatement: FieldStatement,
        force: Boolean
    ) {
        val modifiers = fieldStatement.fields[0].modifiers
        if(Modifiers.EXTERNAL in modifiers)
            return

        buffer.append("let ")
        fieldStatement.fields.forEachIndexed { i, field ->
            buffer.append(field.name)
            if(field.initialExpression != null){
                buffer.append("=")
                stringifyExpression(header, buffer, field.initialExpression!!)
            }
            if(i < fieldStatement.fields.lastIndex)
                buffer.append(",")
        }
        buffer.append(";")
    }

    override fun stringifyFieldExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: FieldExpression
    ) {
        if(expression.field in buffers)
            buffer.append("this.")
        if(expression.field.name in predefinedMathFields)
            buffer.append("Math.")
        super.stringifyFieldExpression(header, buffer, expression)
    }

    override fun convertPredefinedFunctionName(functionExpression: FunctionCallExpression) =
        "Math.${functionExpression.function.name}"

    override fun stringifyArrayAccessExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ArrayAccessExpression
    ) {
        if(expression.array.type.isDynamicArray)
            buffer.append("this.")
        super.stringifyArrayAccessExpression(header, buffer, expression)
    }

    override fun stringifyConstExpression(
        header: MutableMap<String, String>,
        buffer: StringBuilder,
        expression: ConstExpression
    ) {
        val type = expression.type

        buffer.append(expression.lexeme.text)
        if(type == FLOAT && "." !in expression.lexeme.text)
            buffer.append(".0")
    }
}