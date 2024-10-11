package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.ArrayAccessExpression
import com.huskerdev.gpkt.ast.FieldExpression
import com.huskerdev.gpkt.ast.FieldStatement
import com.huskerdev.gpkt.ast.FunctionCallExpression
import com.huskerdev.gpkt.ast.FunctionStatement
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.predefinedMathFunctions
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader


class OpenCLProgram(
    private val context: OpenCLContext,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val cl = context.opencl
    private val program: CLProgram
    private val kernel: CLKernel

    init {
        val buffer = StringBuilder()

        buffer.append("typedef struct{")
        buffers.joinTo(buffer, separator = ";", postfix = ";") { convertToFuncArg(it) }
        buffer.append("}__in;")

        stringifyScopeStatement(buffer, ast, false)

        program = cl.compileProgram(context.device.peer, context.peer, buffer.toString())
        kernel = cl.createKernel(program, "__m")
    }

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        buffers.forEachIndexed { i, field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            when(value){
                is Float -> cl.setArgument1f(kernel, i, value)
                is Double -> cl.setArgument1d(kernel, i, value)
                is Int -> cl.setArgument1i(kernel, i, value)
                is Byte -> cl.setArgument1b(kernel, i, value)
                is OpenCLMemoryPointer<*> -> cl.setArgument(kernel, i, value.mem)
                else -> throw UnsupportedOperationException()
            }
        }
        cl.setArgument1i(kernel, buffers.size, indexOffset) // Set index offset variable
        cl.executeKernel(context.commandQueue, kernel, context.device.peer, instances.toLong())
    }

    override fun dealloc() {
        cl.deallocProgram(program)
        cl.deallocKernel(kernel)
    }

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        if(expression.field in buffers)
            buffer.append("__v.")
        when(expression.field.name){
            "PI" -> buffer.append("M_PI")
            "E" -> buffer.append("M_E")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }

    override fun stringifyFieldStatement(fieldStatement: FieldStatement, buffer: StringBuilder) {
        if(Modifiers.EXTERNAL !in fieldStatement.fields[0].modifiers)
            super.stringifyFieldStatement(fieldStatement, buffer)
    }

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        if(function.name == "main"){
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__kernel"),
                type = function.returnType.text,
                name = "__m",
                args = buffers.map(::transformKernelArg) + listOf("int __o")
            )
            buffer.append("{")

            // Struct with inputs
            buffer.append("__in __v={")
            buffers.forEachIndexed { index, field ->
                buffer.append("__v").append(field.name)
                if(index != buffers.lastIndex)
                    buffer.append(",")
            }
            buffer.append("};")

            // Index
            buffer.append("int ${function.arguments[0].name}=get_global_id(0)+__o;")

            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        } else {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = function.modifiers.map { it.text },
                type = convertToReturnType(function.returnType),
                name = function.name,
                args = listOf("__in __v") + function.arguments.map(::convertToFuncArg)
            )
            stringifyStatement(buffer, statement.function.body)
        }
    }

    override fun stringifyArrayAccessExpression(
        buffer: StringBuilder,
        expression: ArrayAccessExpression
    ) {
        if(Modifiers.EXTERNAL in expression.array.modifiers)
            buffer.append("__v.")
        super.stringifyArrayAccessExpression(buffer, expression)
    }

    override fun stringifyFunctionCallExpression(
        buffer: StringBuilder,
        expression: FunctionCallExpression
    ) {
        buffer.append(expression.function.name)
        buffer.append("(")
        if(expression.function.name !in predefinedMathFunctions) {
            buffer.append("__v")
            if (expression.arguments.isNotEmpty())
                buffer.append(",")
        }

        expression.arguments.forEachIndexed { i, arg ->
            stringifyExpression(buffer, arg)
            if(i != expression.arguments.lastIndex)
                buffer.append(",")
        }
        buffer.append(")")
    }

    private fun transformKernelArg(field: Field): String{
        return if(field.type.isArray)
            "${toCType(field.type)}*__v${field.name}"
        else
            "${toCType(field.type)} __v${field.name}"
    }

    override fun toCType(type: Type) = when(type){
        Type.FLOAT_ARRAY -> "__global float"
        Type.INT_ARRAY -> "__global int"
        Type.BYTE_ARRAY -> "__global char"
        Type.BOOLEAN_ARRAY -> "__global bool"
        else -> super.toCType(type)
    }

    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> ""
        Modifiers.CONST -> "__constant"
        Modifiers.READONLY -> ""
    }
}