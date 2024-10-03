package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.FieldExpression
import com.huskerdev.gpkt.ast.FunctionStatement
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import com.huskerdev.gpkt.utils.useStack

class OpenCLProgram(
    private val cl: OpenCL,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val program: Long
    private val kernel: Long

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)

        program = cl.compileProgram(buffer.toString())
        kernel = cl.createKernel(program, "__m")
    }

    override fun executeRange(indexOffset: Int, instances: Int, vararg mapping: Pair<String, Any>) = useStack {
        val map = hashMapOf(*mapping)

        buffers.forEachIndexed { i, field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            if(value !is OpenCLMemoryPointer<*>){
                val buffer = when(value){
                    is Float -> floats(value)
                    is Double -> doubles(value)
                    is Int -> ints(value)
                    is Byte -> bytes(value)
                    else -> throw UnsupportedOperationException()
                }
                cl.setArgument(kernel, i, buffer)
            }else
                cl.setArgument(kernel, i, value.size, value.ptr)
        }
        cl.setArgument(kernel, buffers.size, ints(indexOffset)) // Index offset variable
        cl.executeKernel(kernel, instances.toLong())
        Unit
    }

    override fun dealloc() {
        cl.deallocProgram(program)
        cl.deallocKernel(kernel)
    }

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("M_PI")
            "E" -> buffer.append("M_E")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
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
            buffers.forEach {
                buffer.append("${it.name}=__v_${it.name};")
            }
            buffer.append("int ${function.arguments[0].name}=get_global_id(0)+__o;")
            stringifyScopeStatement(buffer, function.body, false)
            buffer.append("}")
        } else super.stringifyFunctionStatement(statement, buffer)
    }

    private fun transformKernelArg(field: Field): String{
        return if(field.type.isArray)
            "__global ${toCType(field.type)}*__v_${field.name}"
        else
            "${toCType(field.type)} __v_${field.name}"
    }

    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> "__global"
        Modifiers.CONST -> "__constant"
        Modifiers.READONLY -> ""
    }
}