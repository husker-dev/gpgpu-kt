package com.huskerdev.gpkt.apis.opencl

import com.huskerdev.gpkt.ast.FieldExpression
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.utils.SimpleCProgram
import com.huskerdev.gpkt.utils.appendCFunctionDefinition


class OpenCLProgram(
    private val context: OpenCLContext,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val cl = context.opencl
    private val program: CLProgram
    private val kernel: CLKernel

    init {
        val buffer = StringBuilder()
        stringifyScopeStatement(buffer, ast, false)

        program = cl.compileProgram(context.device.peer, context.peer, buffer.toString())
        kernel = cl.createKernel(program, "__m")
    }

    override fun executeRangeImpl(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        buffers.forEachIndexed { i, field ->
            when(val value = map[field.name]!!){
                is Float -> cl.setArgument1f(kernel, i, value)
                is Int -> cl.setArgument1i(kernel, i, value)
                is Byte -> cl.setArgument1b(kernel, i, value)
                is OpenCLMemoryPointer<*> -> cl.setArgument(kernel, i, value.mem)
                else -> throw UnsupportedOperationException()
            }
        }
        cl.setArgument1i(kernel, buffers.size, indexOffset) // Set index offset variable
        cl.executeKernel(context.commandQueue, kernel, context.device.peer, instances.toLong())
    }

    override fun stringifyModifiersInStruct(field: Field) =
        stringifyModifiersInLocal(field)

    override fun stringifyModifiersInGlobal(obj: Any) =
        if(obj is Field && obj.isConstant) "__constant"
        else if(obj is Function && obj.returnType.isArray) "__global" else ""

    override fun stringifyModifiersInLocal(field: Field) =
        if(field.type.isArray) "__global" else ""

    override fun stringifyModifiersInArg(field: Field) =
        stringifyModifiersInLocal(field)

    override fun dealloc() {
        cl.deallocProgram(program)
        cl.deallocKernel(kernel)
    }

    override fun stringifyMainFunctionDefinition(buffer: StringBuilder, function: Function) {
        buffer.append("__kernel ")
        appendCFunctionDefinition(
            buffer = buffer,
            type = function.returnType.text,
            name = "__m",
            args = buffers.map {
                if(it.type.isArray) "__global ${toCType(it.type)}*__v${it.name}"
                else "${toCType(it.type)} __v${it.name}"
            } + listOf("int __o")
        )
    }

    override fun stringifyMainFunctionBody(buffer: StringBuilder, function: Function) {
        buffer.append("int ${function.arguments[0].name}=get_global_id(0)+__o;")
    }

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("M_PI")
            "E" -> buffer.append("M_E")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }
}