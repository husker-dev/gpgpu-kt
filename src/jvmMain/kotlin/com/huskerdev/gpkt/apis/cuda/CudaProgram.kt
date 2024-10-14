package com.huskerdev.gpkt.apis.cuda

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.*
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule


class CudaProgram(
    private val context: CudaContext,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val cuda = context.cuda

    private val module: CUmodule
    private val function: CUfunction

    init {
        val buffer = StringBuilder()
        buffer.append("extern \"C\"{")
        stringifyScopeStatement(buffer, ast, false)
        buffer.append("}")

        module = cuda.compileToModule(context.peer, buffer.toString())
        function = cuda.getFunctionPointer(context.peer, module, "__m")
    }

    override fun executeRange(indexOffset: Int, instances: Int, map: Map<String, Any>) {
        val instancesVal = Pointer.to(intArrayOf(instances))
        val offsetVal = Pointer.to(intArrayOf(indexOffset))
        val arrays = buffers.map { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
            if(!areEqualTypes(value, field.type))
                throw TypesMismatchException(field.name)

            when(value){
                is Float -> Pointer.to(floatArrayOf(value))
                is Double -> Pointer.to(doubleArrayOf(value))
                is Long -> Pointer.to(longArrayOf(value))
                is Int -> Pointer.to(intArrayOf(value))
                is Byte -> Pointer.to(byteArrayOf(value))
                is CudaMemoryPointer<*> -> Pointer.to(value.ptr)
                else -> throw UnsupportedOperationException()
            }
        }.toTypedArray()
        cuda.launch(
            context.device.peer, context.peer,
            function, instances, instancesVal, offsetVal, *arrays)
    }

    override fun dealloc() = Unit

    override fun stringifyFieldExpression(buffer: StringBuilder, expression: FieldExpression) {
        when(expression.field.name){
            "PI" -> buffer.append("3.141592653589793")
            "E" -> buffer.append("2.718281828459045")
            else -> super.stringifyFieldExpression(buffer, expression)
        }
    }

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        if(function.name == "main") {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__global__"),
                type = function.returnType.text,
                name = "__m",
                args = listOf("int __c", "int __o") + buffers.map(::transformKernelArg)
            )
            buffer.append("{")
            buffer.append("const int ${function.arguments[0].name}=blockIdx.x*blockDim.x+threadIdx.x+__o;")
            buffer.append("if(${function.arguments[0].name}>__c+__o)return;")
            buffers.forEach {
                buffer.append("${it.name}=__v${it.name};")
            }
            stringifyScopeStatement(buffer, statement.function.body, false)
            buffer.append("}")
        }else {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__device__") + function.modifiers.map { it.text },
                type = convertToReturnType(function.returnType),
                name = function.name,
                args = function.arguments.map(::convertToFuncArg)
            )
            stringifyScopeStatement(buffer, statement.function.body, true)
        }
    }

    private fun transformKernelArg(field: Field): String{
        return if(field.type.isArray)
            "${toCType(field.type)}*__v${field.name}"
        else
            "${toCType(field.type)} __v${field.name}"
    }


    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> "__device__"
        Modifiers.CONST -> "__constant__"
        Modifiers.READONLY -> ""
    }
}