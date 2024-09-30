package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.TypesMismatchException
import com.huskerdev.gpkt.ast.FunctionStatement
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule


class CudaProgram(
    private val cuda: Cuda,
    ast: ScopeStatement
): SimpleCProgram(ast) {
    private val module: CUmodule
    private val function: CUfunction

    init {
        val buffer = StringBuilder()
        buffer.append("extern \"C\"{")
        stringifyScopeStatement(ast, buffer, false)
        buffer.append("}")

        module = cuda.compileToModule(buffer.toString())
        function = cuda.getFunctionPointer(module, "__m")
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        val instancesVal = Pointer.to(intArrayOf(instances))
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
                is CudaMemoryPointer -> Pointer.to(value.ptr)
                else -> throw UnsupportedOperationException()
            }
        }.toTypedArray()
        cuda.launch(function, instances, instancesVal, *arrays)
    }

    override fun dealloc() = Unit

    override fun stringifyFunctionStatement(statement: FunctionStatement, buffer: StringBuilder){
        val function = statement.function
        if(function.name == "main") {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__global__"),
                type = function.returnType.text,
                name = "__m",
                args = listOf("int __c") + buffers.map(::transformKernelArg)
            )
            buffer.append("{")
            buffer.append("int ${function.arguments[0].name}=blockIdx.x*blockDim.x+threadIdx.x;")
            buffer.append("if(${function.arguments[0].name}>__c)return;")
            buffers.forEach {
                buffer.append("${it.name}=__v_${it.name};")
            }
            stringifyScopeStatement(statement.function.body, buffer, false)
            buffer.append("}")
        }else {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__device__") + function.modifiers.map { it.text },
                type = function.returnType.text,
                name = function.name,
                args = function.arguments.map(::convertToFuncArg)
            )
            stringifyScopeStatement(statement.function.body, buffer, true)
        }
    }

    private fun transformKernelArg(field: Field): String{
        return if(field.type.isArray)
            "${toCType(field.type)}*__v_${field.name}"
        else
            "${toCType(field.type)} __v_${field.name}"
    }


    override fun toCModifier(modifier: Modifiers) = when(modifier){
        Modifiers.EXTERNAL -> "__device__"
        Modifiers.CONST -> "__constant__"
    }
}