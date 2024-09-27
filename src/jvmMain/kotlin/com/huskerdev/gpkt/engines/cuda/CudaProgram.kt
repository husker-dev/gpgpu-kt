package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.CUmodule


class CudaProgram(
    private val cuda: Cuda,
    ast: Scope
): SimpleCProgram(ast) {
    private val module: CUmodule
    private val function: CUfunction

    init {
        val buffer = StringBuilder()
        buffer.append("extern \"C\"{")
        stringifyScope(ast, buffer)
        buffer.append("}")

        module = cuda.compileToModule(buffer.toString())
        function = cuda.getFunctionPointer(module, "__m")
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)

        val instancesVal = Pointer.to(intArrayOf(instances))
        val arrays = buffers.map { field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }
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

    override fun stringifyFunction(function: Function, buffer: StringBuilder){
        if(function.name == "main") {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__global__"),
                type = function.returnType.text,
                name = "__m",
                args = listOf("int __c") + buffers.map { toCType(it.type, false, it.name) }
            )
            buffer.append("int ${function.arguments[0].name}=blockIdx.x*blockDim.x+threadIdx.x;")
            buffer.append("if(${function.arguments[0].name}>__c)return;")
            stringifyScope(function, buffer)
            buffer.append("}")
        }else {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__device__") + function.modifiers.map { it.text },
                type = function.returnType.text,
                name = function.name,
                args = function.arguments.map { "${toCType(it.type, false)} ${it.name}" }
            )
            stringifyScope(function, buffer)
            buffer.append("}")
        }
    }

    override fun toCType(type: Type, isConst: Boolean, name: String) = if(isConst)
        "__constant__ " + super.toCType(type, false, name)
    else super.toCType(type, false, name)
}