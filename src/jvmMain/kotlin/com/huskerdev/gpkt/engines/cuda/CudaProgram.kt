package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.FieldStatement
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers
import com.huskerdev.gpkt.utils.appendCFieldHeader
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

    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val map = hashMapOf(*mapping)

        val countVal = Pointer.to(intArrayOf(instances))
        val arrays = buffers.map {
            if(it !in map) throw Exception("Source '$it' have not been set")
            else (map[it] as CudaSource).ptr
        }.toTypedArray()

        cuda.launch(function, instances, countVal, *arrays)
    }

    override fun dealloc() = Unit

    override fun stringifyFunction(function: Function, buffer: StringBuilder){
        if(function.name == "main") {
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__global__"),
                type = function.returnType.text,
                name = "__m",
                args = listOf("int __c") + buffers.map { "float*${it}" }
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
                args = function.arguments.map { "${it.type.toCType(false)} ${it.name}" }
            )
            stringifyScope(function, buffer)
            buffer.append("}")
        }
    }

    override fun stringifyFieldStatement(
        fieldStatement: FieldStatement,
        buffer: StringBuilder
    ) {
        val modifiers = fieldStatement.fields[0].modifiers
        val type = fieldStatement.fields[0].type
        if(Modifiers.IN !in modifiers && Modifiers.OUT !in modifiers && fieldStatement.scope.parentScope == null){
            appendCFieldHeader(
                buffer = buffer,
                modifiers = listOf("__constant__") + modifiers.map { it.text },
                type = type.toCType(true),
                fields = fieldStatement.fields,
                expressionGen = { stringifyExpression(buffer, it) }
            )
        } else super.stringifyFieldStatement(fieldStatement, buffer)
    }
}