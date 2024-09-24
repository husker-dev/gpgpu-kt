package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.FieldStatement
import com.huskerdev.gpkt.ast.objects.Field
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Modifiers
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
        function = cuda.getFunctionPointer(module, "_m")
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

    override fun stringifyFunction(function: Function, buffer: StringBuilder, additionalModifier: String?){
        if(function.name == "main") {
            buffer.append("__global__ ")
            stringifyModifiers(function.modifiers, buffer)
            buffer.append(function.returnType.text)
            buffer.append(" ")
            buffer.append("_m")
            buffer.append("(")
            buffer.append((arrayOf("int __c") + buffers.map { "float*${it}" }).joinToString(", "))
            buffer.append("){")
            buffer.append("int ${function.arguments[0].name}=blockIdx.x*blockDim.x+threadIdx.x;")
            buffer.append("if(${function.arguments[0].name}>__c)return;")
            stringifyScope(function, buffer, function.arguments)
            buffer.append("}")
        }else super.stringifyFunction(function, buffer, "__device__")
    }

    override fun stringifyFieldStatement(
        fieldStatement: FieldStatement,
        buffer: StringBuilder,
        ignoredFields: List<Field>?
    ) {
        val modifiers = fieldStatement.fields[0].modifiers
        if(Modifiers.IN !in modifiers && Modifiers.OUT !in modifiers && fieldStatement.scope.parentScope == null)
            buffer.append("__constant__ ")
        super.stringifyFieldStatement(fieldStatement, buffer, ignoredFields)
    }
}