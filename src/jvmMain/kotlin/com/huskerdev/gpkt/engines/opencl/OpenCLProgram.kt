package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import org.jocl.cl_kernel
import org.jocl.cl_program

class OpenCLProgram(
    private val cl: OpenCL,
    ast: Scope
): SimpleCProgram(ast) {
    private val program: cl_program
    private val kernel: cl_kernel

    init {
        val buffer = StringBuilder()
        stringifyScope(ast, buffer)

        program = cl.compileProgram(buffer.toString())
        kernel = cl.createKernel(program, "__m")
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        val map = hashMapOf(*mapping)
        buffers.forEachIndexed { i, it ->
            if(it !in map) throw Exception("Source '$it' have not been set")
            cl.setArgument(kernel, i, map[it] as OpenCLSource)
        }
        cl.executeKernel(kernel, instances.toLong())
    }

    override fun dealloc() {
        cl.dealloc(program)
        cl.dealloc(kernel)
    }

    override fun stringifyFunction(function: Function, buffer: StringBuilder){
        if(function.name == "main"){
            appendCFunctionHeader(
                buffer = buffer,
                modifiers = listOf("__kernel"),
                type = function.returnType.text,
                name = "__m",
                args = buffers.map { "__global float*${it}" }
            )
            buffer.append("int ${function.arguments[0].name}=get_global_id(0);")
            stringifyScope(function, buffer)
            buffer.append("}")
        } else super.stringifyFunction(function, buffer)
    }

}