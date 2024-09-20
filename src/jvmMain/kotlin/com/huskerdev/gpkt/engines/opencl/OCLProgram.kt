package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import org.jocl.cl_kernel
import org.jocl.cl_program

class OCLProgram(
    private val cl: OpenCL,
    ast: Scope
): SimpleCProgram(ast) {
    private val program: cl_program
    private val kernel: cl_kernel

    init {
        val buffer = StringBuilder()
        stringifyScope(ast, buffer)

        program = cl.compileProgram(buffer.toString())
        kernel = cl.createKernel(program, "_m")
    }

    override fun execute(instances: Int, vararg mapping: Pair<String, Source>) {
        mapping.forEach { (key, value) ->
            cl.setArgument(kernel, buffers.indexOf(key), value as OCLSource)
        }
        cl.executeKernel(kernel, instances.toLong())
    }

    override fun stringifyFunction(function: Function, buffer: StringBuilder, additionalModifier: String?){
        if(function.name == "main"){
            buffer.append("__kernel ")
            stringifyModifiers(function.modifiers, buffer)
            buffer.append(function.returnType.text)
            buffer.append(" ")
            buffer.append("_m")
            buffer.append("(")
            buffer.append(buffers.joinToString(",") {
                "__global float*${it}"
            })
            buffer.append("){")
            buffer.append("int ${function.arguments[0].name}=get_global_id(0);")
            stringifyScope(function, buffer, function.arguments)
            buffer.append("}")
        } else super.stringifyFunction(function, buffer, additionalModifier)
    }

}