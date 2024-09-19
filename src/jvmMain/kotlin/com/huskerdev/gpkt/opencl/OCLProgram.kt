package com.huskerdev.gpkt.opencl

import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import org.jocl.cl_kernel
import org.jocl.cl_program
import kotlin.math.max

class OCLProgram(
    private val cl: OpenCL,
    ast: Scope
): SimpleCProgram(ast) {
    private val program: cl_program
    private val kernel: cl_kernel

    init {
        val buffer = StringBuffer()
        stringifyScope(ast, buffer)
        println(buffer.toString())

        program = cl.compileProgram(buffer.toString())
        kernel = cl.createKernel(program, "_m")
    }

    override fun execute(vararg mapping: Pair<String, Source>) {
        var maxSize = 0L
        mapping.forEach { (key, value) ->
            if(key !in buffers)
                throw Exception("Buffer $key is not defined in program")
            cl.setArgument(kernel, buffers.indexOf(key), value as OCLSource)
            maxSize = max(maxSize, value.length.toLong())
        }
        cl.executeKernel(kernel, maxSize)
    }

    override fun stringifyFunction(function: Function, buffer: StringBuffer, additionalModifier: String?){
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