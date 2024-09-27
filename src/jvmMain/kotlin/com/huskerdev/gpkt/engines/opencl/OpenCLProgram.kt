package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.FieldNotSetException
import com.huskerdev.gpkt.SimpleCProgram
import com.huskerdev.gpkt.ast.objects.Function
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.ast.types.Type
import com.huskerdev.gpkt.utils.appendCFunctionHeader
import jcuda.Sizeof
import org.jocl.Pointer
import org.jocl.cl_kernel
import org.jocl.cl_mem
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

    override fun execute(instances: Int, vararg mapping: Pair<String, Any>) {
        val map = hashMapOf(*mapping)
        val variables = arrayListOf<cl_mem>()

        buffers.forEachIndexed { i, field ->
            val value = map.getOrElse(field.name) { throw FieldNotSetException(field.name) }

            // Get pointer if value is OpenCLMemoryPointer, or create cl_mem if value is float,int etc.
            val ptr = if(value !is OpenCLMemoryPointer)
                allocVariable(value).apply { variables += this }
            else value.ptr
            cl.setArgument(kernel, i, ptr)
        }
        cl.executeKernel(kernel, instances.toLong())

        // Free allocated memory for variables
        variables.forEach { cl.dealloc(it) }
    }

    private fun allocVariable(value: Any) = when(value){
        is Float -> cl.allocate(Pointer.to(floatArrayOf(value)), 1L * Sizeof.FLOAT)
        is Double -> cl.allocate(Pointer.to(doubleArrayOf(value)), 1L * Sizeof.DOUBLE)
        is Long -> cl.allocate(Pointer.to(longArrayOf(value)), 1L * Sizeof.LONG)
        is Int -> cl.allocate(Pointer.to(intArrayOf(value)), 1L * Sizeof.INT)
        is Byte -> cl.allocate(Pointer.to(byteArrayOf(value)), 1L * Sizeof.BYTE)
        else -> throw UnsupportedOperationException()
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
                args = buffers.map { "__global ${toCType(it.type, false, it.name)}" }
            )
            buffer.append("int ${function.arguments[0].name}=get_global_id(0);")
            stringifyScope(function, buffer)
            buffer.append("}")
        } else super.stringifyFunction(function, buffer)
    }

    override fun toCType(type: Type, isConst: Boolean, name: String) = if(isConst)
        "__constant " + super.toCType(type, false, name)
    else super.toCType(type, false, name)

}