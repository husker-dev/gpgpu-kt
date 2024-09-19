package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class OCLEngine: GPEngine(GPType.OpenCL) {
    private val cl = OpenCL()

    override fun alloc(array: FloatArray) =
        OCLSource(cl, cl.allocate(array), array.size)

    override fun alloc(length: Int) =
        OCLSource(cl, cl.allocate(length), length)

    override fun compile(ast: Scope) =
        OCLProgram(cl, ast)

}