package com.huskerdev.gpkt.opencl

import com.huskerdev.gpkt.GPGPUEngine
import com.huskerdev.gpkt.Source
import com.huskerdev.gpkt.ast.objects.Scope

class OCLEngine: GPGPUEngine() {
    private val cl = OpenCL()

    override fun allocateImpl(array: FloatArray) =
        OCLSource(cl, array)

    override fun allocateImpl(length: Int) =
        OCLSource(cl, length)

    override fun deallocateImpl(source: Source) =
        (source as OCLSource).dealloc()

    override fun compile(ast: Scope) =
        OCLProgram(cl, ast)

}