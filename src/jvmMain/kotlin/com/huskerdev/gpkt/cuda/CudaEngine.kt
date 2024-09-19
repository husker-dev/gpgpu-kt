package com.huskerdev.gpkt.cuda

import com.huskerdev.gpkt.GPEngine
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.objects.Scope

class CudaEngine: GPEngine(GPType.CUDA) {
    private val cuda = Cuda()

    override fun alloc(array: FloatArray) =
        CudaSource(cuda, cuda.alloc(array), array.size)

    override fun alloc(length: Int) =
        CudaSource(cuda, cuda.alloc(length), length)

    override fun compile(ast: Scope) =
        CudaProgram(cuda, ast)
}