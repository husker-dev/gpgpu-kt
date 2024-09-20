package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.engines.cpu.CPUEngine


internal expect fun createSupportedInstance(vararg expectedEngine: GPType): GPEngine?

abstract class GPEngine(
    val type: GPType
) {
    companion object {
        fun create(
            vararg expectedEngine: GPType = arrayOf(
                GPType.CUDA,
                GPType.OpenCL,
                GPType.Interpreter
            )
        ) = createSupportedInstance(*expectedEngine) ?:
            if(GPType.Interpreter in expectedEngine) CPUEngine() else null
    }

    fun compile(code: String) =
        compile(GPAst.parse(code))

    abstract fun alloc(array: FloatArray): Source
    abstract fun alloc(length: Int): Source

    abstract fun compile(ast: Scope): Program
}

enum class GPType {
    OpenCL,
    CUDA,
    Interpreter
}