package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.Scope


internal expect fun createSupportedInstance(vararg expectedEngine: GPType): GPEngine?

abstract class GPEngine(
    val type: GPType
) {
    companion object {
        @JvmStatic fun create(
            vararg expectedEngine: GPType = arrayOf(
                GPType.CUDA,
                GPType.OpenCL
            )
        ) = createSupportedInstance(*expectedEngine)
    }

    fun compile(code: String) =
        compile(GPAst.parse(code))

    abstract fun alloc(array: FloatArray): Source
    abstract fun alloc(length: Int): Source

    abstract fun compile(ast: Scope): Program
}

enum class GPType {
    OpenCL,
    CUDA
}