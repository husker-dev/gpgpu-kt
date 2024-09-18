package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.Scope


internal expect fun createSupportedInstance(vararg expectedEngine: String): GPGPUEngine?

abstract class GPGPUEngine {
    companion object {
        @JvmStatic fun create(
            vararg expectedEngine: String = arrayOf("opencl")
        ) = createSupportedInstance(*expectedEngine)
    }

    val allocatedSources = mutableListOf<Source>()

    fun alloc(array: FloatArray): Source {
        val source = allocateImpl(array)
        allocatedSources += source
        return source
    }

    fun alloc(length: Int): Source {
        val source = allocateImpl(length)
        allocatedSources += source
        return source
    }

    fun dealloc(source: Source) {
        allocatedSources -= source
        deallocateImpl(source)
    }

    fun compile(code: String) =
        compile(GPGPUAst.parse(code))

    protected abstract fun allocateImpl(array: FloatArray): Source
    protected abstract fun allocateImpl(length: Int): Source
    protected abstract fun deallocateImpl(source: Source)

    protected abstract fun compile(ast: Scope): Program
}