package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.engines.cpu.CPUDevice

internal expect val defaultExpectedTypes: Array<GPType>
internal expect val defaultExpectedDeviceId: Int

internal expect fun createSupportedInstance(
    requestedDeviceId: Int,
    vararg requestedType: GPType
): GPDevice?

abstract class GPDevice(
    val type: GPType
) {
    companion object {
        fun create(
            requestedDeviceId: Int = defaultExpectedDeviceId,
            vararg requestedType: GPType = defaultExpectedTypes,
        ) = createSupportedInstance(requestedDeviceId, *requestedType) ?:
            if(GPType.Interpreter in requestedType) CPUDevice() else null
    }

    abstract val id: Int
    abstract val name: String
    abstract val isGPU: Boolean

    fun compile(code: String) =
        compile(GPAst.parse(code))

    abstract fun alloc(array: FloatArray): Source
    abstract fun alloc(length: Int): Source

    abstract fun compile(ast: Scope): Program
}

enum class GPType(
    val shortName: String
) {
    OpenCL("opencl"),
    CUDA("cuda"),
    Interpreter("interpreter"),
    Javac("javac")
    ;
    companion object {
        val mapped = entries.associateBy { it.shortName }
    }
}