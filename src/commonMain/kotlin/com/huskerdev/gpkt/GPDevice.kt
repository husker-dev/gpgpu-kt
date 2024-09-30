package com.huskerdev.gpkt

import com.huskerdev.gpkt.ast.ScopeStatement
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

    val modules = GPModules(this)

    fun compile(code: String) =
        compile(GPAst.parse(code, this))

    abstract fun allocFloat(array: FloatArray): FloatMemoryPointer
    abstract fun allocFloat(length: Int): FloatMemoryPointer

    abstract fun allocDouble(array: DoubleArray): DoubleMemoryPointer
    abstract fun allocDouble(length: Int): DoubleMemoryPointer

    abstract fun allocLong(array: LongArray): LongMemoryPointer
    abstract fun allocLong(length: Int): LongMemoryPointer

    abstract fun allocInt(array: IntArray): IntMemoryPointer
    abstract fun allocInt(length: Int): IntMemoryPointer

    abstract fun allocByte(array: ByteArray): ByteMemoryPointer
    abstract fun allocByte(length: Int): ByteMemoryPointer

    abstract fun compile(ast: ScopeStatement): Program
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