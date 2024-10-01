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

    abstract fun allocFloat(array: FloatArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): FloatMemoryPointer
    abstract fun allocFloat(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): FloatMemoryPointer

    abstract fun allocDouble(array: DoubleArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): DoubleMemoryPointer
    abstract fun allocDouble(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): DoubleMemoryPointer

    abstract fun allocLong(array: LongArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): LongMemoryPointer
    abstract fun allocLong(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): LongMemoryPointer

    abstract fun allocInt(array: IntArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): IntMemoryPointer
    abstract fun allocInt(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): IntMemoryPointer

    abstract fun allocByte(array: ByteArray, usage: MemoryUsage = MemoryUsage.READ_WRITE): ByteMemoryPointer
    abstract fun allocByte(length: Int, usage: MemoryUsage = MemoryUsage.READ_WRITE): ByteMemoryPointer

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