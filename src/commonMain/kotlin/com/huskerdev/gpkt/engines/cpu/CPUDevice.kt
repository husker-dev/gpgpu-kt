package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.GPDevice
import com.huskerdev.gpkt.GPType
import com.huskerdev.gpkt.ast.ScopeStatement

class CPUDevice: GPDevice(GPType.Interpreter) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun compile(ast: ScopeStatement) =
        CPUProgram(ast)

    override fun allocFloat(array: FloatArray) =
        CPUFloatMemoryPointer(array)

    override fun allocFloat(length: Int) =
        CPUFloatMemoryPointer(FloatArray(length))

    override fun allocDouble(array: DoubleArray) =
        CPUDoubleMemoryPointer(array)

    override fun allocDouble(length: Int) =
        CPUDoubleMemoryPointer(DoubleArray(length))

    override fun allocLong(array: LongArray) =
        CPULongMemoryPointer(array)

    override fun allocLong(length: Int) =
        CPULongMemoryPointer(LongArray(length))

    override fun allocInt(array: IntArray) =
        CPUIntMemoryPointer(array)

    override fun allocInt(length: Int) =
        CPUIntMemoryPointer(IntArray(length))

    override fun allocByte(array: ByteArray) =
        CPUByteMemoryPointer(array)

    override fun allocByte(length: Int) =
        CPUByteMemoryPointer(ByteArray(length))
}