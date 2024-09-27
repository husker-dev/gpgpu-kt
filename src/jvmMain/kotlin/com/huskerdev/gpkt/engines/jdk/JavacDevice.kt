package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.Scope
import com.huskerdev.gpkt.engines.cpu.*

class JavacDevice: GPDevice(GPType.Javac) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun compile(ast: Scope) =
        JavacProgram(ast)

    override fun allocFloat(array: FloatArray) =
        CPUFloatMemoryPointer(array.clone())

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