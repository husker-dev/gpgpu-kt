package com.huskerdev.gpkt.engines.jdk

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.engines.cpu.*

class JavacDevice: GPDevice(GPType.Javac) {
    override val id = 0
    override val name = "CPU"
    override val isGPU = false

    override fun compile(ast: ScopeStatement) =
        JavacProgram(ast)

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        CPUFloatMemoryPointer(array.copyOf(), usage)

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        CPUFloatMemoryPointer(FloatArray(length), usage)

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage) =
        CPUDoubleMemoryPointer(array.copyOf(), usage)

    override fun allocDouble(length: Int, usage: MemoryUsage) =
        CPUDoubleMemoryPointer(DoubleArray(length), usage)

    override fun allocLong(array: LongArray, usage: MemoryUsage) =
        CPULongMemoryPointer(array.copyOf(), usage)

    override fun allocLong(length: Int, usage: MemoryUsage) =
        CPULongMemoryPointer(LongArray(length), usage)

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        CPUIntMemoryPointer(array.copyOf(), usage)

    override fun allocInt(length: Int, usage: MemoryUsage) =
        CPUIntMemoryPointer(IntArray(length), usage)

    override fun allocByte(array: ByteArray, usage: MemoryUsage) =
        CPUByteMemoryPointer(array.copyOf(), usage)

    override fun allocByte(length: Int, usage: MemoryUsage) =
        CPUByteMemoryPointer(ByteArray(length), usage)
}