package com.huskerdev.gpkt.engines.cpu

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement

abstract class CPUDeviceBase: GPDeviceBase {
    override val type = GPType.Interpreter
    override val id = 0
    override val name = "CPU"
    override val isGPU = false
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement): Program =
        CPUProgram(ast)
}

open class CPUSyncDevice: CPUDeviceBase(), GPSyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CPUSyncFloatMemoryPointer(array.copyOf(), usage)

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CPUSyncFloatMemoryPointer(FloatArray(length), usage)

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        CPUSyncDoubleMemoryPointer(array.copyOf(), usage)

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        CPUSyncDoubleMemoryPointer(DoubleArray(length), usage)

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CPUSyncIntMemoryPointer(array.copyOf(), usage)

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CPUSyncIntMemoryPointer(IntArray(length), usage)

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CPUSyncByteMemoryPointer(array.copyOf(), usage)

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CPUSyncByteMemoryPointer(ByteArray(length), usage)
}


open class CPUAsyncDevice: CPUDeviceBase(), GPAsyncDevice {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CPUAsyncFloatMemoryPointer(array.copyOf(), usage)

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CPUAsyncFloatMemoryPointer(FloatArray(length), usage)

    override fun wrapDoubles(array: DoubleArray, usage: MemoryUsage) =
        CPUAsyncDoubleMemoryPointer(array.copyOf(), usage)

    override fun allocDoubles(length: Int, usage: MemoryUsage) =
        CPUAsyncDoubleMemoryPointer(DoubleArray(length), usage)

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CPUAsyncIntMemoryPointer(array.copyOf(), usage)

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CPUAsyncIntMemoryPointer(IntArray(length), usage)

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CPUAsyncByteMemoryPointer(array.copyOf(), usage)

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CPUAsyncByteMemoryPointer(ByteArray(length), usage)
}