package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement

abstract class InterpreterContext: GPContext {
    override val modules = GPModules(this)
    override var disposed = false

    override fun compile(ast: ScopeStatement): Program =
        InterpreterProgram(ast)

    override fun dispose() {
        disposed = true
    }
}

open class InterpreterSyncContext(
    override val device: GPDevice
) : InterpreterContext(), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CPUSyncFloatMemoryPointer(this, array.copyOf(), usage)

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CPUSyncFloatMemoryPointer(this, FloatArray(length), usage)

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CPUSyncIntMemoryPointer(this, array.copyOf(), usage)

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CPUSyncIntMemoryPointer(this, IntArray(length), usage)

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CPUSyncByteMemoryPointer(this, array.copyOf(), usage)

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CPUSyncByteMemoryPointer(this, ByteArray(length), usage)
}


open class InterpreterAsyncContext(
    override val device: GPDevice
) : InterpreterContext(), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        CPUAsyncFloatMemoryPointer(this, array.copyOf(), usage)

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        CPUAsyncFloatMemoryPointer(this, FloatArray(length), usage)

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        CPUAsyncIntMemoryPointer(this, array.copyOf(), usage)

    override fun allocInts(length: Int, usage: MemoryUsage) =
        CPUAsyncIntMemoryPointer(this, IntArray(length), usage)

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        CPUAsyncByteMemoryPointer(this, array.copyOf(), usage)

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        CPUAsyncByteMemoryPointer(this, ByteArray(length), usage)
}