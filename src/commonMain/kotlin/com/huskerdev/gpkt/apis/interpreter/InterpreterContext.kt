package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.ram

abstract class InterpreterContext: GPContext {
    override val modules = GPModules()
    override var released = false

    override val memory = ram

    override val allocated = arrayListOf<GPResource>()

    override fun compile(ast: GPScope): GPProgram =
        addResource(InterpreterProgram(this, ast))

    override fun release() {
        if(released) return
        allocated.toList().forEach(GPResource::release)
        released = true
    }

    override fun releaseMemory(memory: MemoryPointer<*>) {
        allocated -= memory
        (memory as CPUMemoryPointer).array = null
    }

    override fun releaseProgram(program: GPProgram) {
        allocated -= program
    }

    protected fun <T: GPResource> addResource(memory: T): T{
        allocated += memory
        return memory
    }
}

open class InterpreterSyncContext(
    override val device: GPDevice
) : InterpreterContext(), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(CPUSyncFloatMemoryPointer(this, array.copyOf(), usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(CPUSyncFloatMemoryPointer(this, FloatArray(length), usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(CPUSyncIntMemoryPointer(this, array.copyOf(), usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(CPUSyncIntMemoryPointer(this, IntArray(length), usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(CPUSyncByteMemoryPointer(this, array.copyOf(), usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(CPUSyncByteMemoryPointer(this, ByteArray(length), usage))
}


open class InterpreterAsyncContext(
    override val device: GPDevice
) : InterpreterContext(), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(CPUAsyncFloatMemoryPointer(this, array.copyOf(), usage))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(CPUAsyncFloatMemoryPointer(this, FloatArray(length), usage))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(CPUAsyncIntMemoryPointer(this, array.copyOf(), usage))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(CPUAsyncIntMemoryPointer(this, IntArray(length), usage))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(CPUAsyncByteMemoryPointer(this, array.copyOf(), usage))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(CPUAsyncByteMemoryPointer(this, ByteArray(length), usage))
}