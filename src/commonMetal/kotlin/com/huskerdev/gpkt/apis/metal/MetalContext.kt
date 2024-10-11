package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement

abstract class MetalContext(
    metalDevice: MetalDevice
): GPContext{
    val metal = metalDevice.metal
    override val device = metalDevice

    private val commandQueue = metal.newCommandQueue(metalDevice.peer)
    val commandBuffer = metal.newCommandBuffer(commandQueue)

    override var disposed = false
    override val modules = GPModules(this)

    override fun compile(ast: ScopeStatement) =
        MetalProgram(this, ast)

    override fun dispose() {
        if(disposed) return
        disposed = true
        metal.deallocCommandQueue(commandQueue)
        metal.deallocCommandBuffer(commandBuffer)
    }
}

class MetalSyncContext(
    metalDevice: MetalDevice
): MetalContext(metalDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        MetalSyncFloatMemoryPointer(this, array.size, usage, metal.wrapFloats(array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        MetalSyncFloatMemoryPointer(this, length, usage, metal.createBuffer(length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        MetalSyncIntMemoryPointer(this, array.size, usage, metal.wrapInts(array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        MetalSyncIntMemoryPointer(this, length, usage, metal.createBuffer(length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        MetalSyncByteMemoryPointer(this, array.size, usage, metal.wrapBytes(array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        MetalSyncByteMemoryPointer(this, length, usage, metal.createBuffer(length))
}

class MetalAsyncContext(
    metalDevice: MetalDevice
): MetalContext(metalDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        MetalAsyncFloatMemoryPointer(this, array.size, usage, metal.wrapFloats(array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        MetalAsyncFloatMemoryPointer(this, length, usage, metal.createBuffer(length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        MetalAsyncIntMemoryPointer(this, array.size, usage, metal.wrapInts(array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        MetalAsyncIntMemoryPointer(this, length, usage, metal.createBuffer(length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        MetalAsyncByteMemoryPointer(this, array.size, usage, metal.wrapBytes(array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        MetalAsyncByteMemoryPointer(this, length, usage, metal.createBuffer(length))

}