package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.ast.objects.GPScope

abstract class MetalContext(
    metalDevice: MetalDevice
): GPContext{
    protected val devicePeer = metalDevice.peer
    override val device = metalDevice

    private val commandQueue = mtlNewCommandQueue(metalDevice.peer)
    val commandBuffer = mtlNewCommandBuffer(commandQueue)

    override var disposed = false
    override val modules = GPModules()

    override fun compile(ast: GPScope) =
        MetalProgram(this, ast)

    override fun dispose() {
        if(disposed) return
        disposed = true
        mtlDeallocCommandQueue(commandQueue)
        mtlDeallocCommandBuffer(commandBuffer)
    }
}

class MetalSyncContext(
    metalDevice: MetalDevice
): MetalContext(metalDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        MetalSyncFloatMemoryPointer(this, array.size, usage, mtlWrapFloats(devicePeer, array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        MetalSyncFloatMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        MetalSyncIntMemoryPointer(this, array.size, usage, mtlWrapInts(devicePeer, array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        MetalSyncIntMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        MetalSyncByteMemoryPointer(this, array.size, usage, mtlWrapBytes(devicePeer, array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        MetalSyncByteMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length))
}

class MetalAsyncContext(
    metalDevice: MetalDevice
): MetalContext(metalDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        MetalAsyncFloatMemoryPointer(this, array.size, usage, mtlWrapFloats(devicePeer, array))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        MetalAsyncFloatMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        MetalAsyncIntMemoryPointer(this, array.size, usage, mtlWrapInts(devicePeer, array))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        MetalAsyncIntMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        MetalAsyncByteMemoryPointer(this, array.size, usage, mtlWrapBytes(devicePeer, array))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        MetalAsyncByteMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length))

}