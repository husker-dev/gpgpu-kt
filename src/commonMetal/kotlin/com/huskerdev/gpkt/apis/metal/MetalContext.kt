package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope

abstract class MetalContext(
    metalDevice: MetalDevice
): GPContext{
    protected val devicePeer = metalDevice.peer
    override val device = metalDevice

    private val commandQueue = mtlNewCommandQueue(metalDevice.peer)
    val commandBuffer = mtlNewCommandBuffer(commandQueue)

    override val allocated = arrayListOf<GPResource>()

    override var released = false
    override val modules = GPModules()

    override fun compile(ast: GPScope) =
        addResource(MetalProgram(this, ast))

    override fun release() {
        if(released) return
        allocated.toList().forEach(GPResource::release)
        mtlDeallocCommandQueue(commandQueue)
        mtlDeallocCommandBuffer(commandBuffer)
        released = true
    }

    override fun releaseMemory(memory: MemoryPointer<*>) {
        allocated -= memory
        mtlDeallocBuffer((memory as MetalMemoryPointer<*>).buffer)
    }

    override fun releaseProgram(program: GPProgram) {
        allocated -= program
        program as MetalProgram
        mtlDeallocLibrary(program.library)
        mtlDeallocFunction(program.function)
        mtlDeallocPipeline(program.pipeline)
        mtlDeallocCommandEncoder(program.commandEncoder)
    }

    protected fun <T: GPResource> addResource(memory: T): T{
        allocated += memory
        return memory
    }
}

class MetalSyncContext(
    metalDevice: MetalDevice
): MetalContext(metalDevice), GPSyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(MetalSyncFloatMemoryPointer(this, array.size, usage, mtlWrapFloats(devicePeer, array)))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(MetalSyncFloatMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Float.SIZE_BYTES)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(MetalSyncIntMemoryPointer(this, array.size, usage, mtlWrapInts(devicePeer, array)))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(MetalSyncIntMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Int.SIZE_BYTES)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(MetalSyncByteMemoryPointer(this, array.size, usage, mtlWrapBytes(devicePeer, array)))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(MetalSyncByteMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length)))
}

class MetalAsyncContext(
    metalDevice: MetalDevice
): MetalContext(metalDevice), GPAsyncContext {
    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(MetalAsyncFloatMemoryPointer(this, array.size, usage, mtlWrapFloats(devicePeer, array)))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(MetalAsyncFloatMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Float.SIZE_BYTES)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(MetalAsyncIntMemoryPointer(this, array.size, usage, mtlWrapInts(devicePeer, array)))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(MetalAsyncIntMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length * Int.SIZE_BYTES)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage) =
        addResource(MetalAsyncByteMemoryPointer(this, array.size, usage, mtlWrapBytes(devicePeer, array)))

    override fun allocBytes(length: Int, usage: MemoryUsage) =
        addResource(MetalAsyncByteMemoryPointer(this, length, usage, mtlCreateBuffer(devicePeer, length)))

}