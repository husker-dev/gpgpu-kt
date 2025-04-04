package com.huskerdev.gpkt.apis.metal

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope

abstract class MetalContext(
    metalDevice: MetalDevice
): GPContext{
    protected val devicePeer = metalDevice.peer
    override val device = metalDevice

    val commandQueue = mtlNewCommandQueue(metalDevice.peer)

    override val allocated = arrayListOf<GPResource>()

    override val memory = mtlGetDeviceMemory(metalDevice.peer)

    override var released = false
    override val modules = GPModules()

    override fun compile(ast: GPScope) =
        addResource(MetalProgram(this, ast))

    override fun release() {
        if(released) return
        allocated.toList().forEach(GPResource::release)
        mtlRelease(commandQueue)
        released = true
    }

    override fun releaseMemory(memory: MemoryPointer<*>) {
        allocated -= memory
        mtlRelease((memory as MetalMemoryPointer<*>).buffer)
    }

    override fun releaseProgram(program: GPProgram) {
        allocated -= program
        program as MetalProgram
        mtlRelease(program.argumentBuffer)
        mtlRelease(program.argumentEncoder)

        mtlRelease(program.library)
        mtlRelease(program.function)
        mtlRelease(program.pipeline)
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