package com.huskerdev.gpkt.apis.webgpu

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.objects.GPScope
import com.huskerdev.gpkt.utils.toArrayBuffer

class WebGPUAsyncContext(
    override val device: WebGPUDevice,
    val devicePeer: dynamic,
    var commandEncoder: dynamic
): GPAsyncContext {
    val webgpu = device.webgpu

    override val allocated = arrayListOf<GPResource>()

    override var released = false
    override val modules = GPModules()

    override fun release() {
        if(released) return
        allocated.toList().forEach(GPResource::release)
        webgpu.dispose(devicePeer)
        released = true
    }

    override fun releaseMemory(memory: MemoryPointer<*>) {
        allocated -= memory
        webgpu.dealloc((memory as WebGPUMemoryPointer).gpuBuffer)
    }

    override fun releaseProgram(program: GPProgram) {
        allocated -= program
    }

    private fun <T: GPResource> addResource(memory: T): T{
        allocated += memory
        return memory
    }

    fun flush(){
        commandEncoder = webgpu.flush(devicePeer, commandEncoder)
    }

    override fun compile(ast: GPScope) =
        addResource(WebGPUProgram(this, ast))

    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        addResource(WebGPUFloatMemoryPointer(this, array.size, usage, webgpu.alloc(devicePeer, array.toArrayBuffer())))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        addResource(WebGPUFloatMemoryPointer(this, length, usage, webgpu.alloc(devicePeer, length * Float.SIZE_BYTES)))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        addResource(WebGPUIntMemoryPointer(this, array.size, usage, webgpu.alloc(devicePeer, array.toArrayBuffer())))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        addResource(WebGPUIntMemoryPointer(this, length, usage, webgpu.alloc(devicePeer, length * Int.SIZE_BYTES)))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocBytes(length: Int, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

}