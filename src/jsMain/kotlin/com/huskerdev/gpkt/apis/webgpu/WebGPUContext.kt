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

    override var disposed = false
    override val modules = GPModules()

    override fun dispose() {
        if(disposed) return
        disposed = true
        webgpu.dispose(devicePeer)
    }

    fun flush(){
        commandEncoder = webgpu.flush(devicePeer, commandEncoder)
    }

    override fun compile(ast: GPScope) =
        WebGPUProgram(this, ast)

    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        WebGPUFloatMemoryPointer(this, array.size, usage, webgpu.alloc(devicePeer, array.toArrayBuffer()))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        WebGPUFloatMemoryPointer(this, length, usage, webgpu.alloc(devicePeer, length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        WebGPUIntMemoryPointer(this, array.size, usage, webgpu.alloc(devicePeer, array.toArrayBuffer()))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        WebGPUIntMemoryPointer(this, length, usage, webgpu.alloc(devicePeer, length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocBytes(length: Int, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

}