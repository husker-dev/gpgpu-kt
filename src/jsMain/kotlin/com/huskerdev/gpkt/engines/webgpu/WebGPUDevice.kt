package com.huskerdev.gpkt.engines.webgpu

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.utils.toArrayBuffer

class WebGPUAsyncDevice(
    private val webgpu: WebGPU
): GPAsyncDevice {
    companion object {
        suspend fun create() =
            WebGPUAsyncDevice(WebGPU.create())
    }

    override val type = GPType.WebGPU
    override val id = 0
    override val name = webgpu.name
    override val isGPU = true
    override val modules = GPModules(this)


    override fun compile(ast: ScopeStatement) =
        WebGPUProgram(webgpu, ast)

    override fun wrapFloats(array: FloatArray, usage: MemoryUsage) =
        WebGPUFloatMemoryPointer(webgpu, array.size, usage, webgpu.alloc(array.toArrayBuffer()))

    override fun allocFloats(length: Int, usage: MemoryUsage) =
        WebGPUFloatMemoryPointer(webgpu, length, usage, webgpu.alloc(length * Float.SIZE_BYTES))

    override fun wrapInts(array: IntArray, usage: MemoryUsage) =
        WebGPUIntMemoryPointer(webgpu, array.size, usage, webgpu.alloc(array.toArrayBuffer()))

    override fun allocInts(length: Int, usage: MemoryUsage) =
        WebGPUIntMemoryPointer(webgpu, length, usage, webgpu.alloc(length * Int.SIZE_BYTES))

    override fun wrapBytes(array: ByteArray, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocBytes(length: Int, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

}