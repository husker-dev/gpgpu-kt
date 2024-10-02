package com.huskerdev.gpkt.engines.webgpu

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.ast.ScopeStatement
import com.huskerdev.gpkt.utils.toArrayBuffer

class WebGPUAsyncDevice(
    private val webgpu: WebGPU
): GPAsyncDevice(GPType.WebGPU) {
    companion object {
        suspend fun create() =
            WebGPUAsyncDevice(WebGPU.create())
    }

    override val id = 0
    override val name = webgpu.name
    override val isGPU = true


    override fun compile(ast: ScopeStatement) =
        WebGPUProgram(webgpu, ast)

    override fun allocFloat(array: FloatArray, usage: MemoryUsage) =
        WebGPUFloatMemoryPointer(webgpu, array.size, usage, webgpu.alloc(array.toArrayBuffer()))

    override fun allocFloat(length: Int, usage: MemoryUsage) =
        WebGPUFloatMemoryPointer(webgpu, length, usage, webgpu.alloc(length * Float.SIZE_BYTES))

    override fun allocDouble(array: DoubleArray, usage: MemoryUsage): AsyncDoubleMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocDouble(length: Int, usage: MemoryUsage): AsyncDoubleMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocLong(array: LongArray, usage: MemoryUsage): AsyncLongMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocLong(length: Int, usage: MemoryUsage): AsyncLongMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocInt(array: IntArray, usage: MemoryUsage) =
        WebGPUIntMemoryPointer(webgpu, array.size, usage, webgpu.alloc(array.toArrayBuffer()))

    override fun allocInt(length: Int, usage: MemoryUsage) =
        WebGPUIntMemoryPointer(webgpu, length, usage, webgpu.alloc(length * Int.SIZE_BYTES))

    override fun allocByte(array: ByteArray, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

    override fun allocByte(length: Int, usage: MemoryUsage): AsyncByteMemoryPointer {
        TODO("Not yet implemented")
    }

}