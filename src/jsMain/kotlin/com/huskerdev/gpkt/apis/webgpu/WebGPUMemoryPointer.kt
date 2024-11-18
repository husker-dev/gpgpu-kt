package com.huskerdev.gpkt.apis.webgpu

import com.huskerdev.gpkt.*
import com.huskerdev.gpkt.utils.await
import com.huskerdev.gpkt.utils.toArrayBuffer
import org.khronos.webgl.ArrayBuffer
import org.khronos.webgl.ArrayBufferView
import org.khronos.webgl.Float32Array
import org.khronos.webgl.Int32Array
import kotlin.js.Promise


abstract class WebGPUMemoryPointer<T>(
    private val toArrayBuffer: (T) -> ArrayBuffer,
    private val toTypedArray: (ArrayBuffer) -> ArrayBufferView
): AsyncMemoryPointer<T> {
    abstract override val context: WebGPUAsyncContext
    abstract val webgpu: WebGPU

    abstract val gpuBuffer: dynamic
    override var released = false
        get() = field || context.released

    override fun release() {
        if(released) return
        released = true
        context.releaseMemory(this)
    }

    override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        assertNotReleased()

        val buffer = toArrayBuffer(src)
        val writeBuffer = webgpu.allocWrite(context.devicePeer, buffer)
        webgpu.copyBufferToBuffer(context.commandEncoder, writeBuffer, gpuBuffer, srcOffset, dstOffset, gpuBuffer.size as Int)
        context.flush()
        webgpu.dealloc(writeBuffer)
    }

    override suspend fun read(length: Int, offset: Int): T {
        assertNotReleased()

        val size = gpuBuffer.size as Int
        val readBuffer = webgpu.allocRead(context.devicePeer, size)
        webgpu.copyBufferToBuffer(context.commandEncoder, gpuBuffer, readBuffer, offset, 0, size)
        context.flush()

        // Set readable
        (readBuffer.mapAsync(js("GPUMapMode.READ")) as Promise<*>).await()

        // Read
        val arrayBuffer = readBuffer.getMappedRange() as ArrayBuffer

        // Convert from ArrayBuffer to typed array
        val result = js("Array").from(toTypedArray(arrayBuffer)) as T

        webgpu.dealloc(readBuffer)
        return result
    }
}

class WebGPUFloatMemoryPointer(
    override val context: WebGPUAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val gpuBuffer: dynamic,
    override val webgpu: WebGPU = context.webgpu
): WebGPUMemoryPointer<FloatArray>(
    FloatArray::toArrayBuffer,
    ::Float32Array
), AsyncFloatMemoryPointer

class WebGPUIntMemoryPointer(
    override val context: WebGPUAsyncContext,
    override val length: Int,
    override val usage: MemoryUsage,
    override val gpuBuffer: dynamic,
    override val webgpu: WebGPU = context.webgpu
): WebGPUMemoryPointer<IntArray>(
    IntArray::toArrayBuffer,
    ::Int32Array
), AsyncIntMemoryPointer

