package com.huskerdev.gpkt.engines.webgpu

import com.huskerdev.gpkt.AsyncFloatMemoryPointer
import com.huskerdev.gpkt.AsyncIntMemoryPointer
import com.huskerdev.gpkt.AsyncMemoryPointer
import com.huskerdev.gpkt.MemoryUsage
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
    abstract val webgpu: WebGPU
    abstract val gpuBuffer: dynamic

    override fun dealloc() =
        webgpu.dealloc(gpuBuffer)

    override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        val buffer = toArrayBuffer(src)
        val writeBuffer = webgpu.allocWrite(buffer)
        webgpu.copyBufferToBuffer(writeBuffer, gpuBuffer, srcOffset, dstOffset, gpuBuffer.size as Int)
        webgpu.dealloc(writeBuffer)
    }

    override suspend fun read(length: Int, offset: Int): T {
        val size = gpuBuffer.size as Int
        val readBuffer = webgpu.allocRead(size)
        webgpu.copyBufferToBuffer(gpuBuffer, readBuffer, offset, 0, size)

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
    override val webgpu: WebGPU,
    override val length: Int,
    override val usage: MemoryUsage,
    override val gpuBuffer: dynamic,
): WebGPUMemoryPointer<FloatArray>(
    FloatArray::toArrayBuffer,
    ::Float32Array
), AsyncFloatMemoryPointer

class WebGPUIntMemoryPointer(
    override val webgpu: WebGPU,
    override val length: Int,
    override val usage: MemoryUsage,
    override val gpuBuffer: dynamic,
): WebGPUMemoryPointer<IntArray>(
    IntArray::toArrayBuffer,
    ::Int32Array
), AsyncIntMemoryPointer

