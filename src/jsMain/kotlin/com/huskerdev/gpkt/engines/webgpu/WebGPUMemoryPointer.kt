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
    private val typeSize: Int,
    private val toArrayBuffer: (T) -> ArrayBuffer,
    private val toTypedArray: (ArrayBuffer) -> ArrayBufferView,
    private val copyInto: (src: T, dst: T, dstOffset: Int) -> Unit
): AsyncMemoryPointer<T> {
    abstract val webgpu: WebGPU
    abstract val gpuBuffer: dynamic

    override fun dealloc() =
        webgpu.dealloc(gpuBuffer)

    override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        val writeBuffer = webgpu.allocWrite(toArrayBuffer(src))
        webgpu.copyBufferToBuffer(writeBuffer, gpuBuffer, srcOffset, dstOffset, length * typeSize)
        webgpu.dealloc(writeBuffer)
    }

    override suspend fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        val readBuffer = webgpu.allocRead(length * typeSize)
        webgpu.copyBufferToBuffer(gpuBuffer, readBuffer, srcOffset, 0, length * typeSize)

        // Set readable
        (readBuffer.mapAsync(js("GPUMapMode.READ")) as Promise<*>).await()

        // Read
        val arrayBuffer = readBuffer.getMappedRange() as ArrayBuffer

        // Convert from ArrayBuffer to typed array
        val result = js("Array").from(toTypedArray(arrayBuffer)) as T

        copyInto(result, dst, dstOffset)
        webgpu.dealloc(readBuffer)
    }
}

class WebGPUFloatMemoryPointer(
    override val webgpu: WebGPU,
    override val length: Int,
    override val usage: MemoryUsage,
    override val gpuBuffer: dynamic,
): WebGPUMemoryPointer<FloatArray>(
    Float.SIZE_BYTES,
    FloatArray::toArrayBuffer,
    ::Float32Array,
    { src, dst, dstOffset -> src.copyInto(dst, destinationOffset = dstOffset) }
), AsyncFloatMemoryPointer

class WebGPUIntMemoryPointer(
    override val webgpu: WebGPU,
    override val length: Int,
    override val usage: MemoryUsage,
    override val gpuBuffer: dynamic,
): WebGPUMemoryPointer<IntArray>(
    Int.SIZE_BYTES,
    IntArray::toArrayBuffer,
    ::Int32Array,
    { src, dst, dstOffset -> src.copyInto(dst, destinationOffset = dstOffset) }
), AsyncIntMemoryPointer

