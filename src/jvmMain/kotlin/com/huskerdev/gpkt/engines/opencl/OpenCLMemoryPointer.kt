package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer


abstract class OpenCLMemoryPointer<T>(
    private val typeSize: Int,
    private val wrapper: (type: T, offset: Int, length: Int) -> Buffer
): MemoryPointer<T> {
    abstract val cl: OpenCL
    abstract val ptr: Long

    val size: Long
        get() = typeSize.toLong() * length

    override fun dealloc() =
        cl.deallocMemory(ptr)

    protected fun writeImpl(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        cl.write(ptr, wrapper(src, srcOffset, length), dstOffset.toLong() * typeSize)
    }

    protected fun readImpl(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        cl.read(ptr, wrapper(dst, dstOffset, length), srcOffset.toLong() * typeSize)
    }

    abstract class Sync<T>(
        typeSize: Int,
        wrapper: (type: T, offset: Int, length: Int) -> Buffer
    ): OpenCLMemoryPointer<T>(typeSize, wrapper), SyncMemoryPointer<T>{
        override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        typeSize: Int,
        wrapper: (type: T, offset: Int, length: Int) -> Buffer
    ): OpenCLMemoryPointer<T>(typeSize, wrapper), AsyncMemoryPointer<T>{
        override suspend fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }
}

// ===================
//        Sync
// ===================

class CLSyncFloatMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Sync<FloatArray>(
    Float.SIZE_BYTES, FloatBuffer::wrap
), SyncFloatMemoryPointer

class CLSyncDoubleMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Sync<DoubleArray>(
    Double.SIZE_BYTES, DoubleBuffer::wrap
), SyncDoubleMemoryPointer

class CLSyncIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Sync<IntArray>(
    Int.SIZE_BYTES, IntBuffer::wrap
), SyncIntMemoryPointer

class CLSyncByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Sync<ByteArray>(
    Byte.SIZE_BYTES, ByteBuffer::wrap
), SyncByteMemoryPointer

// ===================
//       Async
// ===================

class CLAsyncFloatMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Async<FloatArray>(
    Float.SIZE_BYTES, FloatBuffer::wrap
), AsyncFloatMemoryPointer

class CLAsyncDoubleMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Async<DoubleArray>(
    Double.SIZE_BYTES, DoubleBuffer::wrap
), AsyncDoubleMemoryPointer

class CLAsyncIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Async<IntArray>(
    Int.SIZE_BYTES, IntBuffer::wrap
), AsyncIntMemoryPointer

class CLAsyncByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): OpenCLMemoryPointer.Async<ByteArray>(
    Byte.SIZE_BYTES, ByteBuffer::wrap
), AsyncByteMemoryPointer