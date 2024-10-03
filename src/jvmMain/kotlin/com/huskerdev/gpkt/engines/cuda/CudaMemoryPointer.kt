package com.huskerdev.gpkt.engines.cuda

import com.huskerdev.gpkt.*
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer


abstract class CudaMemoryPointer<T>(
    private val typeSize: Int,
    private val wrapper: (type: T, offset: Int, length: Int) -> Buffer
): MemoryPointer<T> {
    abstract val cuda: Cuda
    abstract val ptr: Long

    override fun dealloc() = cuda.dealloc(ptr)

    protected fun writeImpl(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        cuda.write(ptr, wrapper(src, srcOffset, length), dstOffset * typeSize)
    }

    protected fun readImpl(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        cuda.read(ptr, wrapper(dst, dstOffset, length), srcOffset * typeSize)
    }

    abstract class Sync<T>(
        typeSize: Int,
        wrapper: (type: T, offset: Int, length: Int) -> Buffer
    ): CudaMemoryPointer<T>(typeSize, wrapper), SyncMemoryPointer<T>{
        override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        typeSize: Int,
        wrapper: (type: T, offset: Int, length: Int) -> Buffer
    ): CudaMemoryPointer<T>(typeSize, wrapper), AsyncMemoryPointer<T>{
        override suspend fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }
}

// ===================
//        Sync
// ===================

class CudaSyncFloatMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Sync<FloatArray>(
    Float.SIZE_BYTES, FloatBuffer::wrap
), SyncFloatMemoryPointer

class CudaSyncDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Sync<DoubleArray>(
    Double.SIZE_BYTES, DoubleBuffer::wrap
), SyncDoubleMemoryPointer

class CudaSyncIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Sync<IntArray>(
    Int.SIZE_BYTES, IntBuffer::wrap
), SyncIntMemoryPointer

class CudaSyncByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Sync<ByteArray>(
    Byte.SIZE_BYTES, ByteBuffer::wrap
), SyncByteMemoryPointer


// ===================
//       Async
// ===================

class CudaAsyncFloatMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Async<FloatArray>(
    Float.SIZE_BYTES, FloatBuffer::wrap
), AsyncFloatMemoryPointer

class CudaAsyncDoubleMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Async<DoubleArray>(
    Double.SIZE_BYTES, DoubleBuffer::wrap
), AsyncDoubleMemoryPointer

class CudaAsyncIntMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Async<IntArray>(
    Int.SIZE_BYTES, IntBuffer::wrap
), AsyncIntMemoryPointer

class CudaAsyncByteMemoryPointer(
    override val cuda: Cuda,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: Long
): CudaMemoryPointer.Async<ByteArray>(
    Byte.SIZE_BYTES, ByteBuffer::wrap
), AsyncByteMemoryPointer