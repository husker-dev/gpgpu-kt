package com.huskerdev.gpkt.engines.opencl

import com.huskerdev.gpkt.*
import org.jocl.Pointer
import org.jocl.Sizeof
import org.jocl.cl_mem


abstract class OpenCLMemoryPointer<T>(
    private val typeSize: Int,
    private val wrapper: (T) -> Pointer
): MemoryPointer<T> {
    abstract val cl: OpenCL
    abstract val ptr: cl_mem

    override fun dealloc() =
        cl.dealloc(ptr)

    protected fun writeImpl(src: T, length: Int, srcOffset: Int, dstOffset: Int) {
        cl.write(ptr, wrapper(src),
            size = length.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize
        )
    }

    protected fun readImpl(dst: T, length: Int, dstOffset: Int, srcOffset: Int) {
        cl.read(ptr, wrapper(dst),
            size = length.toLong() * typeSize,
            dstOffset = dstOffset.toLong() * typeSize,
            srcOffset = srcOffset.toLong() * typeSize
        )
    }

    abstract class Sync<T>(
        typeSize: Int,
        wrapper: (T) -> Pointer
    ): OpenCLMemoryPointer<T>(typeSize, wrapper), SyncMemoryPointer<T>{
        override fun read(dst: T, length: Int, dstOffset: Int, srcOffset: Int) =
            readImpl(dst, length, dstOffset, srcOffset)
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)
    }

    abstract class Async<T>(
        typeSize: Int,
        wrapper: (T) -> Pointer
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
    override val ptr: cl_mem
): OpenCLMemoryPointer.Sync<FloatArray>(
    Sizeof.cl_float, Pointer::to
), SyncFloatMemoryPointer

class CLSyncDoubleMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Sync<DoubleArray>(
    Sizeof.cl_double, Pointer::to
), SyncDoubleMemoryPointer

class CLSyncLongMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Sync<LongArray>(
    Sizeof.cl_long, Pointer::to
), SyncLongMemoryPointer

class CLSyncIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Sync<IntArray>(
    Sizeof.cl_int, Pointer::to
), SyncIntMemoryPointer

class CLSyncByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Sync<ByteArray>(
    Sizeof.cl_char, Pointer::to
), SyncByteMemoryPointer

// ===================
//       Async
// ===================

class CLAsyncFloatMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Async<FloatArray>(
    Sizeof.cl_float, Pointer::to
), AsyncFloatMemoryPointer

class CLAsyncDoubleMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Async<DoubleArray>(
    Sizeof.cl_double, Pointer::to
), AsyncDoubleMemoryPointer

class CLAsyncLongMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Async<LongArray>(
    Sizeof.cl_long, Pointer::to
), AsyncLongMemoryPointer

class CLAsyncIntMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Async<IntArray>(
    Sizeof.cl_int, Pointer::to
), AsyncIntMemoryPointer

class CLAsyncByteMemoryPointer(
    override val cl: OpenCL,
    override val length: Int,
    override val usage: MemoryUsage,
    override val ptr: cl_mem
): OpenCLMemoryPointer.Async<ByteArray>(
    Sizeof.cl_char, Pointer::to
), AsyncByteMemoryPointer