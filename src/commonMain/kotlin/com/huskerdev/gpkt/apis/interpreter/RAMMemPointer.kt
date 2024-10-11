package com.huskerdev.gpkt.apis.interpreter

import com.huskerdev.gpkt.*

private typealias MemoryCopier<T> = (src: T, dst: T, dstOffset: Int, startIndex: Int, endIndex: Int) -> Unit
private typealias MemoryAllocator<T> = (length: Int) -> T

abstract class CPUMemoryPointer<T>(
    override val length: Int,
    val copyInto: MemoryCopier<T>,
    val allocator: MemoryAllocator<T>
): MemoryPointer<T>{
    override var disposed = false
        get() = field || context.disposed

    abstract var array: T?

    override fun dealloc() {
        if(disposed) return
        disposed = true
        array = null
    }

    protected fun writeImpl(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
        copyInto(src, array!!, dstOffset, srcOffset, srcOffset + length)

    protected fun readImpl(length: Int, offset: Int) = allocator(length).apply {
        copyInto(array!!, this, 0, offset, offset + length)
    }

    abstract class Sync<T>(
        length: Int,
        copyInto: MemoryCopier<T>,
        allocator: MemoryAllocator<T>
    ): CPUMemoryPointer<T>(length, copyInto, allocator), SyncMemoryPointer<T>{
        override fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)

        override fun read(length: Int, offset: Int) =
            readImpl(length, offset)
    }

    abstract class Async<T>(
        length: Int,
        copyInto: MemoryCopier<T>,
        allocator: MemoryAllocator<T>
    ): CPUMemoryPointer<T>(length, copyInto, allocator), AsyncMemoryPointer<T>{
        override suspend fun write(src: T, length: Int, srcOffset: Int, dstOffset: Int) =
            writeImpl(src, length, srcOffset, dstOffset)

        override suspend fun read(length: Int, offset: Int) =
            readImpl(length, offset)
    }
}

// ===================
//        Sync
// ===================

class CPUSyncFloatMemoryPointer(
    override val context: GPSyncContext,
    override var array: FloatArray?,
    override val usage: MemoryUsage,
): CPUMemoryPointer.Sync<FloatArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) },
    ::FloatArray
), SyncFloatMemoryPointer

class CPUSyncIntMemoryPointer(
    override val context: GPSyncContext,
    override var array: IntArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Sync<IntArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) },
    ::IntArray
), SyncIntMemoryPointer

class CPUSyncByteMemoryPointer(
    override val context: GPSyncContext,
    override var array: ByteArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Sync<ByteArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) },
    ::ByteArray
), SyncByteMemoryPointer

// ===================
//       Async
// ===================

class CPUAsyncFloatMemoryPointer(
    override val context: GPAsyncContext,
    override var array: FloatArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<FloatArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) },
    ::FloatArray
), AsyncFloatMemoryPointer

class CPUAsyncIntMemoryPointer(
    override val context: GPAsyncContext,
    override var array: IntArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<IntArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) },
    ::IntArray
), AsyncIntMemoryPointer

class CPUAsyncByteMemoryPointer(
    override val context: GPAsyncContext,
    override var array: ByteArray?,
    override val usage: MemoryUsage
): CPUMemoryPointer.Async<ByteArray>(
    array!!.size,
    { src, dst, dstOffset, startIndex, endIndex -> src.copyInto(dst, dstOffset, startIndex, endIndex) },
    ::ByteArray
), AsyncByteMemoryPointer